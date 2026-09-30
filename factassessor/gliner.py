"""GlinerRunner: GLiNER2.5-decide as a `DecisionRunner`, via ONNX (onnxruntime + tokenizers + numpy, no torch).

GLiNER2.5-decide (fastino, Apache-2.0) is, like Laya, a non-autoregressive decision model: a task with labels in,
one probability per label out, in a single encoder pass. Weights: the ONNX export at
https://huggingface.co/nishparadox/gliner2.5-decide-onnx (`model="2.5-decide"`). Install `fact-assessor[gliner]`.
Every runner shares one loaded model per (model, variant) for the whole process (`gliner_model`).

The input encoding reimplements that repo's `gliner_onnx.py` (which rebuilds gliner2's processor): the task prompt
`( [P] "{task}: {instruction} [DESCRIPTION] label: desc ..." ( [L] l1 [L] l2 ... ) ) [SEP_TEXT] <words>`, each
piece tokenized on its own, logits read at the `[L]` markers. We load the weights and tokenizer, not its code.

Judge benchmark (15 cases, M-series Mac, CPU): fp32 12/15 at ~118ms/pair (Laya: 13/15), int8 7-8/15 at ~55ms/pair
(too lossy), CoreML slower than CPU (only ~1/3 of the graph runs on it). End to end on web passages it was far
weaker than Laya (data/results/).
"""

from __future__ import annotations

import asyncio
import os
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np

from factassessor.decisions import Answer, DecisionRequest, DecisionResponse, DecisionRunner

MODELS = {"2.5-decide": "nishparadox/gliner2.5-decide-onnx"}  # short names -> Hugging Face repos
VARIANTS = {"fp32": "model.onnx", "fp16": "model_fp16.onnx", "int8": "model_int8.onnx"}


# gliner2's whitespace word splitter: URLs, emails, @handles, words (with - or _), then any other character
_WORDS = re.compile(
    r"""(?:https?://[^\s]+|www\.[^\s]+)
    |[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}
    |@[a-z0-9_]+
    |\w+(?:[-_]\w+)*
    |\S""",
    re.VERBOSE | re.IGNORECASE,
)


class GlinerModel:
    """One loaded ONNX session + tokenizer, called from `workers` threads at once. Get it with `gliner_model()`.

    The model is CPU-bound (~1.2ms per token, ~250ms per page passage), and one caller at a time left cores idle:
    2 callers x 7 intra-op threads did 7.4 rows/s vs 5.8 for one caller (M3 Max, 14 cores; 14 x 1 was 7.5 but
    runs 14 inferences' worth of buffers at once). `threads` defaults to the cores split between the workers.
    The ONNX session is thread-safe, so the workers share one copy of the weights.
    """

    def __init__(self, repo: str, variant: str = "fp32", threads: int | None = None, workers: int = 2) -> None:
        self.repo, self.variant, self.workers = repo, variant, workers
        self.threads = threads or max(1, (os.cpu_count() or workers) // workers)
        self.session: Any = None
        self.tok: Any = None
        self.thread = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="gliner")
        self._load_lock = threading.Lock()
        self._tok_lock = threading.Lock()  # HF fast tokenizers aren't thread-safe
        self._piece_ids: dict[str, list[int]] = {}
        self._prompts: dict[str, tuple[list[int], list[int]]] = {}

    async def aload(self) -> None:
        await asyncio.get_running_loop().run_in_executor(self.thread, self.load)

    def load(self) -> None:
        with self._load_lock:
            if self.session is not None:
                return
            import onnxruntime as ort
            from huggingface_hub import hf_hub_download
            from tokenizers import Tokenizer

            model_path = hf_hub_download(self.repo, VARIANTS[self.variant])
            self.tok = Tokenizer.from_file(hf_hub_download(self.repo, "tokenizer.json"))
            options = ort.SessionOptions()
            options.intra_op_num_threads = self.threads
            self.session = ort.InferenceSession(model_path, options, providers=["CPUExecutionProvider"])

    async def probabilities(self, task: dict[str, Any], texts: list[str], batch_size: int = 16) -> list[dict[str, float]]:
        await self.aload()
        return await asyncio.get_running_loop().run_in_executor(self.thread, self._probabilities, task, texts, batch_size)

    def _probabilities(self, task: dict[str, Any], texts: list[str], batch_size: int) -> list[dict[str, float]]:
        prompt_ids, positions = self._prompt(task)
        results: list[dict[str, float]] = []
        for start in range(0, len(texts), batch_size):
            rows = [self._encode_row(prompt_ids, t) for t in texts[start : start + batch_size]]
            width = max(len(r) for r in rows)
            input_ids = np.zeros((len(rows), width), dtype=np.int64)
            attention = np.zeros((len(rows), width), dtype=np.int64)
            for i, row in enumerate(rows):
                input_ids[i, : len(row)] = row
                attention[i, : len(row)] = 1
            (logits,) = self.session.run(
                ["logits"],
                {"input_ids": input_ids, "attention_mask": attention, "label_positions": np.asarray([positions] * len(rows), dtype=np.int64)},
            )
            for row_logits in np.asarray(logits):
                x = row_logits - row_logits.max()
                p = np.exp(x) / np.exp(x).sum()  # one exclusive label: softmax
                results.append(dict(zip(task["labels"], p.tolist())))
        return results

    def _prompt(self, task: dict[str, Any]) -> tuple[list[int], list[int]]:
        if task["task"] not in self._prompts:
            prompt = f"{task['task']}: {task['instruction']}" + "".join(
                f" [DESCRIPTION] {label}: {desc}" for label, desc in task["labels"].items()
            )
            pieces = ["(", "[P]", prompt, "("]
            for label in task["labels"]:
                pieces += ["[L]", label]
            pieces += [")", ")"]
            ids: list[int] = []
            positions: list[int] = []
            for piece in pieces:
                if piece == "[L]":
                    positions.append(len(ids))
                ids += self._ids(piece)
            self._prompts[task["task"]] = (ids, positions)
        return self._prompts[task["task"]]

    def _encode_row(self, prompt_ids: list[int], text: str) -> list[int]:
        if not text.endswith((".", "!", "?")):
            text += "."
        ids = prompt_ids + self._ids("[SEP_TEXT]")
        for word in _WORDS.finditer(text):
            ids += self._ids(word.group().lower())
        return ids

    def _ids(self, piece: str) -> list[int]:
        if (ids := self._piece_ids.get(piece)) is None:
            with self._tok_lock:
                ids = self._piece_ids[piece] = self.tok.encode(piece, add_special_tokens=False).ids
        return ids


_models: dict[tuple[str, str], GlinerModel] = {}
_models_lock = threading.Lock()


def gliner_model(model: str = "2.5-decide", variant: str = "fp32", threads: int | None = None) -> GlinerModel:
    """The process-wide GLiNER model for (model, variant); loaded on first use."""
    repo = MODELS.get(model, model)
    with _models_lock:
        return _models.setdefault((repo, variant), GlinerModel(repo, variant, threads))


class GlinerRunner(DecisionRunner):
    """GLiNER2.5-decide answering decision requests on the CPU.

    A request's state fields become one text, "field: value" in the state's order (the judge puts the evidence
    first, then the claim: 12/15 vs 11/15 claim-first on the judge benchmark); each `choice` question is one
    GLiNER task (its instructions without the backticks, its criteria as the labels), and a call's rows that share
    a task go through the model together, padded, `batch_size` at a time.

    `concurrency`: claims a judge on this runner takes at once. GLiNER does ~7 rows/s on CPU and a claim needs
    ~10: with no limit, the 20 claims of a long text shared it evenly and all 20 timed out; with a limit none did,
    and 3 gave the first verdict soonest (4.8s vs 7.0s at 5, 9.6s at 8; ~35s in total either way).
    """

    def __init__(
        self,
        model: str = "2.5-decide",  # a short name from MODELS or a Hugging Face repo with the same ONNX export
        variant: str = "fp32",  # int8 is 2x faster but lost ~5/15 on the judge benchmark
        threads: int | None = None,  # onnxruntime intra-op threads
        batch_size: int = 16,
        concurrency: int | None = 3,
    ) -> None:
        self.model, self.variant, self.threads = model, variant, threads
        self.batch_size = batch_size
        self.concurrency = concurrency
        self._model: GlinerModel | None = None  # tests inject a fake

    @property
    def gliner(self) -> GlinerModel:
        if self._model is None:
            self._model = gliner_model(self.model, self.variant, self.threads)
        return self._model

    async def aload(self) -> None:
        """Download (first time, ~1.75 GB for fp32) and load the ONNX model."""
        await self.gliner.aload()

    async def predict(self, requests: list[DecisionRequest]) -> list[DecisionResponse]:
        if not requests:
            return []
        tasks: dict[tuple[Any, ...], tuple[dict[str, Any], list[tuple[int, str, str]]]] = {}  # a task -> its rows
        for i, r in enumerate(requests):
            text = r.state if isinstance(r.state, str) else " ".join(f"{k}: {v}" for k, v in r.state.items())
            for key, q in r.questions.items():
                task = {"task": key, "instruction": q.instructions.replace("`", ""), "labels": q.criteria}
                tasks.setdefault((key, task["instruction"], tuple(q.criteria.items())), (task, []))[1].append((i, key, text))
        answers: list[dict[str, Answer]] = [{} for _ in requests]

        async def run(task: dict[str, Any], rows: list[tuple[int, str, str]]) -> None:
            for (i, key, _), dist in zip(rows, await self.gliner.probabilities(task, [t for _, _, t in rows], self.batch_size)):
                answers[i][key] = Answer(choice=max(dist, key=dist.__getitem__), probabilities=dist)

        await asyncio.gather(*(run(task, rows) for task, rows in tasks.values()))
        return [DecisionResponse(answers=a) for a in answers]
