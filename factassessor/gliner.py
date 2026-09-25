"""GLiNER2.5-decide via ONNX (onnxruntime + tokenizers + numpy, no torch): `GlinerJudge` and `GlinerClaimFilter`.

GLiNER2.5-decide (fastino, Apache-2.0) is, like Laya, a non-autoregressive decision model: a task with labels in,
one probability per label out, in a single encoder pass. Weights: the ONNX export at
https://huggingface.co/nishparadox/gliner2.5-decide-onnx (`model="2.5-decide"`). Install `fact-assessor[gliner]`.
Both classes share one loaded model per (model, variant) for the whole process.

The input encoding reimplements that repo's `gliner_onnx.py` (which rebuilds gliner2's processor): the task prompt
`( [P] "{task}: {instruction} [DESCRIPTION] label: desc ..." ( [L] l1 [L] l2 ... ) ) [SEP_TEXT] <words>`, each
piece tokenized on its own, logits read at the `[L]` markers. We load the weights and tokenizer, not its code.

Judge benchmark (15 cases, M-series Mac, CPU): fp32 12/15 at ~118ms/pair (Laya: 13/15), int8 7-8/15 at ~55ms/pair
(too lossy), CoreML slower than CPU (only ~1/3 of the graph runs on it).
"""

from __future__ import annotations

import asyncio
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np

from factassessor.claim_filter import KINDS, ClaimFilter
from factassessor.evidence_judge import Judge
from factassessor.passages import chunk, top_passages
from factassessor.schema import Atom, Evidence

MODELS = {"2.5-decide": "nishparadox/gliner2.5-decide-onnx"}  # short names -> Hugging Face repos
VARIANTS = {"fp32": "model.onnx", "fp16": "model_fp16.onnx", "int8": "model_int8.onnx"}

STANCE = {  # (task, instruction, labels in logit order)
    "task": "stance",
    "instruction": "Does the evidence support or refute the claim?",
    "labels": {
        "supports": "the evidence says the same thing as the claim",
        "refutes": "the evidence contradicts the claim",
        "not_enough_info": "the evidence does not mention what the claim is about",
    },
}
KIND = {"task": "kind", "instruction": "What kind of statement is this?", "labels": KINDS}
LABELS = STANCE["labels"]  # kept for callers that list the judge's labels

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
    """One loaded ONNX session + tokenizer, on one thread. Get it with `gliner_model(model, variant)`."""

    def __init__(self, repo: str, variant: str = "fp32", threads: int | None = None) -> None:
        self.repo, self.variant, self.threads = repo, variant, threads
        self.session: Any = None
        self.tok: Any = None
        self.thread = ThreadPoolExecutor(max_workers=1, thread_name_prefix="gliner")
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
            if self.threads:
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

    def chunk(self, text: str, max_tokens: int) -> list[str]:
        with self._tok_lock:
            return chunk(text, _Offsets(self.tok), max_tokens=max_tokens, overlap=max_tokens // 4)


_models: dict[tuple[str, str], GlinerModel] = {}
_models_lock = threading.Lock()


def gliner_model(model: str = "2.5-decide", variant: str = "fp32", threads: int | None = None) -> GlinerModel:
    """The process-wide GLiNER model for (model, variant); loaded on first use."""
    repo = MODELS.get(model, model)
    with _models_lock:
        return _models.setdefault((repo, variant), GlinerModel(repo, variant, threads))


class GlinerJudge(Judge):
    """Every (evidence, claim) pair is one GLiNER decision; a call's pairs share padded forward passes."""

    def __init__(
        self,
        model: str = "2.5-decide",  # a short name from MODELS or a Hugging Face repo with the same ONNX export
        variant: str = "fp32",  # int8 is 2x faster but lost ~5/15 on the judge benchmark
        threads: int | None = None,  # onnxruntime intra-op threads
        passages_per_page: int = 1,
        passage_tokens: int = 128,
        batch_size: int = 16,
    ) -> None:
        self.model, self.variant, self.threads = model, variant, threads
        self.passages_per_page = passages_per_page
        self.passage_tokens = passage_tokens
        self.batch_size = batch_size
        self._model: GlinerModel | None = None  # tests inject a fake

    @property
    def gliner(self) -> GlinerModel:
        if self._model is None:
            self._model = gliner_model(self.model, self.variant, self.threads)
        return self._model

    async def aload(self) -> None:
        """Download (first time, ~1.75 GB for fp32) and load the ONNX model."""
        await self.gliner.aload()

    async def judge(self, claim: str, docs: list[dict[str, Any]]) -> list[Evidence]:
        if not docs:
            return []
        await self.gliner.aload()
        passages = [p for group in await asyncio.gather(*(asyncio.to_thread(self._passages_of, claim, d) for d in docs)) for p in group]
        if not passages:
            return []
        # evidence first, then claim: 12/15 vs 11/15 claim-first on the judge benchmark
        texts = [f"evidence: {p['text']} claim: {claim}" for p in passages]
        evidence = []
        for passage, dist in zip(passages, await self.gliner.probabilities(STANCE, texts, self.batch_size)):
            label = max(dist, key=dist.get)
            evidence.append(Evidence(**passage, label=label, prob=dist[label]))
        return evidence

    def _passages_of(self, claim: str, doc: dict[str, Any]) -> list[dict[str, Any]]:
        base = {"url": doc["url"], "title": doc.get("title", "")}
        if "text" not in doc:
            return [{**base, "text": doc["snippet"], "source": "snippet"}] if doc.get("snippet") else []
        chunks = self.gliner.chunk(doc["text"], self.passage_tokens)
        return [{**base, "text": t, "source": "page"} for t in top_passages(claim, chunks, k=self.passages_per_page)]


class GlinerClaimFilter(ClaimFilter):
    """GLiNER decides the kind of statement (factual claim / opinion / question or request / social)."""

    def __init__(self, threshold: float = 0.4, model: str = "2.5-decide", variant: str = "fp32", threads: int | None = None) -> None:
        super().__init__(threshold)
        self.model, self.variant, self.threads = model, variant, threads
        self._model: GlinerModel | None = None  # tests inject a fake

    @property
    def gliner(self) -> GlinerModel:
        if self._model is None:
            self._model = gliner_model(self.model, self.variant, self.threads)
        return self._model

    async def score(self, atom: Atom) -> float:
        [dist] = await self.gliner.probabilities(KIND, [atom.text])
        return dist["factual_claim"]

    async def start(self) -> None:
        await self.gliner.aload()


class _Offsets:
    """Adapts a `tokenizers.Tokenizer` to the HF-style call `passages.chunk` expects (offset_mapping)."""

    def __init__(self, tok: Any) -> None:
        self.tok = tok

    def __call__(self, text: str, add_special_tokens: bool = False, return_offsets_mapping: bool = False) -> dict[str, Any]:
        encoding = self.tok.encode(text, add_special_tokens=add_special_tokens)
        return {"input_ids": encoding.ids, "offset_mapping": encoding.offsets}
