"""GlinerJudge: the evidence judge on GLiNER2.5-decide, via ONNX (onnxruntime + tokenizers + numpy, no torch).

GLiNER2.5-decide (fastino, Apache-2.0) is, like Laya, a non-autoregressive decision model: a task with labels in,
one probability per label out, in a single encoder pass. Weights: the ONNX export at
https://huggingface.co/nishparadox/gliner2.5-decide-onnx. Install with `fact-assessor[gliner]`.

The input encoding below reimplements that repo's `gliner_onnx.py` (which rebuilds gliner2's processor): the task
prompt `( [P] "stance: {instruction} [DESCRIPTION] label: desc ..." ( [L] l1 [L] l2 ... ) ) [SEP_TEXT] <words>`,
each piece tokenized on its own, logits read at the `[L]` markers. We don't execute the repo's Python, only load
its weights and tokenizer.

Measured on the 15-case judge benchmark (M-series Mac, CPU): fp32 12/15 at ~118ms/pair (Laya: 13/15), int8 7-8/15
at ~55ms/pair (too lossy for judging), CoreML slower than CPU (only ~1/3 of the graph runs on it).
"""

from __future__ import annotations

import asyncio
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np

from factassessor.evidence_judge import Judge
from factassessor.passages import chunk, top_passages
from factassessor.schema import Evidence

DEFAULT_REPO = "nishparadox/gliner2.5-decide-onnx"
VARIANTS = {"fp32": "model.onnx", "fp16": "model_fp16.onnx", "int8": "model_int8.onnx"}

INSTRUCTION = "Does the evidence support or refute the claim?"
LABELS = {  # order matters: it's the order of the logits
    "supports": "the evidence says the same thing as the claim",
    "refutes": "the evidence contradicts the claim",
    "not_enough_info": "the evidence does not mention what the claim is about",
}

# gliner2's whitespace word splitter: URLs, emails, @handles, words (with - or _), then any other character
_WORDS = re.compile(
    r"""(?:https?://[^\s]+|www\.[^\s]+)
    |[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}
    |@[a-z0-9_]+
    |\w+(?:[-_]\w+)*
    |\S""",
    re.VERBOSE | re.IGNORECASE,
)


class GlinerJudge(Judge):
    """Every (evidence, claim) pair is one GLiNER decision; a call's pairs share padded forward passes."""

    def __init__(
        self,
        repo: str = DEFAULT_REPO,
        variant: str = "fp32",  # int8 is 2x faster but lost ~5/15 on the judge benchmark
        threads: int | None = None,  # onnxruntime intra-op threads (default: onnxruntime's choice)
        passages_per_page: int = 1,
        passage_tokens: int = 128,
        batch_size: int = 16,
    ) -> None:
        self.repo = repo
        self.variant = variant
        self.threads = threads
        self.passages_per_page = passages_per_page
        self.passage_tokens = passage_tokens
        self.batch_size = batch_size
        self._session: Any = None
        self._tok: Any = None
        self._prompt_ids: list[int] = []
        self._label_positions: list[int] = []
        self._piece_ids: dict[str, list[int]] = {}
        self._load_lock = asyncio.Lock()
        self._tok_lock = threading.Lock()  # HF fast tokenizers aren't thread-safe
        self._thread = ThreadPoolExecutor(max_workers=1, thread_name_prefix="gliner")  # one session, one thread

    async def aload(self) -> None:
        """Download (first time) and load the ONNX model: ~1.75 GB for fp32."""
        async with self._load_lock:
            if self._session is None:
                await asyncio.get_running_loop().run_in_executor(self._thread, self._load)

    async def judge(self, claim: str, docs: list[dict[str, Any]]) -> list[Evidence]:
        if not docs:
            return []
        await self.aload()
        passages = await asyncio.gather(*(asyncio.to_thread(self._passages_of, claim, d) for d in docs))
        flat = [p for group in passages for p in group]
        if not flat:
            return []
        probs = await asyncio.get_running_loop().run_in_executor(
            self._thread, self._probabilities, claim, [p["text"] for p in flat]
        )
        evidence = []
        for passage, dist in zip(flat, probs):
            label = max(dist, key=dist.get)
            evidence.append(Evidence(**passage, label=label, prob=dist[label]))
        return evidence

    # --- inference (runs on the model thread) -----------------------------------------------------------

    def _probabilities(self, claim: str, texts: list[str]) -> list[dict[str, float]]:
        results: list[dict[str, float]] = []
        for start in range(0, len(texts), self.batch_size):
            rows = [self._encode_row(f"evidence: {t} claim: {claim}") for t in texts[start : start + self.batch_size]]
            width = max(len(r) for r in rows)
            input_ids = np.zeros((len(rows), width), dtype=np.int64)
            attention = np.zeros((len(rows), width), dtype=np.int64)
            for i, row in enumerate(rows):
                input_ids[i, : len(row)] = row
                attention[i, : len(row)] = 1
            positions = np.asarray([self._label_positions] * len(rows), dtype=np.int64)
            (logits,) = self._session.run(
                ["logits"], {"input_ids": input_ids, "attention_mask": attention, "label_positions": positions}
            )
            for row_logits in np.asarray(logits):
                x = row_logits - row_logits.max()
                p = np.exp(x) / np.exp(x).sum()  # one exclusive label: softmax
                results.append(dict(zip(LABELS, p.tolist())))
        return results

    def _encode_prompt(self) -> tuple[list[int], list[int]]:
        prompt = f"stance: {INSTRUCTION}" + "".join(f" [DESCRIPTION] {label}: {desc}" for label, desc in LABELS.items())
        pieces = ["(", "[P]", prompt, "("]
        for label in LABELS:
            pieces += ["[L]", label]
        pieces += [")", ")"]
        ids: list[int] = []
        positions: list[int] = []
        for piece in pieces:
            if piece == "[L]":
                positions.append(len(ids))
            ids += self._ids(piece)
        return ids, positions

    def _encode_row(self, text: str) -> list[int]:
        if not text.endswith((".", "!", "?")):
            text += "."
        ids = self._prompt_ids + self._ids("[SEP_TEXT]")
        for word in _WORDS.finditer(text):
            ids += self._ids(word.group().lower())
        return ids

    def _ids(self, piece: str) -> list[int]:
        if (ids := self._piece_ids.get(piece)) is None:
            with self._tok_lock:
                ids = self._piece_ids[piece] = self._tok.encode(piece, add_special_tokens=False).ids
        return ids

    # --- passages (worker threads) -----------------------------------------------------------------------

    def _passages_of(self, claim: str, doc: dict[str, Any]) -> list[dict[str, Any]]:
        base = {"url": doc["url"], "title": doc.get("title", "")}
        if "text" not in doc:
            return [{**base, "text": doc["snippet"], "source": "snippet"}] if doc.get("snippet") else []
        with self._tok_lock:
            chunks = chunk(doc["text"], _Offsets(self._tok), max_tokens=self.passage_tokens, overlap=self.passage_tokens // 4)
        return [{**base, "text": t, "source": "page"} for t in top_passages(claim, chunks, k=self.passages_per_page)]

    def _load(self) -> None:
        import onnxruntime as ort
        from huggingface_hub import hf_hub_download
        from tokenizers import Tokenizer

        model_path = hf_hub_download(self.repo, VARIANTS[self.variant])
        self._tok = Tokenizer.from_file(hf_hub_download(self.repo, "tokenizer.json"))
        options = ort.SessionOptions()
        if self.threads:
            options.intra_op_num_threads = self.threads
        self._session = ort.InferenceSession(model_path, options, providers=["CPUExecutionProvider"])
        self._prompt_ids, self._label_positions = self._encode_prompt()


class _Offsets:
    """Adapts a `tokenizers.Tokenizer` to the HF-style call `passages.chunk` expects (offset_mapping)."""

    def __init__(self, tok: Any) -> None:
        self.tok = tok

    def __call__(self, text: str, add_special_tokens: bool = False, return_offsets_mapping: bool = False) -> dict[str, Any]:
        encoding = self.tok.encode(text, add_special_tokens=add_special_tokens)
        return {"input_ids": encoding.ids, "offset_mapping": encoding.offsets}
