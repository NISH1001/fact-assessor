"""GlinerJudge: every (evidence, claim) pair is one GLiNER2.5-decide decision (ONNX, CPU; `fact-assessor[gliner]`).

Shares the process-wide model from factassessor.gliner with `GlinerClaimFilter`. Judge benchmark (15 cases): fp32
12/15 at ~118ms/pair (Laya 13/15); end to end on web passages it was far weaker (see data/results/).
"""

from __future__ import annotations

import asyncio
from typing import Any

from factassessor.gliner import GlinerModel, gliner_model
from factassessor.judges._base import Judge
from factassessor.passages import top_passages
from factassessor.schema import Evidence

STANCE = {  # (task, instruction, labels in logit order)
    "task": "stance",
    "instruction": "Does the evidence support or refute the claim?",
    "labels": {
        "supports": "the evidence says the same thing as the claim",
        "refutes": "the evidence contradicts the claim",
        "not_enough_info": "the evidence does not mention what the claim is about",
    },
}
LABELS = STANCE["labels"]


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
        base = {"url": doc["url"], "title": doc.get("title") or ""}
        if "text" not in doc:
            snippet = doc.get("snippet")
            if not snippet:
                return []
            # most snippets are ~120 tokens, but some DuckDuckGo ones run to 3,000+: a batch pads every row to its
            # longest, so one of those made a 5-snippet call 9x slower. Cut those like pages; short ones stay whole.
            chunks = self.gliner.chunk(snippet, self.passage_tokens)
            texts = [snippet] if len(chunks) <= 1 else top_passages(claim, chunks, k=1)
            return [{**base, "text": t, "source": "snippet"} for t in texts]
        chunks = self.gliner.chunk(doc["text"], self.passage_tokens)
        return [{**base, "text": t, "source": "page"} for t in top_passages(claim, chunks, k=self.passages_per_page)]
