"""GlinerClaimFilter: GLiNER2.5-decide decides the kind of statement (ONNX, CPU; `fact-assessor[gliner]`).

Shares the process-wide model from factassessor.gliner with `GlinerJudge`. Claim benchmark: 17/19 like Laya, but ~10x
slower, and its misses drop real claims (Laya's keep opinions).
"""

from __future__ import annotations

from factassessor.claim_filters._base import KINDS, ClaimFilter
from factassessor.gliner import GlinerModel, gliner_model
from factassessor.schema import Atom

KIND = {"task": "kind", "instruction": "What kind of statement is this?", "labels": KINDS}


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
