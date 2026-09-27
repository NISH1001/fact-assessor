"""LayaClaimFilter: the local Laya model decides the kind of statement (the default claim filter)."""

from __future__ import annotations

from typing import Any

from factassessor.claim_filters._base import KINDS, ClaimFilter
from factassessor.laya import laya_runner
from factassessor.schema import Atom

# `choice` beat `noul` clearly in our tests (17/19 vs 8/12 correct): Laya's yes/no head was near-random here.
LAYA_QUESTION = {"kind": {"type": "choice", "instructions": "What kind of statement is `claim`?", "criteria": KINDS}}


class LayaClaimFilter(ClaimFilter):
    """Laya decides the kind of statement (17/19 on data/claim_cases.json). Shares the process's Laya model."""

    def __init__(self, threshold: float = 0.4, model: str = "english", device: str = "auto", runner: Any = None) -> None:
        super().__init__(threshold)
        self.model = model  # Laya checkpoint: english | multilingual | typed-decisions
        self.device = device
        self._runner = runner  # tests inject a fake; normally the shared runner for this device

    async def score(self, atom: Atom) -> float:
        [result] = await self._laya().predict_batch(
            [{"state": {"claim": atom.text}, "questions": LAYA_QUESTION, "model": self.model}]
        )
        return result["answers"]["kind"]["probabilities"]["factual_claim"]

    async def start(self) -> None:
        await self._laya().agent(self.model)  # load the weights up front

    def _laya(self) -> Any:
        return self._runner or laya_runner(self.device)
