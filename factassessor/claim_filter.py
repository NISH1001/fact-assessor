"""Claim filters: is an atom a factual claim worth checking? A step: atoms -> the atoms worth checking.

`ClaimFilter` is the role: implement `score(atom) -> float`, the probability the atom is a factual claim (as opposed
to an opinion, question, greeting, or filler). The base keeps atoms scoring at least `threshold`, records the score
as `atom.claim_score`, and reports the rest as `skipped`. Implementations: `LayaClaimFilter` (default) and
`GlinerClaimFilter` (in factassessor.gliner).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from typing import Any

from factassessor.laya import laya_runner
from factassessor.pipeline import Map, Step, report_dropped
from factassessor.schema import Atom

# the four kinds of statement both model-backed filters choose between; P(factual_claim) is the score
KINDS = {
    "factual_claim": "asserts something about the world that can be verified as true or false "
    "(people, places, dates, numbers, events)",
    "opinion": "a personal view, preference, or judgment",
    "question_or_request": "asks something or asks someone to do something",
    "social": "greeting, thanks, introduction, or small talk",
}


class ClaimFilter(Step, ABC):
    """Role: score atoms, keep the factual claims. Threshold is low on purpose: a dropped real claim is never
    checked, while a kept opinion only costs one extra search."""

    def __init__(self, threshold: float = 0.4) -> None:
        self.threshold = threshold

    @abstractmethod
    async def score(self, atom: Atom) -> float:
        """P(atom is a factual claim), 0..1."""

    def __call__(self, atoms: AsyncIterator[Atom]) -> AsyncIterator[Atom]:
        return Map(self._keep)(atoms)  # every atom scored concurrently

    async def _keep(self, atom: Atom) -> Atom | None:
        scored = atom.model_copy(update={"claim_score": await self.score(atom)})
        if scored.claim_score >= self.threshold:
            return scored
        report_dropped(scored)
        return None


# `choice` beat `noul` clearly in our tests (17/19 vs 8/12 correct): Laya's yes/no head was near-random here.
LAYA_QUESTION = {"kind": {"type": "choice", "instructions": "What kind of statement is `claim`?", "criteria": KINDS}}


class LayaClaimFilter(ClaimFilter):
    """Laya decides the kind of statement (17/19 on benchmarks/claim_cases.py). Shares the process's Laya model."""

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
