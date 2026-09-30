"""DecisionClaimFilter: a `DecisionRunner` (Laya by default) decides the kind of statement (the default filter)."""

from __future__ import annotations

from factassessor.claim_filters._base import KINDS, ClaimFilter
from factassessor.decisions import DecisionRequest, DecisionRunner, Question
from factassessor.laya import LayaRunner
from factassessor.schema import Atom

# `choice` beat `noul` clearly with Laya (17/19 vs 8/12 correct): its yes/no head was near-random here.
QUESTION = {"kind": Question(type="choice", instructions="What kind of statement is `claim`?", criteria=KINDS)}


class DecisionClaimFilter(ClaimFilter):
    """The runner decides the kind of statement; P(factual_claim) is the score (Laya: 17/19 on data/claim_cases.json)."""

    def __init__(self, runner: DecisionRunner | None = None, threshold: float = 0.4) -> None:
        super().__init__(threshold)
        self.runner = runner or LayaRunner()  # a resource: `aload()` on the chain loads it up front

    async def score(self, atom: Atom) -> float:
        [response] = await self.runner.predict([DecisionRequest(state={"claim": atom.text}, questions=QUESTION)])
        return response.answers["kind"].probabilities["factual_claim"]
