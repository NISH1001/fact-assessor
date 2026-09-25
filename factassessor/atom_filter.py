"""Drop atoms that aren't factual claims (opinions, questions, small talk), as a step: atoms -> atoms.

Compose: `LLMAtomizer() >> LayaCheckworthy() >> Take(8)`. Any `Filter(pred)` works in its place.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

from factassessor.laya import LayaRunner
from factassessor.pipeline import Map, Step, report_dropped
from factassessor.schema import Atom

# `choice` beat `noul` clearly in our tests (17/19 vs 8/12 correct): Laya's yes/no head was near-random here.
QUESTION = {
    "kind": {
        "type": "choice",
        "instructions": "What kind of statement is `claim`?",
        "criteria": {
            "factual_claim": "asserts something about the world that can be verified as true or false "
            "(people, places, dates, numbers, events)",
            "opinion": "a personal view, preference, or judgment",
            "question_or_request": "asks something or asks someone to do something",
            "social": "greeting, thanks, introduction, or small talk",
        },
    }
}


class LayaCheckworthy(Step):
    """Scores P(factual_claim) with Laya (one decision per atom; the shared runner batches atoms that arrive
    together), stores it on `atom.checkworthiness`, and drops atoms below `threshold`."""

    def __init__(
        self,
        laya: LayaRunner | None = None,
        threshold: float = 0.4,  # low on purpose: dropping a real claim costs more than one extra search
        model: str = "english",  # Laya checkpoint: english | multilingual | typed-decisions
    ) -> None:
        self.laya = laya or LayaRunner()
        self.threshold = threshold
        self.model = model

    def __call__(self, atoms: AsyncIterator[Atom]) -> AsyncIterator[Atom]:
        return Map(self.score)(atoms)

    async def score(self, atom: Atom) -> Atom | None:
        [result] = await self.laya.predict_batch(
            [{"state": {"claim": atom.text}, "questions": QUESTION, "model": self.model}]
        )
        scored = atom.model_copy(update={"checkworthiness": result["answers"]["kind"]["probabilities"]["factual_claim"]})
        if scored.checkworthiness >= self.threshold:
            return scored
        report_dropped(scored)
        return None

    async def start(self) -> None:
        await self.laya.agent(self.model)  # load this checkpoint's weights up front
