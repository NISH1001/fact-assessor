"""Atoms -> (kept, skipped): drop atoms that aren't factual claims (opinions, questions, small talk).

Swap in anything with `async afilter(atoms) -> tuple[list[Atom], list[Atom]]`.
"""

from __future__ import annotations

from factassessor.laya import LayaRunner
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


class AtomFilter:
    """One Laya batch scores P(factual_claim) for every atom."""

    def __init__(
        self,
        laya: LayaRunner | None = None,
        n_atoms: int = 5,
        threshold: float = 0.4,  # low on purpose: dropping a real claim costs more than one extra search
        model: str = "english",  # Laya checkpoint: english | multilingual | typed-decisions
    ) -> None:
        self.laya = laya or LayaRunner()
        self.n_atoms = n_atoms
        self.threshold = threshold
        self.model = model

    async def afilter(self, atoms: list[Atom]) -> tuple[list[Atom], list[Atom]]:
        """Keep the top n_atoms at or above threshold, in input order. Returns (kept, skipped)."""
        if not atoms:
            return [], []
        results = await self.laya.predict_batch(
            [{"state": {"claim": a.text}, "questions": QUESTION, "model": self.model} for a in atoms]
        )
        scored = [
            a.model_copy(update={"checkworthiness": r["answers"]["kind"]["probabilities"]["factual_claim"]})
            for a, r in zip(atoms, results)
        ]
        passing = [a for a in scored if a.checkworthiness >= self.threshold]
        kept_ids = {a.id for a in sorted(passing, key=lambda a: -a.checkworthiness)[: self.n_atoms]}
        return [a for a in scored if a.id in kept_ids], [a for a in scored if a.id not in kept_ids]
