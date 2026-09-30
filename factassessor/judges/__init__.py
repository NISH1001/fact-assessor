"""Judges: (claim, snippets or pages) -> Evidence. `Judge` is the role; pick one with `FactAssessor(judge=...)`."""

from factassessor.judges._base import Judge
from factassessor.judges.decision import DecisionJudge

__all__ = ["Judge", "DecisionJudge"]
