"""Judges: (claim, snippets or pages) -> Evidence. `Judge` is the role; pick one with `FactAssessor(judge=...)`."""

from factassessor.judges._base import Judge
from factassessor.judges.decision_api import DecisionAPIJudge
from factassessor.judges.gliner import GlinerJudge
from factassessor.judges.laya import LayaJudge
from factassessor.judges.llm import LLMJudge

__all__ = ["Judge", "LayaJudge", "GlinerJudge", "LLMJudge", "DecisionAPIJudge"]
