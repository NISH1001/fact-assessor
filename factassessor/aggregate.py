"""Per-claim results -> fact score. (The knowledge graph lives in `kg.py`, built on demand.)"""

from __future__ import annotations

from factassessor.schema import AtomResult


def fact_score(results: list[AtomResult]) -> float | None:
    """supported / (supported + refuted + contested); None if nothing was decided."""
    decided = [r for r in results if r.verdict != "unverified"]
    return sum(r.verdict == "supported" for r in decided) / len(decided) if decided else None
