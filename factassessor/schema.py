from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, computed_field

Label = Literal["supports", "refutes", "not_enough_info"]
Verdict = Literal["supported", "refuted", "contested", "unverified"]


class Atom(BaseModel):
    id: int
    text: str  # self-contained claim, as written by the Atomizer
    span: tuple[int, int]  # char offsets in the input text, for UI highlighting
    claim_score: float | None = None  # P(this is a factual claim worth checking), set by the claim filter


class Evidence(BaseModel):
    url: str
    title: str
    text: str
    source: Literal["snippet", "page"]
    label: Label
    prob: float


class AtomResult(BaseModel):
    atom: Atom
    verdict: Verdict
    confidence: float = 0.0
    evidence: list[Evidence] = Field(default_factory=list)
    error: str | None = None


class CheckResult(BaseModel):
    text: str
    atoms: list[AtomResult]
    skipped: list[Atom] = Field(default_factory=list)
    latency_ms: float  # (the knowledge graph is built on demand: `factassessor.kg.build(result)`)

    @computed_field  # derived from `atoms`, so it can't go stale; still included in model_dump / JSON
    @property
    def fact_score(self) -> float | None:
        """supported / (supported + refuted + contested); None when no claim was decided."""
        return fact_score(self.atoms)



def fact_score(results: list[AtomResult]) -> float | None:
    """supported / (supported + refuted + contested); None if nothing was decided."""
    decided = [r for r in results if r.verdict != "unverified"]
    return sum(r.verdict == "supported" for r in decided) / len(decided) if decided else None

# --- stream events (FactAssessor.stream) -------------------------------------------------------------


class ClaimFound(BaseModel):
    """A claim passed the filter and is being checked (UI: underline `atom.span` as "checking…")."""

    type: Literal["claim_found"] = "claim_found"
    atom: Atom


class ClaimVerified(BaseModel):
    """One claim's verdict, the moment it's settled."""

    type: Literal["claim_verified"] = "claim_verified"
    result: AtomResult


class Done(BaseModel):
    """Everything is in: fact score and all results in text order (graph: `kg.build(event.result)`)."""

    type: Literal["done"] = "done"
    result: CheckResult


Event = ClaimFound | ClaimVerified | Done
