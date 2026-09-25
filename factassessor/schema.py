from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

Label = Literal["supports", "refutes", "not_enough_info"]
Verdict = Literal["supported", "refuted", "contested", "unverified"]


class Atom(BaseModel):
    id: int
    text: str  # self-contained claim, as written by the Atomizer
    span: tuple[int, int]  # char offsets in the input text, for UI highlighting
    checkworthiness: float | None = None


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
    fact_score: float | None  # None when nothing was checked
    graph: dict[str, Any]  # {"nodes": [...], "edges": [...]}
    latency_ms: float


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
    """Everything is in: fact score, graph, and all results in text order."""

    type: Literal["done"] = "done"
    result: CheckResult


Event = ClaimFound | ClaimVerified | Done
