"""The request and the answer (Laya's wire shape, which Jev's System One protocol shares), and the runner role."""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal, Protocol, runtime_checkable

from pydantic import BaseModel


class Question(BaseModel):
    type: Literal["choice", "noul", "score"]  # the protocol's word for it
    instructions: str  # may refer to state fields in backticks: "According to `evidence`, is `claim` true?"
    criteria: dict[str, str] = {}  # choice: option -> what it means

    def wire(self) -> dict[str, Any]:
        return self.model_dump(exclude_defaults=True)  # no empty criteria on a noul/score question


class DecisionRequest(BaseModel):
    state: str | dict[str, Any]  # e.g. {"evidence": passage, "claim": claim}; field order matters to the model
    questions: dict[str, Question]
    model: str | None = None  # a checkpoint or model id; the runner's own default when None


class Answer(BaseModel):
    choice: str | None = None
    probabilities: dict[str, float] = {}  # every option; sums to 1
    confidence: float | None = None

    @property
    def label(self) -> str:
        """The most probable option (the model's `choice` when it reports no probabilities)."""
        return max(self.probabilities, key=self.probabilities.__getitem__) if self.probabilities else self.choice or ""


class DecisionResponse(BaseModel):
    answers: dict[str, Answer]  # the request's question keys
    usage: dict[str, float] = {}  # tokens, cost: whatever the server reports, this request's share


@runtime_checkable
class DecisionRunner(Protocol):
    """Role: answer decision requests. One method; `batch_size` is the items per model call (rows per GPU pass,
    questions per HTTP call), set by the runner's own `__init__`, never a `predict` argument: a per-call size would
    defeat merging requests across callers. Our runners subclass this explicitly (the type checker verifies them);
    anything with the same shape conforms, and `isinstance(x, DecisionRunner)` works for both."""

    batch_size: int

    async def predict(self, requests: list[DecisionRequest]) -> list[DecisionResponse]: ...


class DecisionPacking(StrEnum):
    """What shares one System One call, and so one model context (it changes the answers; see `SystemOneRunner`)."""

    CALL = "call"  # one `predict` call (a judge call: one claim's passages), up to `batch_size` per call: the default
    ALL = "all"  # every caller's requests in flight, `batch_size` per call: most throughput, claims mixed
    NONE = "none"  # one request per call, exactly the request Laya gets
