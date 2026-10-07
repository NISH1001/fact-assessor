"""OpenAIDecisionRunner: gpt-6-luna through OpenAI's Decisions API (`POST /v1/decisions`)."""

from __future__ import annotations

from functools import partial
from typing import Any

import httpx
from loguru import logger

from factassessor.decisions.types import Answer, DecisionPacking, DecisionRequest, DecisionResponse, DecisionRunner
from factassessor.decisions.utils import RUN_LOCALLY, Batcher, pack, post_packed, post_with_retries
from factassessor.keys import require_key


class OpenAIDecisionRunner(DecisionRunner):
    """gpt-6-luna through OpenAI's Decisions API (`POST /v1/decisions`): a decision model like Jev (probabilities in
    one pass, no generated text), priced on input tokens only. Key: `api_key`, else `OPENAI_API_KEY`.

    A request is the evidence as `input` text and a list of questions. Our requests are packed per predict call as
    for Jev (`pack`: the shared claim once, the passages numbered), then written out as text: `evidence[0]: ...`
    lines and `claim: ...`, which the questions name in backticks (`` `evidence[2]` ``). Question types: `choice`
    (options with descriptions), `noul` -> OpenAI's `predicate` (yes/no: `{"yes": p, "no": 1 - p}`), `score`
    (ordered levels). `cost` is the USD spent so far when the response reports input tokens ($0.10 per million).

    `packing`: `CALL` (the default) sends a claim's passages in one call. `NONE` sends each passage alone, the
    layout OpenAI's docs describe (one input, questions about that input). Measured on 10 answers, same pages:
    `NONE` F1 0.668 vs 0.637, about the same cost, but 3x slower (77 s vs 25 s), because a claim's ~30 calls queue
    for `max_concurrent` (16 by default, shared by every claim). With `NONE`, raise `max_concurrent` (e.g. 64) as
    far as your OpenAI rate limit allows; a 429 is retried with backoff."""

    URL = "https://api.openai.com/v1/decisions"
    PRICE_PER_INPUT_TOKEN = 0.10 / 1e6

    def __init__(
        self,
        model: str = "gpt-6-luna",
        url: str | None = None,
        api_key: str | None = None,
        batch_size: int = 40,  # requests per call
        max_concurrent: int = 16,  # calls in flight
        packing: DecisionPacking | str = DecisionPacking.CALL,  # as SystemOneRunner
        max_wait_ms: float = 5.0,
        timeout: float = 30.0,
    ) -> None:
        self.model = model
        self.url = url or self.URL
        self.api_key = require_key("OPENAI_API_KEY", api_key, needed_by="OpenAI's Decisions API (OpenAIDecisionRunner)",
                                   instead=RUN_LOCALLY)
        self.batch_size = batch_size
        self.packing = DecisionPacking(packing)
        self.timeout = timeout
        self.cost = 0.0
        self._batch = Batcher(partial(post_packed, post=self._post), max_wait_ms, max_concurrent, take=batch_size)
        self._http: httpx.AsyncClient | None = None
        if self.packing is DecisionPacking.NONE and max_concurrent <= 16:
            logger.info("OpenAIDecisionRunner(packing='none') makes one call per passage, {} in flight: claims queue "
                        "(3x slower than packing='call' on the eval). Raise max_concurrent (e.g. 64) as far as your "
                        "OpenAI rate limit allows.", max_concurrent)

    async def aload(self) -> None:
        if self._http is None:
            self._http = httpx.AsyncClient(timeout=self.timeout, headers={"Authorization": f"Bearer {self.api_key}"})

    async def aclose(self) -> None:
        if self._http is not None:
            await self._http.aclose()
            self._http = None

    async def predict(self, requests: list[DecisionRequest]) -> list[DecisionResponse]:
        if not requests:
            return []
        await self.aload()
        if self.packing is DecisionPacking.ALL:
            return await self._batch.submit(requests)
        return await self._batch.alone(requests, take=1 if self.packing is DecisionPacking.NONE else None)

    async def _post(self, requests: list[DecisionRequest]) -> list[DecisionResponse]:
        packed, routes = pack(requests, self.model)
        body = {"model": packed["model"], "input": _as_text(packed["state"]),
                "questions": [_openai_question(name, q) for name, q in packed["questions"].items()]}
        data = await post_with_retries(self._http, self.url, body)  # type: ignore[arg-type]
        answers = {a["name"]: _openai_answer(a) for a in data.get("answers", [])}
        tokens = float(((data.get("usage") or {}).get("input_tokens")) or 0)
        self.cost += tokens * self.PRICE_PER_INPUT_TOKEN
        share = {"input_tokens": tokens / len(requests)} if tokens else {}
        return [DecisionResponse(answers={key: answers[sent] for key, sent in route.items()}, usage=share) for route in routes]


def _as_text(state: str | dict[str, Any]) -> str:
    """A packed state as `field: value` lines; a list field as `field[i]: value`, as the questions refer to them."""
    if isinstance(state, str):
        return state
    lines = []
    for field, value in state.items():
        if isinstance(value, list):
            lines += [f"{field}[{i}]: {v}" for i, v in enumerate(value)]
        else:
            lines.append(f"{field}: {value}")
    return "\n".join(lines)


def _openai_question(name: str, wire: dict[str, Any]) -> dict[str, Any]:
    kind = wire["type"]
    question: dict[str, Any] = {"type": "predicate" if kind == "noul" else kind, "name": name, "instructions": wire["instructions"]}
    criteria = wire.get("criteria") or {}
    if kind == "choice":
        question["choices"] = [{"value": k, "description": v} for k, v in criteria.items()]
    elif kind == "score":
        question["levels"] = [{"label": k, "description": v} for k, v in criteria.items()]
    return question


def _openai_answer(a: dict[str, Any]) -> Answer:
    if a.get("type") == "predicate":
        p = float(a["probability"])
        return Answer(probabilities={"yes": p, "no": 1 - p})
    key = "label" if a.get("type") == "score" else "value"
    return Answer(choice=a.get("choice"), probabilities={str(x[key]): float(x["probability"]) for x in a.get("probabilities", [])},
                  confidence=a.get("confidence"))
