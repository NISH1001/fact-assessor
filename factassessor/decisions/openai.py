"""OpenAIDecisionRunner: gpt-6-luna through OpenAI's Decisions API (`POST /v1/decisions`)."""

from __future__ import annotations

import asyncio
from functools import partial
from typing import Any

import httpx
from loguru import logger

from factassessor.decisions.types import Answer, DecisionPacking, DecisionRequest, DecisionResponse, DecisionRunner
from factassessor.decisions.utils import RUN_LOCALLY, Batcher, post_packed, post_with_retries
from factassessor.keys import require_key
from factassessor.utils import http_pool


class OpenAIDecisionRunner(DecisionRunner):
    """gpt-6-luna through OpenAI's Decisions API (`POST /v1/decisions`): a decision model like Jev (probabilities in
    one pass, no generated text), priced on input tokens only. Key: `api_key`, else `OPENAI_API_KEY`.

    A request is one shared `input` text and a list of questions about it, evaluated independently (OpenAI's
    Decisions guide). A predict call's requests (a claim's passages) are laid out the way the guide describes for
    several candidates against one reference: the fields every request shares (`claim: ...`) are the `input`, and
    each request's own fields (`evidence: ...`) go inside its own questions, so a passage is judged with only the
    claim in view, not the other passages. A single request, or a string state, is the input itself with its
    questions. Requests that share no field can't be one call and go one by one. Question types: `choice`
    (options with descriptions), `noul` -> OpenAI's `predicate` (yes/no: `{"yes": p, "no": 1 - p}`), `score`
    (ordered levels). `cost` is the USD spent so far when the response reports input tokens ($0.10 per million).

    `packing`: `CALL` (the default) sends a claim's passages in one call (one input, one question per passage).
    `NONE` sends each passage alone (3x the calls; with it raise `max_concurrent`, e.g. 64, as far as your OpenAI
    rate limit allows; a 429 is retried with backoff). Before 2026-10-25 the layout was Jev's written out as text
    (every passage in the input as `evidence[i]:` lines, questions naming their line); on 10 answers that scored
    0.637 against 0.668 for one passage per call, so the guide's layout replaced it."""

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
        self.max_concurrent = max_concurrent
        self._batch = Batcher(partial(post_packed, post=self._post), max_wait_ms, max_concurrent, take=batch_size)
        self._http: httpx.AsyncClient | None = None
        if self.packing is DecisionPacking.NONE and max_concurrent <= 16:
            logger.info("OpenAIDecisionRunner(packing='none') makes one call per passage, {} in flight: claims queue "
                        "(3x slower than packing='call' on the eval). Raise max_concurrent (e.g. 64) as far as your "
                        "OpenAI rate limit allows.", max_concurrent)

    async def aload(self) -> None:
        if self._http is None:
            self._http = http_pool(self.timeout, self.max_concurrent, headers={"Authorization": f"Bearer {self.api_key}"})

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
        body, routes = _body(requests, self.model)
        if body is None:  # nothing shared between the requests: each is its own call
            return [res for [res] in await asyncio.gather(*(self._post([r]) for r in requests))]
        data = await post_with_retries(self._http, self.url, body)  # type: ignore[arg-type]
        answers = {a["name"]: _openai_answer(a) for a in data.get("answers", [])}
        tokens = float(((data.get("usage") or {}).get("input_tokens")) or 0)
        self.cost += tokens * self.PRICE_PER_INPUT_TOKEN
        share = {"input_tokens": tokens / len(requests)} if tokens else {}
        return [DecisionResponse(answers={key: answers[sent] for key, sent in route.items()}, usage=share) for route in routes]


def _body(requests: list[DecisionRequest], model: str) -> tuple[dict[str, Any] | None, list[dict[str, str]]]:
    """One Decisions request for these requests, and per request its question keys -> the body's question names.
    Alone (or a string state): the state is the input, the questions as they are. Several: the fields with the same
    value in every request (the claim) are the input, and each request's own fields (its passage) open its
    questions as `field: value` lines, one question per request and key (`stance_0`, `stance_1`, ...): OpenAI's
    layout for several candidates against one reference. No shared field: None (they can't be one call)."""
    if len(requests) == 1 or not all(isinstance(r.state, dict) for r in requests):
        [r] = requests
        state = r.state if isinstance(r.state, str) else "\n".join(f"{f}: {v}" for f, v in r.state.items())
        return {"model": r.model or model, "input": state, "questions": [_openai_question(k, q.wire()) for k, q in r.questions.items()]}, [{k: k for k in r.questions}]
    fields = list(dict.fromkeys(f for r in requests for f in r.state))  # type: ignore[union-attr]
    shared = [f for f in fields if all(r.state.get(f) == requests[0].state.get(f) for r in requests)]  # type: ignore[union-attr]
    if not shared:
        return None, []
    questions, routes = [], []
    for i, r in enumerate(requests):
        own = "\n".join(f"{f}: {r.state[f]}" for f in fields if f not in shared and f in r.state)  # type: ignore[index, operator]
        route = {}
        for k, q in r.questions.items():
            route[k] = f"{k}_{i}"
            questions.append(_openai_question(route[k], {**q.wire(), "instructions": f"{own}\n\n{q.instructions}" if own else q.instructions}))
        routes.append(route)
    body = {"model": requests[0].model or model, "input": "\n".join(f"{f}: {requests[0].state[f]}" for f in shared), "questions": questions}  # type: ignore[index]
    return body, routes


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
