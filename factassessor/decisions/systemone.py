"""SystemOneRunner: Jev's System One protocol over HTTP (OpenRouter, or a remote `python -m laya.serve`)."""

from __future__ import annotations

import os
from functools import partial

import httpx

from factassessor.decisions.types import Answer, DecisionPacking, DecisionRequest, DecisionResponse, DecisionRunner
from factassessor.decisions.utils import RUN_LOCALLY, Batcher, pack, post_packed, post_with_retries
from factassessor.keys import require_key


class SystemOneRunner(DecisionRunner):
    """Jev's System One protocol over HTTP: OpenRouter's Jev by default, or a remote `python -m laya.serve`.

    Jev is a hosted decision model of the same kind as Laya: no local GPU floor, calls run in parallel (~0.5s
    each), priced per input token ($0.042 per million for jev-1.13 at the time of writing, output free): a
    30,000-passage eval replay is about $0.50 and minutes instead of a 20-minute GPU queue.

    One call carries one state and many questions, answered together, and Jev's docs say to ask every question
    about the same state in one request. So the requests of one `predict` call (a judge call: one claim, its
    passages) are packed into one call, up to `batch_size`: a field with the same value in every request (the
    claim) is sent once, a field that differs (the evidence) becomes a list, and each question names its own entry
    (`evidence[2]`). Every request gets its own answers back; a single request goes out verbatim. Jev's window is
    32k tokens: 40 passages of 90 words plus a claim are ~6k. `cost` is the USD spent so far (`usage.cost`, summed).

    What shares a call is the model's context, and it changes answers (on a 4-answer sample, 13% of passage
    labels differed between one request per call and 40 mixed from every claim in flight; alone, Jev was less
    sure of true claims). `packing` picks it: `DecisionPacking.CALL` (the default, one predict call per request),
    `DecisionPacking.ALL` (every caller's requests in flight, more throughput, claims mixed), `DecisionPacking.NONE` (one request
    per call).

    URL: `url`, else `OPENROUTER_DECISIONS_URL`, else OpenRouter. Key: `api_key`, else `OPENROUTER_API_KEY`
    (env or .env); a Laya server needs none.
    """

    URL = "https://openrouter.ai/api/v1/systemone"

    def __init__(
        self,
        model: str = "~typesafe/jev-latest",
        url: str | None = None,
        api_key: str | None = None,
        batch_size: int = 40,  # requests per call
        max_concurrent: int = 16,  # calls in flight
        packing: DecisionPacking | str = DecisionPacking.CALL,  # what shares a call: one predict call, every caller in flight, nothing
        max_wait_ms: float = 5.0,  # DecisionPacking.ALL: how long a request waits for company
        timeout: float = 30.0,
    ) -> None:
        self.model = model
        self.url = url or os.environ.get("OPENROUTER_DECISIONS_URL", "").strip() or self.URL
        if "openrouter.ai" in self.url:  # a local Laya server needs no key
            api_key = require_key("OPENROUTER_API_KEY", api_key, needed_by="Jev (SystemOneRunner, on OpenRouter)",
                                  instead=RUN_LOCALLY)
        self.api_key = api_key
        self.batch_size = batch_size
        self.packing = DecisionPacking(packing)
        self.timeout = timeout
        self.cost = 0.0
        self._batch = Batcher(partial(post_packed, post=self._post), max_wait_ms, max_concurrent, take=batch_size)
        self._http: httpx.AsyncClient | None = None

    def _headers(self) -> dict[str, str]:
        headers = {"X-OpenRouter-Title": "fact-assessor"}
        key = self.api_key or os.environ.get("OPENROUTER_API_KEY", "").strip()
        if key:
            headers["Authorization"] = f"Bearer {key}"
        elif "openrouter.ai" in self.url:
            raise RuntimeError("OPENROUTER_API_KEY is missing (set it in the environment or .env)")
        return headers

    async def aload(self) -> None:
        if self._http is None:
            self._http = httpx.AsyncClient(timeout=self.timeout, headers=self._headers())

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
        body, routes = pack(requests, self.model)
        data = await post_with_retries(self._http, self.url, body)  # type: ignore[arg-type]
        usage = {k: float(v) for k, v in (data.get("usage") or {}).items() if isinstance(v, (int, float))}
        self.cost += usage.get("cost", 0.0)
        share = {k: v / len(requests) for k, v in usage.items()}
        return [
            DecisionResponse(answers={key: Answer.model_validate(data["answers"][packed]) for key, packed in route.items()}, usage=share)
            for route in routes
        ]
