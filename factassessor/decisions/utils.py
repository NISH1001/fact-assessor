"""Shared by the runners: merging concurrent callers (`Batcher`), packing requests into one call, POST with retries."""

from __future__ import annotations

import asyncio
import re
from collections.abc import Awaitable, Callable
from typing import Any

import httpx

from factassessor.decisions.types import DecisionRequest

RUN_LOCALLY = "To run without an API key, use the local model: FactAssessor(runner=LayaRunner())."  # what a runner without its API key suggests


class Batcher:
    """Merges every concurrent caller's requests into shared calls of `run(items) -> results`.

    Requests wait `max_wait_ms` for company, then go out `take` at a time (everything pending when None) with at
    most `max_concurrent` calls in flight; what arrives while every slot is busy piles up for the next call, so a
    busy model always gets full batches (measured on Laya: a batch per wait window instead left it working through
    2-3-row passes, 14x slower). Each request has its own future: a cancelled caller (a claim past its deadline)
    leaves the shared call alone, and a request still waiting when its caller goes is never sent.
    """

    def __init__(
        self, run: Callable[[list[Any]], Awaitable[list[Any]]], max_wait_ms: float = 5.0, max_concurrent: int = 1, take: int | None = None
    ) -> None:
        self.run = run
        self.max_wait_ms = max_wait_ms
        self.take = take
        self._slots = asyncio.Semaphore(max_concurrent)
        self._pending: list[tuple[Any, asyncio.Future[Any]]] = []
        self._flush: asyncio.Task[None] | None = None
        self._calls: set[asyncio.Task[None]] = set()

    async def submit(self, items: list[Any]) -> list[Any]:
        if not items:
            return []
        loop = asyncio.get_running_loop()
        futures = [loop.create_future() for _ in items]
        self._pending += zip(items, futures)
        if self._flush is None:
            self._flush = asyncio.create_task(self._flush_after_wait())
        return list(await asyncio.gather(*futures))

    async def alone(self, items: list[Any], take: int | None = None) -> list[Any]:
        """These items only, `take` per call (the batcher's own when None), under the same in-flight cap: no
        merging with other callers."""
        n = take or self.take or len(items) or 1

        async def one(piece: list[Any]) -> list[Any]:
            async with self._slots:
                return await self.run(piece)

        return [r for piece in await asyncio.gather(*(one(items[i : i + n]) for i in range(0, len(items), n))) for r in piece]

    async def _flush_after_wait(self) -> None:
        await asyncio.sleep(self.max_wait_ms / 1000)
        while self._pending:
            await self._slots.acquire()  # a free slot first, so everything arriving meanwhile joins this call
            self._pending = [(item, f) for item, f in self._pending if not f.done()]
            n = len(self._pending) if self.take is None else self.take
            batch, self._pending = self._pending[:n], self._pending[n:]
            if not batch:
                self._slots.release()
                continue
            call = asyncio.create_task(self._call(batch))
            self._calls.add(call)
            call.add_done_callback(self._calls.discard)
        self._flush = None

    async def _call(self, batch: list[tuple[Any, asyncio.Future[Any]]]) -> None:
        try:
            results = await self.run([item for item, _ in batch])
        except Exception as exc:
            for _, future in batch:
                if not future.done():
                    future.set_exception(exc)
            return
        finally:
            self._slots.release()
        for (_, future), result in zip(batch, results):
            if not future.done():
                future.set_result(result)


_FIELD = re.compile(r"`(\w+)`")  # a state field named in a question


def pack(requests: list[DecisionRequest], model: str) -> tuple[dict[str, Any], list[dict[str, str]]]:
    """One System One body for several requests, and per request its question keys -> the body's keys.
    A field with the same value in every request stays one value; a field that differs becomes a list with one
    entry per request, and each question's backticked reference to it gets the request's index (`evidence[2]`,
    Jev's own path syntax). Questions are keyed per request (`stance_0`, `stance_1`); field order is kept."""
    if len(requests) == 1:
        [r] = requests
        body = {"model": r.model or model, "state": r.state, "questions": {k: q.wire() for k, q in r.questions.items()}}
        return body, [{k: k for k in r.questions}]
    fields = list(dict.fromkeys(f for r in requests for f in r.state))  # type: ignore[union-attr]  # dict states only
    state: dict[str, Any] = {}
    for field in fields:
        values = [r.state.get(field, "") for r in requests]  # type: ignore[union-attr]
        state[field] = values[0] if all(v == values[0] for v in values) else values
    listed = {f for f, v in state.items() if isinstance(v, list)}
    questions: dict[str, Any] = {}
    routes: list[dict[str, str]] = []
    for i, r in enumerate(requests):
        route: dict[str, str] = {}
        for k, q in r.questions.items():
            route[k] = f"{k}_{i}"
            instructions = _FIELD.sub(lambda m: f"`{m.group(1)}[{i}]`" if m.group(1) in listed else m.group(0), q.instructions)
            questions[route[k]] = {**q.wire(), "instructions": instructions}
        routes.append(route)
    return {"model": requests[0].model or model, "state": state, "questions": questions}, routes


async def post_with_retries(http: httpx.AsyncClient, url: str, body: dict[str, Any], attempts: int = 4) -> dict[str, Any]:
    """POST `body`, retrying network errors, 429s and 5xx with backoff; the response JSON. Out of retries, the most
    recent failure is raised: the network error, or the HTTP error. An HTTP error carries the server's explanation
    (the response body), so a rejected request says why, not just "400 Bad Request"."""
    response: httpx.Response | None = None  # the most recent reply; None when the last attempt never got one
    network_error: httpx.TransportError | None = None
    for attempt in range(attempts):
        try:
            response = await http.post(url, json=body)
        except httpx.TransportError as exc:
            response, network_error = None, exc
            await asyncio.sleep(0.5 * 2**attempt)
            continue
        if response.status_code == 429 or response.status_code >= 500:
            await asyncio.sleep(0.5 * 2**attempt)  # rate limited or a provider hiccup: back off
            continue
        break
    if response is None:
        raise network_error  # type: ignore[misc]
    if response.is_error:
        raise httpx.HTTPStatusError(f"{response.status_code} {response.reason_phrase} from {url}: {response.text[:500]}",
                                    request=response.request, response=response)
    return response.json()
