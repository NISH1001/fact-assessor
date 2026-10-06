"""Decision runners: the model layer under the claim filter and the judge.

A decision model reads a `state` (named text fields) and answers typed `questions` about it with probabilities; it
generates no text. Several models answer that same request: Laya in-process (`LayaRunner`, factassessor.laya),
TypeSafe's Jev over HTTP (`SystemOneRunner`, also a remote `python -m laya.serve`), any chat LLM asked to pick an
option (`LLMRunner`, here), and GLiNER2.5-decide (`GlinerRunner`, factassessor.gliner). `DecisionJudge` and
`DecisionClaimFilter` are written once on top of the `DecisionRunner` protocol, so the model is one argument, and
the filter and the judge can run on different ones.

A runner never sees claims, pages or evidence: only requests. Batching is its job, not the caller's: `predict`
takes a list, concurrent callers' requests are merged into shared model calls (`Batcher`), cut at `batch_size`
and capped in flight, so 100 incoming requests become a few GPU passes or HTTP calls, never 100 parallel anything.
"""

from __future__ import annotations

import asyncio
import os
import re
from collections.abc import Awaitable, Callable
from enum import StrEnum
from typing import Any, Literal, Protocol, runtime_checkable

import httpx
from pydantic import BaseModel, Field
from pydantic_ai import Agent

from factassessor._llm import reasoning_off
from factassessor.keys import require_key, require_model_key

# --- the request and the answer (Laya's wire shape, which Jev's System One protocol shares) ----------------------


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


# --- merging concurrent callers ---------------------------------------------------------------------------------


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


# --- Jev / System One over HTTP ---------------------------------------------------------------------------------

_FIELD = re.compile(r"`(\w+)`")  # a state field named in a question


class DecisionPacking(StrEnum):
    """What shares one System One call, and so one model context (it changes the answers; see `SystemOneRunner`)."""

    CALL = "call"  # one `predict` call (a judge call: one claim's passages), up to `batch_size` per call: the default
    ALL = "all"  # every caller's requests in flight, `batch_size` per call: most throughput, claims mixed
    NONE = "none"  # one request per call, exactly the request Laya gets


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
                                  instead="To run without an API key, use the local model: FactAssessor(runner=LayaRunner()).")
        self.api_key = api_key
        self.batch_size = batch_size
        self.packing = DecisionPacking(packing)
        self.timeout = timeout
        self.cost = 0.0
        self._batch = Batcher(self._run, max_wait_ms, max_concurrent, take=batch_size)
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

    async def _run(self, requests: list[DecisionRequest]) -> list[DecisionResponse]:
        """One packed call for the dict-state requests; a string state can't be packed, so each goes alone."""
        packed = [i for i, r in enumerate(requests) if isinstance(r.state, dict)]
        calls = ([packed] if packed else []) + [[i] for i, r in enumerate(requests) if not isinstance(r.state, dict)]
        results = await asyncio.gather(*(self._post([requests[i] for i in call]) for call in calls))
        answered = {i: res for call, group in zip(calls, results) for i, res in zip(call, group)}
        return [answered[i] for i in range(len(requests))]

    async def _post(self, requests: list[DecisionRequest]) -> list[DecisionResponse]:
        body, routes = pack(requests, self.model)
        for attempt in range(4):
            try:
                response = await self._http.post(self.url, json=body)  # type: ignore[union-attr]
            except httpx.TransportError:
                await asyncio.sleep(0.5 * 2**attempt)
                continue
            if response.status_code == 429 or response.status_code >= 500:
                await asyncio.sleep(0.5 * 2**attempt)  # rate limited or a provider hiccup: back off
                continue
            response.raise_for_status()
            break
        else:
            response.raise_for_status()
        data = response.json()
        usage = {k: float(v) for k, v in (data.get("usage") or {}).items() if isinstance(v, (int, float))}
        self.cost += usage.get("cost", 0.0)
        share = {k: v / len(requests) for k, v in usage.items()}
        return [
            DecisionResponse(answers={key: Answer.model_validate(data["answers"][packed]) for key, packed in route.items()}, usage=share)
            for route in routes
        ]


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


# --- any chat LLM -----------------------------------------------------------------------------------------------

DEFAULT_LLM = "openai:gpt-6-luna"  # the most accurate judge we measured (15/15 on the judge cases); reasoning off

LLM_INSTRUCTIONS = """\
Each item shows named fields, a question about them (field names in backticks), and the options with what each
means. Using the fields alone, pick exactly one option per item and give your confidence from 0 to 1. Return
exactly one item per id."""


class Pick(BaseModel):
    id: int
    choice: str
    confidence: float = Field(ge=0, le=1)


class Picks(BaseModel):
    items: list[Pick]


class LLMRunner(DecisionRunner):
    """Any chat model as a decision model (pydantic-ai: OpenAI, OpenRouter, Ollama, an OpenAI-compatible endpoint).

    Every question becomes one numbered item of a prompt (its state fields, the question, the options); the model
    returns one pick with a confidence per item, as structured output, and that becomes the answer: the pick gets
    its confidence, the other options share the rest. Self-reported, not calibrated like Laya's or Jev's, so set
    `strong` from a replay before trusting a threshold. `batch_size` items per call; concurrent callers' requests
    are merged into shared calls (instructions sent once) with up to `max_concurrent` in flight. A call that fails
    raises: the claim comes back unverified with the error, not silently unsupported. An item the model skips or
    misnames gets even probabilities and confidence 0.
    """

    def __init__(
        self,
        model: str = DEFAULT_LLM,
        model_settings: dict[str, Any] | None = None,  # default: reasoning as low as the model allows
        batch_size: int = 40,
        max_concurrent: int = 8,
    ) -> None:
        require_model_key(model, needed_by="LLMRunner", instead="To run without an API key, use the local model: FactAssessor(runner=LayaRunner()).")
        self.model = model
        self.batch_size = batch_size
        self.agent = Agent(
            model,
            output_type=Picks,
            instructions=LLM_INSTRUCTIONS,
            model_settings=reasoning_off(model) if model_settings is None else model_settings,
            defer_model_check=True,  # don't require an API key until the first call
        )
        self._batch = Batcher(self._run, max_wait_ms=5.0, max_concurrent=max_concurrent, take=batch_size)

    async def predict(self, requests: list[DecisionRequest]) -> list[DecisionResponse]:
        if not requests:
            return []
        return await self._batch.submit(requests)

    async def _run(self, requests: list[DecisionRequest]) -> list[DecisionResponse]:
        items = [(i, key, r, q) for i, r in enumerate(requests) for key, q in r.questions.items()]
        prompt = "\n\n".join(_item(n, r.state, q) for n, (_, _, r, q) in enumerate(items, 1))
        picks = {p.id: p for p in (await self.agent.run(prompt)).output.items}
        answers: list[dict[str, Answer]] = [{} for _ in requests]
        for n, (i, key, _, q) in enumerate(items, 1):
            answers[i][key] = _answer(picks.get(n), list(q.criteria))
        return [DecisionResponse(answers=a) for a in answers]


def _item(n: int, state: str | dict[str, Any], q: Question) -> str:
    fields = f"state: {state}" if isinstance(state, str) else "\n".join(f"{k}: {v}" for k, v in state.items())
    options = "\n".join(f"- {label}: {meaning}" for label, meaning in q.criteria.items())
    return f"[{n}]\n{fields}\nquestion: {q.instructions}\noptions:\n{options}"


def _answer(pick: Pick | None, options: list[str]) -> Answer:
    chosen = next((o for o in options if pick and o.lower() == pick.choice.strip().lower()), None)
    if chosen is None or pick is None:  # skipped or an option that doesn't exist: no lean either way
        return Answer(probabilities={o: 1 / len(options) for o in options}, confidence=0.0)
    rest = (1 - pick.confidence) / (len(options) - 1) if len(options) > 1 else 0.0
    return Answer(choice=chosen, probabilities={o: pick.confidence if o == chosen else rest for o in options}, confidence=pick.confidence)
