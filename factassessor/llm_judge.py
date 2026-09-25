"""LLMJudge: the evidence judge as an LLM call with structured output (pydantic-ai).

Measured on benchmarks/judge_cases.py (4 runs each): gpt-6-luna 15/15 in 3 of 4 runs (Laya 13/15, GLiNER 12/15),
~2.1s for 15 pairs, so ~10x slower than Laya: the pick when accuracy matters more than speed.

Batching: with `window_ms > 0`, judge requests arriving within that window (from any claims, snippets or pages) go
out as ONE structured call (split into chunks of `max_pairs`), and each caller gets back its own results. That
sends the instructions once (fewer calls, fewer tokens), but it measured ~0.2-0.3s slower than parallel calls
(2.34s vs 2.07s median for 15 pairs): one response has to write every answer in sequence. So the default is
`window_ms=0`, one call per request, all concurrent.
"""

from __future__ import annotations

import asyncio
import logging
import re
from typing import Any, Literal

from pydantic import BaseModel, Field
from pydantic_ai import Agent

from factassessor.evidence_judge import Judge
from factassessor.passages import chunk, top_passages
from factassessor.schema import Evidence

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "openai:gpt-6-luna"  # most accurate judge we measured; reasoning off
DEFAULT_SETTINGS: dict[str, Any] = {"openai_reasoning_effort": "none"}

INSTRUCTIONS = """\
You are a careful fact-checker. Each item is a claim and one evidence passage. Using the evidence alone, decide:
- supports: the evidence states the same thing as the claim (the dates, numbers, names, and places match)
- refutes: the evidence contradicts the claim (a different date, number, place, or person for the same thing)
- not_enough_info: the evidence is about something else, or doesn't settle the claim
Give your confidence from 0 to 1. Return exactly one item per id."""


class Stance(BaseModel):
    id: int
    label: Literal["supports", "refutes", "not_enough_info"]
    confidence: float = Field(ge=0, le=1)


class Stances(BaseModel):
    items: list[Stance]


class LLMJudge(Judge):
    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        model_settings: dict[str, Any] | None = None,
        window_ms: float = 0.0,  # > 0: merge requests arriving this close together into one call (cheaper, a bit slower)
        max_pairs: int = 40,  # pairs per API call; bigger batches are split and sent concurrently
        passages_per_page: int = 1,
        passage_words: int = 200,
    ) -> None:
        self.agent = Agent(
            model,
            output_type=Stances,
            instructions=INSTRUCTIONS,
            model_settings=DEFAULT_SETTINGS if model_settings is None else model_settings,
            defer_model_check=True,  # don't require an API key until the first call
        )
        self.window_ms = window_ms
        self.max_pairs = max_pairs
        self.passages_per_page = passages_per_page
        self.passage_words = passage_words
        self._pending: list[tuple[str, list[dict[str, Any]], asyncio.Future[list[Evidence]]]] = []
        self._flush: asyncio.Task[None] | None = None

    async def judge(self, claim: str, docs: list[dict[str, Any]]) -> list[Evidence]:
        passages = self._passages(claim, docs)
        if not passages:
            return []
        if not self.window_ms:
            [evidence] = await self._run([(claim, passages)])
            return evidence
        future: asyncio.Future[list[Evidence]] = asyncio.get_running_loop().create_future()
        self._pending.append((claim, passages, future))
        if self._flush is None:
            self._flush = asyncio.create_task(self._flush_after_wait())
        return await future

    async def _flush_after_wait(self) -> None:
        await asyncio.sleep(self.window_ms / 1000)
        batch, self._pending, self._flush = self._pending, [], None
        try:
            results = await self._run([(claim, passages) for claim, passages, _ in batch])
        except Exception as exc:
            for *_, future in batch:
                if not future.done():
                    future.set_exception(exc)
            return
        for (*_, future), evidence in zip(batch, results):
            if not future.done():  # the caller may have been cancelled (e.g. its claim settled)
                future.set_result(evidence)

    async def _run(self, groups: list[tuple[str, list[dict[str, Any]]]]) -> list[list[Evidence]]:
        """One structured call per `max_pairs` pairs (concurrently); results regrouped per caller, in order."""
        pairs = [(g, claim, p) for g, (claim, passages) in enumerate(groups) for p in passages]
        stances: dict[int, Stance] = {}
        chunks = [list(range(i, min(i + self.max_pairs, len(pairs)))) for i in range(0, len(pairs), self.max_pairs)]
        for found in await asyncio.gather(*(self._ask([(i, pairs[i][1], pairs[i][2]["text"]) for i in ids]) for ids in chunks)):
            stances.update(found)
        out: list[list[Evidence]] = [[] for _ in groups]
        for i, (g, _, passage) in enumerate(pairs):
            s = stances.get(i)
            label, prob = (s.label, s.confidence) if s else ("not_enough_info", 0.0)  # skipped by the model: no weight
            out[g].append(Evidence(**passage, label=label, prob=prob))
        return out

    async def _ask(self, items: list[tuple[int, str, str]]) -> dict[int, Stance]:
        prompt = "\n\n".join(f"[{i}] claim: {claim}\n    evidence: {text}" for i, claim, text in items)
        try:
            result = await self.agent.run(prompt)
        except Exception as exc:  # an outage degrades these pairs to "no evidence", not a failed check
            logger.warning("LLMJudge call failed for %d pairs: %r", len(items), exc)
            return {}
        wanted = {i for i, _, _ in items}
        return {s.id: s for s in result.output.items if s.id in wanted}

    def _passages(self, claim: str, docs: list[dict[str, Any]]) -> list[dict[str, Any]]:
        passages = []
        for doc in docs:
            base = {"url": doc["url"], "title": doc.get("title", "")}
            if "text" in doc:
                chunks = chunk(doc["text"], _words, max_tokens=self.passage_words, overlap=self.passage_words // 4)
                passages += [{**base, "text": t, "source": "page"} for t in top_passages(claim, chunks, k=self.passages_per_page)]
            elif doc.get("snippet"):
                passages.append({**base, "text": doc["snippet"], "source": "snippet"})
        return passages


def _words(text: str, add_special_tokens: bool = False, return_offsets_mapping: bool = False) -> dict[str, Any]:
    """Word "tokenizer" for `passages.chunk`: an LLM doesn't need exact token counts, just passage-sized pieces."""
    spans = [m.span() for m in re.finditer(r"\S+", text)]
    return {"input_ids": list(range(len(spans))), "offset_mapping": spans}
