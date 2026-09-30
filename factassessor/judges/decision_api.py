"""DecisionAPIJudge: the judge as calls to OpenRouter's Decisions API (TypeSafe Jev and whatever else it serves).

Jev is a hosted decision model of the same kind as Laya: it reads a `state` and answers typed `questions` with
probabilities, generating no text. This judge sends Laya's exact question and state per (claim, passage) pair,
so the two judges are interchangeable and comparable: `DecisionAPIJudge()` in place of `LayaJudge()`.

Why: no local GPU floor. Calls run in parallel (`max_concurrent`), ~0.5s each, priced per input token
($0.042 per million for jev-1.13 at the time of writing, output free): a 30,000-passage eval replay is about
$0.50 and minutes instead of a 20-minute GPU queue. Needs `OPENROUTER_API_KEY` (env or `.env`).
"""

from __future__ import annotations

import asyncio
import os
from typing import Any

import httpx

from factassessor.judges._base import Judge
from factassessor.judges.laya import QUESTION
from factassessor.passages import normalize_text, word_windows
from factassessor.rankers import BM25Ranker, Ranker
from factassessor.schema import Evidence


class DecisionAPIJudge(Judge):
    URL = "https://openrouter.ai/api/alpha/decisions"

    def __init__(
        self,
        model: str = "~typesafe/jev-latest",
        api_key: str | None = None,  # default: OPENROUTER_API_KEY
        passages_per_page: int = 1,
        passage_words: int = 90,  # about Laya's 128 tokens, so both judges see passages of the same size
        max_concurrent: int = 16,
        timeout: float = 30.0,
        ranker: Ranker | None = None,
        url: str = URL,
    ) -> None:
        self.model = model
        self.api_key = api_key
        self.passages_per_page = passages_per_page
        self.passage_words = passage_words
        self.timeout = timeout
        self.ranker = ranker or BM25Ranker()
        self.url = url
        self.cost = 0.0  # USD so far, from the API's own `usage.cost`
        self._slots = asyncio.Semaphore(max_concurrent)
        self._http: httpx.AsyncClient | None = None

    def _headers(self) -> dict[str, str]:
        key = self.api_key or os.environ.get("OPENROUTER_API_KEY", "").strip()
        if not key:
            raise RuntimeError("OPENROUTER_API_KEY is missing (set it in the environment or .env)")
        return {"Authorization": f"Bearer {key}", "X-OpenRouter-Title": "fact-assessor"}

    async def aload(self) -> None:
        if self._http is None:
            self._http = httpx.AsyncClient(timeout=self.timeout, headers=self._headers())

    async def aclose(self) -> None:
        if self._http is not None:
            await self._http.aclose()
            self._http = None

    async def judge(self, claim: str, docs: list[dict[str, Any]]) -> list[Evidence]:
        await self.aload()
        claim = normalize_text(claim)
        passages: list[dict[str, Any]] = []
        for d in docs:
            base = {"url": d["url"], "title": d.get("title") or ""}
            if "text" in d:  # a crawled page: the ranker's best windows
                chunks = word_windows(normalize_text(d["text"]), self.passage_words, self.passage_words // 4)
                passages += [{**base, "text": t, "source": "page"} for t in self.ranker.top(claim, chunks, self.passages_per_page)]
            elif d.get("snippet"):
                text = normalize_text(d["snippet"])
                if len(text.split()) > 2 * self.passage_words:  # a snippet that is really a page
                    text = self.ranker.top(claim, word_windows(text, self.passage_words, self.passage_words // 4), 1)[0]
                passages.append({**base, "text": text, "source": "snippet"})
        decided = await asyncio.gather(*(self._decide(claim, p["text"]) for p in passages))
        return [Evidence(**p, label=label, prob=prob) for p, (label, prob) in zip(passages, decided)]

    async def _decide(self, claim: str, passage: str) -> tuple[str, float]:
        body = {"model": self.model, "state": {"evidence": passage, "claim": claim}, "questions": QUESTION}
        async with self._slots:
            for attempt in range(4):
                response = await self._http.post(self.url, json=body)  # type: ignore[union-attr]
                if response.status_code == 429 or response.status_code >= 500:
                    await asyncio.sleep(0.5 * 2**attempt)  # rate limited or a provider hiccup: back off
                    continue
                response.raise_for_status()
                break
            else:
                response.raise_for_status()
        data = response.json()
        answer = data["answers"]["stance"]
        probs = answer["probabilities"]
        label = answer.get("choice") or max(probs, key=probs.get)
        self.cost += (data.get("usage") or {}).get("cost") or 0.0
        return label, float(probs[label])
