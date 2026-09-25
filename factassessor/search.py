"""Searchers: query -> hits `{"url", "title", "snippet"}`, in rank order. A step.

`Searcher` is the role: implement `search(query)`. Compose: `SerperSearcher() >> not_blocked() >> Take(5)`.
"""

from __future__ import annotations

import asyncio
import os
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from typing import Any
from urllib.parse import urlparse

import httpx

from factassessor.pipeline import FlatMap, Pred, Step

SERPER_URL = "https://google.serper.dev/search"

# Social media pages are mostly reposts/comments and often crawl badly; video pages have no usable text.
# Twitter/X and LinkedIn are kept: they're primary sources for people and organisations.
BLOCKED_DOMAINS = (
    "facebook.com", "fb.com", "instagram.com", "tiktok.com", "pinterest.com", "threads.net",
    "youtube.com", "youtu.be",
)


class Searcher(Step, ABC):
    """Role: query -> hits. Implement `search`; streaming, concurrency, and chaining come from here."""

    @abstractmethod
    async def search(self, query: str) -> list[dict[str, Any]]: ...

    def __call__(self, queries: AsyncIterator[str]) -> AsyncIterator[dict[str, Any]]:
        async def hits(query: str) -> AsyncIterator[dict[str, Any]]:
            for hit in await self.search(query):
                yield hit

        return FlatMap(hits)(queries)


class SerperSearcher(Searcher):
    """Google results via Serper (needs SERPER_API_KEY).

    `hedge_after`: Serper's p50 is ~0.8s but outliers hit 3s+, so a request that hasn't answered by then is
    raced against a duplicate and the first reply wins.
    """

    def __init__(
        self,
        api_key: str | None = None,
        num: int = 10,
        timeout: float = 5.0,
        hedge_after: float | None = 1.2,
    ) -> None:
        self.api_key = api_key or os.environ.get("SERPER_API_KEY")
        self.num = num
        self.timeout = timeout
        self.hedge_after = hedge_after
        self._http: httpx.AsyncClient | None = None

    async def search(self, query: str) -> list[dict[str, Any]]:
        if not self.api_key:
            raise RuntimeError("SERPER_API_KEY is not set (pass api_key= or add it to .env)")
        if self.hedge_after is None:
            return await self._request(query)
        return await hedged(lambda: self._request(query), self.hedge_after)

    async def start(self) -> None:
        if self._http is None:
            self._http = httpx.AsyncClient(timeout=self.timeout)

    async def stop(self) -> None:
        if self._http is not None:
            await self._http.aclose()
            self._http = None

    async def _request(self, query: str) -> list[dict[str, Any]]:
        await self.start()
        response = await self._http.post(SERPER_URL, headers={"X-API-KEY": self.api_key}, json={"q": query, "num": self.num})
        response.raise_for_status()
        return [
            {"url": r["link"], "title": r.get("title", ""), "snippet": r.get("snippet", "")}
            for r in response.json().get("organic", [])
        ]


def not_blocked(domains: tuple[str, ...] = BLOCKED_DOMAINS) -> Pred:
    """Hit condition: False for hits on `domains` or their subdomains (m.facebook.com). Use it in a chain
    (`SerperSearcher() >> not_blocked() >> Take(5)`) or combine it (`not_blocked() & official`)."""
    return Pred(lambda hit: not is_blocked(hit["url"], domains))


def is_blocked(url: str, domains: tuple[str, ...] = BLOCKED_DOMAINS) -> bool:
    host = urlparse(url).netloc.lower()
    return any(host == d or host.endswith("." + d) for d in domains)


async def hedged(make: Any, after: float) -> Any:
    """Await `make()`; if it hasn't answered within `after` seconds, race a duplicate and take the first success."""
    first = asyncio.ensure_future(make())
    done, _ = await asyncio.wait({first}, timeout=after)
    if done:
        return first.result()
    racing = {first, asyncio.ensure_future(make())}
    error: BaseException | None = None
    try:
        while racing:
            done, racing = await asyncio.wait(racing, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                if task.exception() is None:
                    return task.result()
                error = task.exception()
        raise error  # type: ignore[misc]  # both attempts failed
    finally:
        for task in racing:
            task.cancel()
