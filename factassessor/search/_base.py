"""The searcher role: query -> hits `{"url", "title", "snippet"}`, what to search (SearchType), and hit filters."""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from enum import Enum
from typing import Any
from urllib.parse import urlparse

from factassessor.pipeline import FlatMap, Pred, Step


class SearchType(str, Enum):
    """What to search, for any searcher that supports both: the web, or scholarly literature.

    SCIENCE: Serper -> Google Scholar; SearXNG -> its "science" engines (Google Scholar, arXiv, Semantic Scholar...).
    On scientific claims, scholarly search found the claim's source paper far more often than web search.
    """

    GENERAL = "general"
    SCIENCE = "science"

# Removed after search, before crawling: social media and forums are mostly reposts, opinions, and comments (and
# often crawl badly); video pages have no usable text. LinkedIn is kept as a primary source for people and orgs.
BLOCKED_DOMAINS = (
    "facebook.com", "fb.com", "instagram.com", "threads.net", "tiktok.com", "pinterest.com",
    "twitter.com", "x.com", "reddit.com", "quora.com",
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
