"""The searcher role: query -> hits (`Hit`: url, title, snippet), what to search (SearchType), hit filters, and how
several searches' rankings merge (`HitOrder`, `merge`)."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable
from enum import Enum, StrEnum
from itertools import zip_longest
from typing import Any, TypedDict
from urllib.parse import urlparse

from factassessor.pipeline import FlatMap, Predicate, Step


class SearchType(str, Enum):
    """What to search, for any searcher that supports both: the web, or scholarly literature.

    SCIENCE: Serper -> Google Scholar; SearXNG -> its "science" engines (Google Scholar, arXiv, Semantic Scholar...).
    On scientific claims, scholarly search found the claim's source paper far more often than web search.
    """

    GENERAL = "general"
    SCIENCE = "science"

# Never used as evidence: social media and forums are mostly reposts, opinions, and comments (and often crawl
# badly); video pages have no usable text; LinkedIn posts are the same, and its pages block crawlers. Dropped
# after search by `not_blocked()`, and excluded in the query itself where the searcher can (`SerperSearcher`).
# Ordered by how often they show up in web results, so a searcher that can only fit some keeps the leakiest.
BLOCKED_DOMAINS = (
    "reddit.com", "youtube.com", "facebook.com", "quora.com", "twitter.com", "x.com", "linkedin.com",
    "instagram.com", "pinterest.com", "tiktok.com", "tumblr.com", "youtu.be", "fb.com", "threads.com", "threads.net",
)


class Hit(TypedDict):
    """One search result, what every searcher yields: the judge reads `snippet`, the crawler fetches `url`."""

    url: str
    title: str
    snippet: str


Ranking = list[Hit]  # one search's hits, best first


class HitOrder(StrEnum):
    """How a claim's rankings (its own search's, then the text's source queries') merge into the one list its pages
    are read in, so the page cap (`pages_per_claim`) decides between the searches.

    ORIGINAL (default until the Serper runs are re-measured with the others): the claim's own hits, then each
    source query's: the order of every run before 2026-10-24. With every page readable (AKD's full text) a cap of 10
    filled with the claim's own hits before the source paper (cap 30 scored 0.840 vs 0.672 at cap 10).
    INTERLEAVED: rank 1 of every search, then rank 2...: a cap reaches each search's best hits.
    RRF: reciprocal rank fusion, the sum of 1 / (60 + rank) over the searches a url appears in: a paper that several
    searches return (the claim's and a source query's: likely the source paper) comes first; ties fall back to the
    interleaved order.
    """

    ORIGINAL = "original"
    INTERLEAVED = "interleaved"
    RRF = "rrf"


def merge(rankings: list[Ranking], order: HitOrder | str = HitOrder.ORIGINAL) -> Ranking:
    """The rankings as one list in `order`, each url once (its first appearance)."""
    return list({h["url"]: h for h in _MERGE[HitOrder(order)](rankings)}.values())


def _interleaved(rankings: list[Ranking]) -> list[Hit]:
    return [h for rank in zip_longest(*rankings) for h in rank if h is not None]


def _original(rankings: list[Ranking]) -> list[Hit]:
    return [h for ranking in rankings for h in ranking]


def _rrf(rankings: list[Ranking], k: int = 60) -> list[Hit]:
    score: dict[str, float] = {}
    for ranking in rankings:
        for rank, h in enumerate(ranking):
            score[h["url"]] = score.get(h["url"], 0.0) + 1 / (k + rank)
    by_url = {h["url"]: h for h in _interleaved(rankings)}  # first appearance in interleaved order breaks ties
    return [by_url[url] for url in sorted(by_url, key=lambda url: -score[url])]


_MERGE: dict[HitOrder, Callable[[list[Ranking]], list[Hit]]] = {
    HitOrder.ORIGINAL: _original, HitOrder.INTERLEAVED: _interleaved, HitOrder.RRF: _rrf,
}


class Searcher(Step):
    """Role: query -> hits. Implement `search`; streaming, concurrency, and chaining come from here."""

    async def search(self, query: str) -> list[Hit]:
        raise NotImplementedError(f"{type(self).__name__}.search")

    def __call__(self, queries: AsyncIterator[str]) -> AsyncIterator[Hit]:
        async def hits(query: str) -> AsyncIterator[Hit]:
            for hit in await self.search(query):
                yield hit

        return FlatMap(hits)(queries)


def not_blocked(domains: tuple[str, ...] = BLOCKED_DOMAINS) -> Predicate:
    """Hit condition: False for hits on `domains` or their subdomains (m.facebook.com). Use it in a chain
    (`SerperSearcher() >> not_blocked() >> Take(5)`) or combine it (`not_blocked() & official`)."""
    return Predicate(lambda hit: not is_blocked(hit["url"], domains))


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
