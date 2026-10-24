"""The crawler role: url -> `Fetch` (the page `{"url", "title", "text"}` and the facts about the response), and
ways to combine crawlers."""

from __future__ import annotations

import inspect
from collections.abc import AsyncIterator
from typing import Any

from factassessor.crawlers.predicates import Fetch
from factassessor.pipeline import Map, Predicate, Step


class Crawler(Step):
    """Role: url -> `Fetch`. Implement `crawl(url)`: fetch the page (never raise: a failure is a `Fetch` without a
    page) and return `self.mark(f)`, which records this crawler's name and runs its `accept` rule into `f.usable`.
    The page is kept either way; `if f:` is "a page, and usable".

    `accept` is what counts as a usable page from this crawler. None (the base default): every page fetched is
    usable. As a step, every url is crawled concurrently and the usable pages come out in the order they finish,
    so each can be judged the moment it lands. `concurrency` bounds the urls read at once per stream (per claim in
    Verify): None reads them all at once, right when most hits fail to read (web search); with a source whose every
    page reads (AKD's full text) and a page cap, cap + a few spares keeps the reads the cap will cancel from ever
    starting and queueing ahead of other claims' (10 texts at once: 6,086 of 7,300 reads cancelled, p50 wait 15s)."""

    accept: Predicate | None = None  # the defaults for crawlers that never call super().__init__()
    concurrency: int | None = None

    def __init__(self, accept: Predicate | None = None, concurrency: int | None = None) -> None:
        self.accept = accept
        self.concurrency = concurrency

    async def crawl(self, url: str) -> Fetch:
        raise NotImplementedError(f"{type(self).__name__}.crawl")

    def mark(self, f: Fetch) -> Fetch:
        """`f` from this crawler, `usable` by its `accept` rule."""
        usable = True if self.accept is None else bool(self.accept(f))
        return f.model_copy(update={"crawler": type(self).__name__, "usable": usable})

    def __call__(self, urls: AsyncIterator[str]) -> AsyncIterator[dict[str, Any]]:
        async def page(url: str) -> dict[str, Any] | None:
            f = await self.crawl(url)
            return f.page if f else None  # dropped unless usable

        return Map(page, concurrency=self.concurrency)(urls)


class NoCrawler(Crawler):
    """Crawls nothing: for searchers whose hits already carry the evidence (`DocumentSearcher` passages), or to
    judge search snippets only."""

    async def crawl(self, url: str) -> Fetch:
        return Fetch(url=url, crawler=type(self).__name__)


class CascadedCrawler(Crawler):
    """Crawlers in a cascade: the first usable page wins (each crawler's own `accept` decides); after a failure,
    `when` decides whether the URL goes on to the next crawler, from what the failing one saw (its `Fetch`).
    `CascadedCrawler(HTTPXCrawler(), Crawl4AICrawler())` reads most pages the fast way and opens a browser only for
    the rest.

    `when` (sync or async predicate): None (the default) sends every failure on. A rule such as
    `(JavaScriptShell() & ~Paywalled()) | BotChallenge() | StatusIn(405)` (the same as `NeedsBrowser()`) sends a
    JavaScript shell or a bot check on but not a 404 or a paywall, so no browser tab is spent on a page no browser
    can read. The cascade has no `accept` of its own: it returns the `Fetch` of the crawler it stopped at, with that
    crawler's name and verdict. A cascade is a crawler, so it nests."""

    def __init__(self, *crawlers: Crawler, when: Predicate | None = None, concurrency: int | None = None) -> None:
        if not crawlers:
            raise ValueError("CascadedCrawler needs at least one crawler")
        super().__init__(concurrency=concurrency)
        self.crawlers = list(crawlers)
        self.when = when

    async def crawl(self, url: str) -> Fetch:
        for crawler in self.crawlers:
            f = await crawler.crawl(url)
            if f or crawler is self.crawlers[-1]:
                return f
            if self.when is not None and not await _holds(self.when, f):
                return f  # not worth the next crawler
        raise AssertionError("unreachable")

    def parts(self) -> list[Any]:
        return list(self.crawlers)


async def _holds(predicate: Predicate, f: Fetch) -> bool:
    decision = predicate(f)
    return bool(await decision if inspect.isawaitable(decision) else decision)
