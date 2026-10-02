"""The crawler role: url -> page `{"url", "title", "text"}` (clean plain text), and ways to combine crawlers."""

from __future__ import annotations

import inspect
from collections.abc import AsyncIterator
from typing import Any

from factassessor.pipeline import Map, Step


class Crawler(Step):
    """Role: url -> page. Implement `crawl` (never raise; None on failure). As a step, every url is crawled
    concurrently and pages come out in the order they finish, so each can be judged the moment it lands."""

    async def crawl(self, url: str) -> dict[str, Any] | None:
        raise NotImplementedError(f"{type(self).__name__}.crawl")

    def __call__(self, urls: AsyncIterator[str]) -> AsyncIterator[dict[str, Any]]:
        return Map(self.crawl)(urls)


class NoCrawler(Crawler):
    """Crawls nothing: for searchers whose hits already carry the evidence (`DocumentSearcher` passages), or to
    judge search snippets only."""

    async def crawl(self, url: str) -> dict[str, Any] | None:
        return None


class FallbackCrawler(Crawler):
    """Try each crawler in turn; the first page with text wins. `FallbackCrawler(HTTPXCrawler(), Crawl4AICrawler())`
    fetches most pages the fast way and only opens a browser for the ones that need JavaScript (or failed).

    `escalate`: which failures of the first crawler go on to the others, a predicate over its `Fetch` (sync or
    async), e.g. `NeedsBrowser()`: a JavaScript shell or a bot check yes, a 404 or a paywall no, so no browser tab
    is spent on a page no browser can read. None (the default): every failure goes on, as before. The first crawler
    must report why it failed (`fetch(url) -> Fetch`, as `HTTPXCrawler` does)."""

    def __init__(self, *crawlers: Crawler, escalate: Any = None) -> None:
        if escalate is not None and not callable(getattr(crawlers[0] if crawlers else None, "fetch", None)):
            raise TypeError("FallbackCrawler(escalate=...) needs a first crawler with fetch(url) -> Fetch, like HTTPXCrawler")
        self.crawlers = list(crawlers)
        self.escalate = escalate

    async def crawl(self, url: str) -> dict[str, Any] | None:
        first, *rest = self.crawlers
        if (page := await first.crawl(url)) is not None:
            return page
        if self.escalate is not None:
            decision = self.escalate(await first.fetch(url))  # cached: the same request crawl() just made
            if inspect.isawaitable(decision):
                decision = await decision
            if not decision:
                return None
        for crawler in rest:
            if (page := await crawler.crawl(url)) is not None:
                return page
        return None

    def parts(self) -> list[Any]:
        return list(self.crawlers)
