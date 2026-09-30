"""The crawler role: url -> page `{"url", "title", "text"}` (clean plain text), and ways to combine crawlers."""

from __future__ import annotations

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
    fetches most pages the fast way and only opens a browser for the ones that need JavaScript (or failed)."""

    def __init__(self, *crawlers: Crawler) -> None:
        self.crawlers = list(crawlers)

    async def crawl(self, url: str) -> dict[str, Any] | None:
        for crawler in self.crawlers:
            if (page := await crawler.crawl(url)) is not None:
                return page
        return None

    def parts(self) -> list[Any]:
        return list(self.crawlers)
