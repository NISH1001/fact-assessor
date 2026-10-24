"""ResolvingCrawler: a resolver and a crawler as one crawler (free copies first, then the url itself)."""

from __future__ import annotations

from typing import Any

from factassessor.crawlers._base import Crawler
from factassessor.crawlers.predicates import Fetch
from factassessor.resolvers import locations


class ResolvingCrawler(Crawler):
    """url -> the first readable of its locations (`resolvers.locations`: a direct-PDF hit, the free copies the
    resolver finds, the url itself), read in order with `crawler` and cited under the url. A copy claims to be the
    full document, so it needs `min_copy_words` words (bot-check pages are shorter); the url's own page is taken as
    the crawler returns it. A failing resolver just means no copies.

    A crawler, so it nests: `CascadedCrawler(AKDDoiCrawler(), ResolvingCrawler(resolver, web))` reads a paper from
    AKD Scholar Search's full text and resolves only the hits AKD couldn't read, each once, for every web crawler in
    `web`. As `Verify(resolver=...)` does inline, but composable.
    """

    def __init__(self, resolver: Any, crawler: Crawler, min_copy_words: int = 300) -> None:
        if not callable(getattr(crawler, "crawl", None)):
            raise TypeError("ResolvingCrawler reads each location with a Crawler (crawl(url) -> Fetch)")
        super().__init__()
        self.resolver = resolver
        self.crawler = crawler
        self.min_copy_words = min_copy_words

    async def crawl(self, url: str) -> Fetch:
        try:
            candidates = await self.resolver.resolve(url)
        except Exception:
            candidates = []
        last = Fetch(url=url, crawler=type(self).__name__)
        for location in locations(url, candidates):
            f = await self.crawler.crawl(location)
            if f and (location == url or f.words >= self.min_copy_words):
                return f.model_copy(update={"url": url, "page": {**f.page, "url": url}})
            if location == url:
                last = f  # what the url itself gave: its status and error, for whoever comes after
        return last

    def parts(self) -> list[Any]:
        return [self.resolver, self.crawler]
