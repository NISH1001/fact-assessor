"""The crawler role: url -> page `{"url", "title", "text"}` (clean plain text), and ways to combine crawlers."""

from __future__ import annotations

import inspect
from collections.abc import AsyncIterator
from typing import Any

from factassessor.crawlers.predicates import Fetch, HasPage
from factassessor.pipeline import Map, Predicate, Step


class Crawler(Step):
    """Role: url -> page. Implement `fetch(url) -> Fetch` (never raise: a failure is a `Fetch` without a page) and
    get `crawl(url)` from here: the page if it passes `accept`, else None. `accept` is what counts as a page from
    this crawler: `HasPage()` (a 2xx with text) unless the crawler or the caller sets another rule. A crawler that
    implements only `crawl()` still works everywhere. As a step, every url is crawled concurrently and pages come
    out in the order they finish, so each can be judged the moment it lands."""

    accept: Predicate = HasPage()  # the default for crawlers that never call super().__init__()

    def __init__(self, accept: Predicate | None = None) -> None:
        if accept is not None:
            self.accept = accept

    async def fetch(self, url: str) -> Fetch:
        raise NotImplementedError(f"{type(self).__name__}.fetch (or override crawl)")

    async def crawl(self, url: str) -> dict[str, Any] | None:
        f = await self.fetch(url)
        return f.page if self.accept(f) else None

    def __call__(self, urls: AsyncIterator[str]) -> AsyncIterator[dict[str, Any]]:
        return Map(self.crawl)(urls)


class NoCrawler(Crawler):
    """Crawls nothing: for searchers whose hits already carry the evidence (`DocumentSearcher` passages), or to
    judge search snippets only."""

    async def crawl(self, url: str) -> dict[str, Any] | None:
        return None


class CascadedCrawler(Crawler):
    """Crawlers in a cascade: the first page one of them accepts wins (each crawler's own `accept`); after a
    failure, `when` decides whether the URL goes on to the next crawler, from what the failing one saw (its `Fetch`).
    `CascadedCrawler(HTTPXCrawler(), Crawl4AICrawler())` reads most pages the fast way and opens a browser only for
    the rest.

    `when` (sync or async predicate): None (the default) sends every failure on. A rule such as
    `(JavaScriptShell() & ~Paywalled()) | BotChallenge() | StatusIn(405)` (the same as `NeedsBrowser()`) sends a
    JavaScript shell or a bot check on but not a 404 or a paywall, so no browser tab is spent on a page no browser
    can read. A crawler that implements only `crawl()` can't say why it failed, so after it the URL always goes on.
    A cascade is a crawler: it nests, and its own `fetch` reports why it failed."""

    def __init__(self, *crawlers: Crawler, when: Predicate | None = None) -> None:
        if not crawlers:
            raise ValueError("CascadedCrawler needs at least one crawler")
        super().__init__()
        self.crawlers = list(crawlers)
        self.when = when

    async def fetch(self, url: str) -> Fetch:
        for i, crawler in enumerate(self.crawlers):
            reports = _fetches(crawler)
            if reports:
                f = await crawler.fetch(url)
                if crawler.accept(f):
                    return f
            else:  # a crawl-only crawler: its page, or nothing (and no reasons)
                page = await crawler.crawl(url)
                if page is not None:
                    return Fetch(url=url, page=page, status=200, words=len(page.get("text", "").split()))
                f = Fetch(url=url)
            if i == len(self.crawlers) - 1:
                return f.model_copy(update={"page": None})
            if reports and self.when is not None and not await _holds(self.when, f):
                return f.model_copy(update={"page": None})  # not worth the next crawler; keep the reasons
        raise AssertionError("unreachable")

    def parts(self) -> list[Any]:
        return list(self.crawlers)


def _fetches(crawler: Crawler) -> bool:
    """Whether the crawler reports why it failed: it implements `fetch`, not only `crawl`."""
    return type(crawler).fetch is not Crawler.fetch


async def _holds(predicate: Predicate, f: Fetch) -> bool:
    decision = predicate(f)
    return bool(await decision if inspect.isawaitable(decision) else decision)
