"""Crawl4AICrawler: pages through a headless browser (renders JavaScript)."""

from __future__ import annotations

import asyncio
from typing import Any

from factassessor.crawlers._base import Crawler
from factassessor.crawlers.predicates import Fetch, HasPage
from factassessor.pipeline import Predicate
from factassessor.passages import clean_text
from factassessor.utils import cache


class Crawl4AICrawler(Crawler):
    """crawl4ai with one shared headless browser.

    `max_concurrent` is a limit on the browser, shared by every stream through this crawler (all claims, all
    checks), not per call.
    """

    HEAD_CHARS = 16_000  # the start of the rendered HTML is kept for markers (bot checks), like HTTPXCrawler's body

    def __init__(self, timeout: float = 2.5, max_concurrent: int = 10, accept: Predicate | None = None) -> None:
        super().__init__(accept or HasPage())  # default: a 2xx with any text
        self.timeout = timeout  # good pages crawl in ~0.6-1.6s; a 6s timeout let one dead site set the latency
        self._slots = asyncio.Semaphore(max_concurrent)
        self._browser: Any = None
        self._browser_lock = asyncio.Lock()

    @cache(maxsize=2048, ttl=600)  # a page rendered once, shared by every claim
    async def crawl(self, url: str) -> Fetch:
        """One render, and what it showed: the text (when any came out), the status and the start of the HTML;
        `usable` by `accept`. Never raises. crawl4ai reports success for pages that loaded with an error status (403
        blocks, 404s, "HTTP 503 temporarily unavailable"), so the status is kept and `accept` (HasPage: a 2xx) marks
        those unusable."""
        from crawl4ai import CacheMode, CrawlerRunConfig, DefaultMarkdownGenerator

        config = CrawlerRunConfig(
            # plain text: markdown links/images were ~2/3 of the characters and wasted the judge's token budget
            markdown_generator=DefaultMarkdownGenerator(options={"ignore_links": True, "ignore_images": True}),
            cache_mode=CacheMode.BYPASS,
            wait_until="domcontentloaded",  # don't wait for ads/trackers to finish loading
            page_timeout=int(self.timeout * 1000),
            excluded_tags=["nav", "footer", "header", "aside", "form", "script", "style"],
            exclude_all_images=True,
            verbose=False,
        )
        try:
            browser = await self._start_browser()
            async with self._slots:
                result = await asyncio.wait_for(browser.arun(url, config=config), self.timeout)
        except Exception as exc:  # timeouts, dead hosts, browser hiccups
            return self.mark(Fetch(url=url, error=type(exc).__name__))
        status = getattr(result, "status_code", None)
        if status is None and result.success:  # crawl4ai sometimes leaves it unset for a page that loaded fine
            status = 200
        text = clean_text(result.markdown.raw_markdown) if result.success and result.markdown else ""
        page = {"url": url, "title": (result.metadata or {}).get("title") or "", "text": text} if text else None
        html = getattr(result, "html", "") or ""
        return self.mark(Fetch(url=url, page=page, status=status, content_type="text/html", words=len(text.split()),
                               body_head=html[: self.HEAD_CHARS] if isinstance(html, str) else ""))

    async def start(self) -> None:
        await self._start_browser()

    async def stop(self) -> None:
        if self._browser is not None:
            await self._browser.close()
            self._browser = None

    async def _start_browser(self) -> Any:
        async with self._browser_lock:
            if self._browser is None:
                from crawl4ai import AsyncWebCrawler, BrowserConfig

                browser = AsyncWebCrawler(config=BrowserConfig(headless=True, text_mode=True, light_mode=True, verbose=False))
                await browser.start()
                self._browser = browser
        return self._browser
