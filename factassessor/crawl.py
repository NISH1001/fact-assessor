"""Crawlers: url -> page `{"url", "title", "text"}` (clean plain text). A step; failed pages are dropped.

`Crawler` is the role: implement `crawl(url)` (return None on failure). `Crawl4AICrawler` uses a headless browser.
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from typing import Any

from factassessor.passages import clean_text
from factassessor.pipeline import Map, Step


class Crawler(Step, ABC):
    """Role: url -> page. Implement `crawl` (never raise; None on failure). As a step, every url is crawled
    concurrently and pages come out in the order they finish, so each can be judged the moment it lands."""

    @abstractmethod
    async def crawl(self, url: str) -> dict[str, Any] | None: ...

    def __call__(self, urls: AsyncIterator[str]) -> AsyncIterator[dict[str, Any]]:
        return Map(self.crawl)(urls)


class Crawl4AICrawler(Crawler):
    """crawl4ai with one shared headless browser.

    `max_concurrent` is a limit on the browser, shared by every stream through this crawler (all claims, all
    checks), not per call.
    """

    def __init__(self, timeout: float = 2.5, max_concurrent: int = 10) -> None:
        self.timeout = timeout  # good pages crawl in ~0.6-1.6s; a 6s timeout let one dead site set the latency
        self._slots = asyncio.Semaphore(max_concurrent)
        self._browser: Any = None
        self._browser_lock = asyncio.Lock()

    async def crawl(self, url: str) -> dict[str, Any] | None:
        """One page, or None on failure/timeout. Never raises."""
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
        except Exception:  # timeouts, dead hosts, browser hiccups
            return None
        text = clean_text(result.markdown.raw_markdown) if result.success and result.markdown else ""
        if not text:
            return None
        return {"url": url, "title": (result.metadata or {}).get("title", ""), "text": text}

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
