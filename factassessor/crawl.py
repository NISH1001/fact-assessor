"""Crawlers: url -> page `{"url", "title", "text"}` (clean plain text). A step; failed pages are dropped.

`Crawler` is the role: implement `crawl(url)` (return None on failure).
- `Crawl4AICrawler`: a headless browser; renders JavaScript; ~0.6-1.6s per page.
- `HTTPXCrawler`: a plain HTTP fetch + HTML-to-text; no JavaScript; much faster for ordinary pages.
- `FallbackCrawler(fast, thorough)`: the first crawler that gets text wins.
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from typing import Any

import httpx

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


class HTTPXCrawler(Crawler):
    """A plain HTTP GET plus HTML-to-text (BeautifulSoup + lxml). No browser, so no JavaScript: pages that build
    their content client-side come back empty (None); pair it with `FallbackCrawler` to catch those.

    Timeouts: `connect_timeout` makes dead hosts fail fast, and `timeout` is a hard deadline on the whole fetch.
    (httpx's own timeout is per phase, e.g. per read, so a server that drips bytes could otherwise run far past it.)
    """

    USER_AGENT = "Mozilla/5.0 (compatible; fact-assessor/0.1; +https://github.com/NISH1001/fact-assessor)"

    def __init__(
        self, timeout: float = 2.5, connect_timeout: float = 1.0, max_concurrent: int = 20, max_bytes: int = 3_000_000
    ) -> None:
        self.timeout = timeout
        self.connect_timeout = connect_timeout
        self.max_bytes = max_bytes  # stop reading huge pages; the useful text is near the top anyway
        self._slots = asyncio.Semaphore(max_concurrent)
        self._http: httpx.AsyncClient | None = None

    async def crawl(self, url: str) -> dict[str, Any] | None:
        try:
            async with self._slots:
                return await asyncio.wait_for(self._fetch(url), self.timeout)
        except Exception:  # timeouts, DNS/connection errors, bad encodings
            return None

    async def start(self) -> None:
        if self._http is None:
            self._http = httpx.AsyncClient(
                timeout=httpx.Timeout(self.timeout, connect=self.connect_timeout),
                follow_redirects=True,
                headers={"User-Agent": self.USER_AGENT},
            )

    async def stop(self) -> None:
        if self._http is not None:
            await self._http.aclose()
            self._http = None

    async def _fetch(self, url: str) -> dict[str, Any] | None:
        await self.start()
        async with self._http.stream("GET", url) as response:
            if response.status_code >= 400 or "html" not in response.headers.get("content-type", ""):
                return None
            body = bytearray()
            async for piece in response.aiter_bytes():
                body += piece
                if len(body) >= self.max_bytes:
                    break
            encoding = response.encoding or "utf-8"
        html = bytes(body[: self.max_bytes]).decode(encoding, errors="replace")  # a single chunk can overshoot
        title, text = await asyncio.to_thread(_html_to_text, html)
        text = clean_text(text)
        return {"url": url, "title": title, "text": text} if text else None


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


_NOT_CONTENT = ["script", "style", "noscript", "template", "svg", "iframe", "nav", "header", "footer", "aside", "form"]


def _html_to_text(html: str) -> tuple[str, str]:
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "lxml")
    title = soup.title.get_text(strip=True) if soup.title else ""
    for tag in soup(_NOT_CONTENT):
        tag.decompose()
    return title, soup.get_text("\n")
