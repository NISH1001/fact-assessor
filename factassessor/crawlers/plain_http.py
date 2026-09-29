"""HTTPXCrawler: pages over plain HTTP (no browser, no JavaScript), and the HTML-to-text used by other crawlers."""

from __future__ import annotations

import asyncio
from typing import Any

import httpx

from factassessor.crawlers._base import Crawler
from factassessor.passages import clean_text


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


_NOT_CONTENT = ["script", "style", "noscript", "template", "svg", "iframe", "nav", "header", "footer", "aside", "form"]


def _html_to_text(html: str) -> tuple[str, str]:
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "lxml")
    title = soup.title.get_text(strip=True) if soup.title else ""
    for tag in soup(_NOT_CONTENT):
        tag.decompose()
    return title, soup.get_text("\n")
