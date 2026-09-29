"""HTTPXCrawler: pages and PDFs over plain HTTP (no browser, no JavaScript)."""

from __future__ import annotations

import asyncio
from typing import Any

import httpx

from factassessor.crawlers._base import Crawler
from factassessor.extract import extract


class HTTPXCrawler(Crawler):
    """A plain HTTP GET plus `extract()`: HTML pages and PDFs (by content type, or the bytes when the type is
    missing or generic). No browser, so no JavaScript: pages that build their content client-side come back empty
    (None); pair it with `FallbackCrawler` to catch those. PDFs need `fact-assessor[pdf]`.

    Timeouts: `connect_timeout` makes dead hosts fail fast, and `timeout` is a hard deadline on the whole fetch.
    (httpx's own timeout is per phase, e.g. per read, so a server that drips bytes could otherwise run far past it.)
    """

    USER_AGENT = "Mozilla/5.0 (compatible; fact-assessor/0.1; +https://github.com/NISH1001/fact-assessor)"

    def __init__(
        self,
        timeout: float = 2.5,
        connect_timeout: float = 1.0,
        max_concurrent: int = 20,
        max_bytes: int = 3_000_000,
        max_pdf_bytes: int = 20_000_000,
    ) -> None:
        self.timeout = timeout
        self.connect_timeout = connect_timeout
        self.max_bytes = max_bytes  # stop reading huge pages; the useful text is near the top anyway
        self.max_pdf_bytes = max_pdf_bytes  # a cut-off PDF can't be read at all, so PDFs get a larger limit
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
            kind = response.headers.get("content-type", "").lower()
            if response.status_code >= 400 or not _maybe_readable(kind):
                return None
            limit = self.max_pdf_bytes if "pdf" in kind or "octet-stream" in kind else self.max_bytes
            body = bytearray()
            async for piece in response.aiter_bytes():
                body += piece
                if len(body) >= limit:
                    break
            encoding = response.charset_encoding  # only what the server declared; else the page's own <meta>
        got = await asyncio.to_thread(extract, bytes(body[:limit]), kind, encoding)  # a single chunk can overshoot
        return {"url": url, "title": got[0], "text": got[1]} if got else None


def _maybe_readable(content_type: str) -> bool:
    """HTML, XHTML, PDF, or a type that doesn't say (generic or missing): `extract()` decides from the bytes."""
    return not content_type or any(k in content_type for k in ("html", "xml", "pdf", "octet-stream"))
