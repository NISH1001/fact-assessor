"""HTTPXCrawler: pages and PDFs over plain HTTP (no browser, no JavaScript)."""

from __future__ import annotations

import asyncio
from typing import Any

import httpx

from factassessor.crawlers._base import Crawler
from factassessor.crawlers.predicates import Fetch
from factassessor.extract import extract
from factassessor.utils import cache


class HTTPXCrawler(Crawler):
    """A plain HTTP GET plus `extract()`: HTML pages and PDFs (by content type, or the bytes when the type is
    missing or generic). No browser, so no JavaScript: pages that build their content client-side come back empty
    (None); pair it with `FallbackCrawler` to catch those. PDFs need `fact-assessor[pdf]`.

    Less than `min_words` is not a page: on 2,748 crawled pages, those under 100 words were login walls,
    "Loading..." shells and browser checks. (Scripts written without spaces, like Chinese, count low.)

    Timeouts: `connect_timeout` makes dead hosts fail fast, and `timeout` is a hard deadline on the whole fetch
    (httpx's own timeout is per phase, e.g. per read, so a server that drips bytes could otherwise run far past it).
    Once the response turns out to be a PDF, the deadline becomes `pdf_timeout`: a paper is several MB.
    """

    USER_AGENT = "Mozilla/5.0 (compatible; fact-assessor/0.1; +https://github.com/NISH1001/fact-assessor)"
    HEAD_BYTES = 16_000  # the start of every response is kept for markers: a bot check's "Just a moment..."

    def __init__(
        self,
        timeout: float = 2.5,
        connect_timeout: float = 1.0,
        max_concurrent: int = 20,
        max_bytes: int = 3_000_000,
        max_pdf_bytes: int = 20_000_000,
        pdf_timeout: float = 8.0,
        min_words: int = 100,
    ) -> None:
        self.timeout = timeout
        self.connect_timeout = connect_timeout
        self.max_bytes = max_bytes  # stop reading huge pages; the useful text is near the top anyway
        self.max_pdf_bytes = max_pdf_bytes  # a cut-off PDF can't be read at all, so PDFs get a larger limit
        self.pdf_timeout = pdf_timeout
        self.min_words = min_words
        self._slots = asyncio.Semaphore(max_concurrent)
        self._http: httpx.AsyncClient | None = None

    @cache(maxsize=2048, ttl=600)  # a page fetched and extracted once, shared by every claim (half of all hits repeat)
    async def fetch(self, url: str) -> Fetch:
        """One GET, and everything it showed: the page when a 2xx gave text (however short), the status, the type,
        the word count and the start of the body, for the predicates to decide on. Never raises."""
        try:
            async with self._slots:  # waiting for a slot doesn't count toward the deadline
                async with asyncio.timeout(self.timeout) as deadline:
                    return await self._fetch(url, deadline)
        except Exception as exc:  # timeouts, DNS/connection errors, bad encodings
            return Fetch(url=url, error=type(exc).__name__)

    async def crawl(self, url: str) -> dict[str, Any] | None:
        """The page, or None: a 2xx with at least `min_words` of text (the crawler's own rule, as before)."""
        f = await self.fetch(url)
        return f.page if f.page is not None and f.status is not None and f.status < 400 and f.words >= self.min_words else None

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

    async def _fetch(self, url: str, deadline: asyncio.Timeout) -> Fetch:
        await self.start()
        async with self._http.stream("GET", url) as response:
            kind = response.headers.get("content-type", "").split(";")[0].strip().lower()
            status = response.status_code
            readable = status < 400 and _maybe_readable(kind)
            maybe_pdf = "pdf" in kind or "octet-stream" in kind
            limit = (self.max_pdf_bytes if maybe_pdf else self.max_bytes) if readable else self.HEAD_BYTES
            if readable and maybe_pdf:  # a paper: several MB, so more time than a page
                deadline.reschedule(asyncio.get_running_loop().time() + self.pdf_timeout)
            body = bytearray()
            async for piece in response.aiter_bytes():
                body += piece
                if len(body) >= limit:
                    break
            encoding = response.charset_encoding  # only what the server declared; else the page's own <meta>
        head = bytes(body[: self.HEAD_BYTES]).decode(encoding or "utf-8", "ignore") if "pdf" not in kind else ""
        fetched = Fetch(url=url, status=status, content_type=kind, body_head=head)
        if not readable:
            return fetched
        got = await asyncio.to_thread(extract, bytes(body[:limit]), kind, encoding)  # a single chunk can overshoot
        if not got:
            return fetched
        return fetched.model_copy(update={"page": {"url": url, "title": got[0], "text": got[1]}, "words": len(got[1].split())})


def _maybe_readable(content_type: str) -> bool:
    """HTML, XHTML, PDF, or a type that doesn't say (generic or missing): `extract()` decides from the bytes."""
    return not content_type or any(k in content_type for k in ("html", "xml", "pdf", "octet-stream"))
