"""HTTPXCrawler: pages and PDFs over plain HTTP (no browser, no JavaScript)."""

from __future__ import annotations

import asyncio
from typing import Any
from urllib.parse import urlsplit

import httpx

from factassessor.crawlers._base import Crawler
from factassessor.crawlers.predicates import Fetch, HasPage, MinWords
from factassessor.pipeline import Predicate
from factassessor.extract import extract
from factassessor.utils import cache


class HTTPXCrawler(Crawler):
    """A plain HTTP GET plus `extract()`: HTML pages and PDFs (by content type, or the bytes when the type is
    missing or generic). No browser, so no JavaScript: pages that build their content client-side come back empty
    (None); pair it with `CascadedCrawler` to catch those. PDFs need `fact-assessor[pdf]`.

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
        max_concurrent: int = 50,  # connections only (text is extracted after): 600 URLs at once, 50 had 319-327 pages by 20s vs 275 at 20
        max_bytes: int = 3_000_000,
        max_pdf_bytes: int = 50_000_000,  # a PDF cut short can't be read at all; theses run 25-35 MB
        pdf_timeout: float = 8.0,
        min_words: int = 100,
        accept: Predicate | None = None,
        max_per_host: int = 6,  # sites throttle bursts: PMC served reCAPTCHA pages mid-run (Scrapy's default per domain: 8)  # what counts as a page; default: a 2xx with min_words of text
    ) -> None:
        super().__init__(accept or HasPage() & MinWords(min_words))
        self.timeout = timeout
        self.connect_timeout = connect_timeout
        self.max_bytes = max_bytes  # stop reading huge pages; the useful text is near the top anyway
        self.max_pdf_bytes = max_pdf_bytes  # a cut-off PDF can't be read at all, so PDFs get a larger limit
        self.pdf_timeout = pdf_timeout
        self.min_words = min_words
        self._slots = asyncio.Semaphore(max_concurrent)
        self.max_per_host = max_per_host
        self._host_slots: dict[str, asyncio.Semaphore] = {}
        self._http: httpx.AsyncClient | None = None

    @cache(maxsize=2048, ttl=600)  # a page fetched and extracted once, shared by every claim (half of all hits repeat)
    async def crawl(self, url: str) -> Fetch:
        """One GET, and everything it showed: the page when a 2xx gave text (however short), the status, the type,
        the word count and the start of the body; `usable` by `accept`. Never raises.

        The slot and the deadline cover the network only: the text is extracted after both are released. Inside
        them, 600 URLs at once had downloads waiting a median 16.7s behind the CPU, and pages that had arrived
        timing out mid-parse."""
        try:
            # the site's slot first, then a connection: a request waiting on its site holds no connection meanwhile
            async with self._host_slot(url), self._slots:  # waiting for either doesn't count toward the deadline
                async with asyncio.timeout(self.timeout) as deadline:
                    fetched, body, encoding = await self._download(url, deadline)
        except Exception as exc:  # timeouts, DNS/connection errors
            return self.mark(Fetch(url=url, error=type(exc).__name__))
        if body is not None:
            got = await asyncio.to_thread(extract, body, fetched.content_type, encoding)
            if got:
                fetched = fetched.model_copy(update={"page": {"url": url, "title": got[0], "text": got[1]}, "words": len(got[1].split())})
        return self.mark(fetched)

    def _host_slot(self, url: str) -> asyncio.Semaphore:
        host = urlsplit(url).hostname or ""
        if host not in self._host_slots:
            self._host_slots[host] = asyncio.Semaphore(self.max_per_host)
        return self._host_slots[host]

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

    async def _download(self, url: str, deadline: asyncio.Timeout) -> tuple[Fetch, bytes | None, str | None]:
        """(the response's facts, the body to extract text from or None, its declared charset)."""
        await self.start()
        async with self._http.stream("GET", url) as response:
            kind = response.headers.get("content-type", "").split(";")[0].strip().lower()
            encoding = charset(response.headers.get("content-type", ""))  # only what the server declared; else the page's <meta>
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
        head = bytes(body[: self.HEAD_BYTES]).decode(encoding or "utf-8", "ignore") if "pdf" not in kind else ""
        fetched = Fetch(url=url, status=status, content_type=kind, body_head=head)
        return fetched, bytes(body[:limit]) if readable else None, encoding  # a single chunk can overshoot the limit


class ImpitCrawler(HTTPXCrawler):
    """`HTTPXCrawler` over impit, an HTTP client that looks like a real browser down to the TLS handshake (where
    bot checks look first; a browser user agent on httpx's own handshake is a mismatch). In the cascade after
    `HTTPXCrawler`: the honest bot gets the real page from sites that challenge browsers, this one gets past many
    403s that block bots. On 313 pages httpx couldn't read, the Firefox fingerprint read 62 (Chrome's: 23)."""

    def __init__(self, browser: str = "firefox", **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.browser = browser

    async def start(self) -> None:
        if self._http is None:
            import impit

            self._http = impit.AsyncClient(browser=self.browser, timeout=self.timeout, follow_redirects=True)

    async def stop(self) -> None:
        self._http = None  # impit's client holds no pool to close


def charset(content_type: str) -> str | None:
    """The charset a Content-Type header declares (`text/html; charset=UTF-8` -> "utf-8"), or None."""
    for part in content_type.split(";")[1:]:
        key, _, value = part.partition("=")
        if key.strip().lower() == "charset" and value.strip().strip('"'):
            return value.strip().strip('"').lower()
    return None


def _maybe_readable(content_type: str) -> bool:
    """HTML, XHTML, PDF, or a type that doesn't say (generic or missing): `extract()` decides from the bytes."""
    return not content_type or any(k in content_type for k in ("html", "xml", "pdf", "octet-stream"))
