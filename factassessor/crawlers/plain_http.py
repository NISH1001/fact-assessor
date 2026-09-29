"""HTTPXCrawler: pages and PDFs over plain HTTP (no browser, no JavaScript)."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from typing import Any

import httpx

from factassessor.crawlers._base import Crawler
from factassessor.extract import extract
from factassessor.resolvers import Resolver


class HTTPXCrawler(Crawler):
    """A plain HTTP GET plus `extract()`: HTML pages and PDFs (by content type, or the bytes when the type is
    missing or generic). No browser, so no JavaScript: pages that build their content client-side come back empty
    (None); pair it with `FallbackCrawler` to catch those. PDFs need `fact-assessor[pdf]`.

    `resolvers` find where a document can be read in full (`ArxivResolver`, `OpenAlexResolver`, or anything with
    `async resolve(url) -> list[str]`). Their candidates are tried first, in order, then the URL itself; the first
    with enough words wins, and it is returned under the original URL (it stays that search hit's evidence).
    A resolved candidate claims to be the full document, so it needs `min_paper_words` (a stub or bot-check page
    is shorter: Wiley's and HAL's are ~180 words); the URL itself is an ordinary page and needs `min_words`.
    Ordinary pages cost nothing extra: resolvers answer `[]` for them without a request.

        FallbackCrawler(HTTPXCrawler(resolvers=[ArxivResolver(), OpenAlexResolver()]), Crawl4AICrawler())

    Timeouts: `connect_timeout` makes dead hosts fail fast, and `timeout` is a hard deadline on each page fetch
    (httpx's own timeout is per phase, e.g. per read, so a server that drips bytes could otherwise run far past it);
    resolved candidates, often PDFs of several MB, get `paper_timeout` each.
    """

    USER_AGENT = "Mozilla/5.0 (compatible; fact-assessor/0.1; +https://github.com/NISH1001/fact-assessor)"

    def __init__(
        self,
        timeout: float = 2.5,
        connect_timeout: float = 1.0,
        max_concurrent: int = 20,
        max_bytes: int = 3_000_000,
        max_pdf_bytes: int = 20_000_000,
        resolvers: Sequence[Resolver] = (),
        min_words: int = 100,
        min_paper_words: int = 300,
        paper_timeout: float = 10.0,
    ) -> None:
        self.timeout = timeout
        self.connect_timeout = connect_timeout
        self.max_bytes = max_bytes  # stop reading huge pages; the useful text is near the top anyway
        self.max_pdf_bytes = max_pdf_bytes  # a cut-off PDF can't be read at all, so PDFs get a larger limit
        self.resolvers = list(resolvers)
        # fewer words is not the document: on 2,748 crawled pages, those under 100 were login walls, "Loading..."
        # shells and browser checks. (Scripts written without spaces, like Chinese, count low.)
        self.min_words = min_words
        self.min_paper_words = min_paper_words
        self.paper_timeout = paper_timeout
        self._slots = asyncio.Semaphore(max_concurrent)
        self._http: httpx.AsyncClient | None = None

    async def crawl(self, url: str) -> dict[str, Any] | None:
        async with self._slots:
            for candidate, min_words, deadline in await self._candidates(url):
                try:
                    page = await asyncio.wait_for(self._fetch(candidate), deadline)
                except Exception:  # timeouts, DNS/connection errors, bad encodings
                    continue
                if page and len(page["text"].split()) >= min_words:
                    return {**page, "url": url}
        return None

    async def _candidates(self, url: str) -> list[tuple[str, int, float]]:
        """(url to fetch, words it needs, its deadline): every resolver's candidates in order, then `url` itself."""
        found: dict[str, tuple[int, float]] = {}
        for resolver in self.resolvers:
            try:
                resolved = await resolver.resolve(url)
            except Exception:  # a resolver should never raise; if one does, it just has no candidates
                resolved = []
            for candidate in resolved:
                found.setdefault(candidate, (self.min_paper_words, self.paper_timeout))
        found.setdefault(url, (self.min_words, self.timeout))
        return [(u, words, deadline) for u, (words, deadline) in found.items()]

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
        for resolver in self.resolvers:  # resolvers with their own HTTP client (OpenAlexResolver)
            if (stop := getattr(resolver, "stop", None)) is not None:
                await stop()

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
