"""OpenAccessCrawler: papers and PDFs (arXiv, direct PDF links, DOIs via their open-access copies)."""

from __future__ import annotations

import asyncio
import re
from typing import Any
from urllib.parse import urlparse

import httpx

from factassessor.crawlers._base import Crawler
from factassessor.crawlers.plain_http import HTTPXCrawler
from factassessor.extract import extract
from factassessor.resolvers import doi_in

_ARXIV = re.compile(r"arxiv\.org/(?:abs|pdf|html)/(\d{4}\.\d{4,5}|[a-z-]+(?:\.[a-z]{2})?/\d{7})(?:v\d+)?", re.IGNORECASE)
_PDF_PATH = re.compile(r"(?:\.pdf$|/(?:e?pdf|pdfdirect)(?:/|$))", re.IGNORECASE)


def arxiv_pdf_url(url: str) -> str | None:
    """The full-paper PDF for any arXiv link (`/abs/`, `/html/`, `/pdf/`, with or without a version), or None.
    The rule is akd-core's ArxivResolver (NASA-IMPACT/akd-core)."""
    match = _ARXIV.search(url)
    return f"https://arxiv.org/pdf/{match.group(1)}" if match else None


def looks_like_pdf(url: str) -> bool:
    """A direct PDF link, judging by the URL (`.pdf`, `/pdf/`, `/epdf/`, `/pdfdirect/`)."""
    return bool(_PDF_PATH.search(urlparse(url).path))


class OpenAccessCrawler(Crawler):
    """Papers and PDFs: the full text of a scholarly link, read directly. It handles, in this order:

    - arXiv links (`/abs/`, `/html/`): the full paper at `arxiv.org/pdf/<id>`, not the abstract page;
    - direct PDF links (`.pdf`, `/pdf/`...): downloaded and extracted;
    - links with a DOI (publisher pages): OpenAlex (free; DOI lookups cost nothing, no key needed) lists the
      open-access copies, PDF or HTML, and the first readable one is used.

    Anything else (ordinary web pages) returns None at once, with no request: put it first and let the browser
    take the rest, `FallbackCrawler(OpenAccessCrawler(), Crawl4AICrawler())`. The text comes back under the
    original URL, so it stays that search hit's evidence; fewer than `min_words` words counts as a bot-check page, not a
    paper. Needs `fact-assessor[pdf]` for PDFs.

    Why: on a scientific eval set, Google Scholar found a claim's source paper 2.5x as often as web search (49% vs
    19%), but 61% of its hits are PDFs, and Wiley and IOP block headless browsers (0 of 15 crawled).
    """

    OPENALEX = "https://api.openalex.org/works/doi:"

    def __init__(
        self, timeout: float = 10.0, max_concurrent: int = 5, max_bytes: int = 20_000_000, min_words: int = 300
    ) -> None:
        self.timeout = timeout  # a lookup plus a PDF download: slower than a page, so it gets its own deadline
        self.max_bytes = max_bytes
        # fewer is a bot-check or error page, not a paper: on 2,748 crawled pages, Wiley's, Science's, and HAL's
        # "Making sure you're not a bot!" pages are ~180 words; a paper is thousands. Words, not characters: links
        # and markup leftovers inflate character counts. (Scripts written without spaces, like Chinese, count low.)
        self.min_words = min_words
        self._slots = asyncio.Semaphore(max_concurrent)  # OpenAlex asks for at most 10 requests/s
        self._http: httpx.AsyncClient | None = None

    async def crawl(self, url: str) -> dict[str, Any] | None:
        direct = [u for u in (arxiv_pdf_url(url), url if looks_like_pdf(url) else None) if u]
        doi = doi_in(url)
        if not direct and doi is None:
            return None  # an ordinary web page: not ours
        try:
            async with self._slots:
                return await asyncio.wait_for(self._fetch(url, list(dict.fromkeys(direct)), doi), self.timeout)
        except Exception:  # timeouts, no free copy, unreadable PDFs
            return None

    async def start(self) -> None:
        if self._http is None:
            self._http = httpx.AsyncClient(
                timeout=self.timeout, follow_redirects=True, headers={"User-Agent": HTTPXCrawler.USER_AGENT}
            )

    async def stop(self) -> None:
        if self._http is not None:
            await self._http.aclose()
            self._http = None

    async def _fetch(self, url: str, direct: list[str], doi: str | None) -> dict[str, Any] | None:
        await self.start()
        for target in direct:  # arXiv's PDF, or the PDF link itself
            if text := await self._read(target):
                return {"url": url, "title": "", "text": text}
        if doi is None:
            return None
        response = await self._http.get(self.OPENALEX + doi)
        if response.status_code != 200:
            return None
        work = response.json()
        # every open-access copy, the "best" first: it's often the publisher's own PDF, which may block us while a
        # repository or author copy doesn't. PDFs before landing pages.
        copies = [work.get("best_oa_location") or {}] + [loc for loc in work.get("locations") or [] if loc.get("is_oa")]
        targets = [c.get("pdf_url") for c in copies] + [c.get("landing_page_url") for c in copies]
        targets.append((work.get("open_access") or {}).get("oa_url"))
        for target in dict.fromkeys(t for t in targets if t):
            if text := await self._read(target):
                return {"url": url, "title": work.get("title") or work.get("display_name") or "", "text": text}
        return None

    async def _read(self, target: str) -> str | None:
        response = await self._http.get(target)
        if not 200 <= response.status_code < 300:
            return None
        body, kind = response.content[: self.max_bytes], response.headers.get("content-type", "")
        got = await asyncio.to_thread(extract, body, kind, response.charset_encoding)
        return got[1] if got and len(got[1].split()) >= self.min_words else None
