"""Resolvers: url -> other urls where the same document can be read in full (a paper's HTML or PDF, a free copy).

`Resolver` is the role: anything with `async resolve(url) -> list[str]`, best candidate first. A resolver only
proposes: it never fetches the documents or checks that they exist (that would cost a request to probe and another
to read). The crawler tries the candidates in order and keeps the first with real text, so a 404 or a bot-check
page just moves on to the next. A URL that isn't the resolver's kind gets `[]` at once, with no request.

- `ArxivResolver`: any arXiv link -> its full paper, `arxiv.org/html/<id>` then `arxiv.org/pdf/<id>`; no request.
- `OpenAlexResolver`: a URL with a DOI -> the paper's open-access copies (one free OpenAlex lookup).
"""

from __future__ import annotations

import asyncio
import re
from typing import Any, Literal, Protocol, runtime_checkable
from urllib.parse import unquote

import httpx


@runtime_checkable
class Resolver(Protocol):
    """Role: url -> candidate urls for the same document, best first; `[]` if it isn't this resolver's kind.
    Never raises: a failed lookup is `[]`."""

    async def resolve(self, url: str) -> list[str]: ...


# --- arXiv -------------------------------------------------------------------------------------------------------

_ARXIV_ID = r"(\d{4}\.\d{4,5}(?:v\d+)?|[a-z-]+(?:\.[a-z]{2})?/\d{7}(?:v\d+)?)"
_ARXIV = re.compile(r"arxiv\.org/(?:abs|pdf|html)/" + _ARXIV_ID, re.IGNORECASE)
_ARXIV_DOI = re.compile(r"10\.48550/arxiv\." + _ARXIV_ID, re.IGNORECASE)


def arxiv_id(url: str) -> str | None:
    """The arXiv id in any arXiv link or arXiv DOI, with its version if it has one, or None:
    `arxiv.org/abs/2104.10311v2` -> `2104.10311v2`; `doi.org/10.48550/arXiv.2104.10311` -> `2104.10311`."""
    match = _ARXIV.search(url) or _ARXIV_DOI.search(unquote(url))
    return match.group(1) if match else None


class ArxivResolver:
    """Any arXiv link (`/abs/`, `/pdf/`, `/html/`, arXiv DOIs) -> the full paper, not the abstract page.

    `prefer="html"` (default) proposes arXiv's HTML first, then the PDF: the HTML keeps paragraphs and sections,
    with no two-column line mixing or hyphen-split words, but exists mainly for papers since late 2023; for the
    others the HTML request fails and the crawler reads the PDF. `prefer="pdf"` reverses the order.
    The version in the link (`v2`) is kept, so the text is the version search found.
    """

    def __init__(self, prefer: Literal["html", "pdf"] = "html") -> None:
        if prefer not in ("html", "pdf"):
            raise ValueError(f"prefer must be 'html' or 'pdf', not {prefer!r}")
        self.prefer = prefer

    async def resolve(self, url: str) -> list[str]:
        if (id := arxiv_id(url)) is None:
            return []
        html, pdf = f"https://arxiv.org/html/{id}", f"https://arxiv.org/pdf/{id}"
        return [html, pdf] if self.prefer == "html" else [pdf, html]


# --- DOIs, via OpenAlex ------------------------------------------------------------------------------------------

_DOI = re.compile(r"10\.\d{4,9}/[^\s?#&]+", re.IGNORECASE)
_DOI_SUFFIXES = ("/pdf", "/epdf", "/full", "/fulltext", "/abstract", "/abs", "/meta", "/html")


def doi_in(url: str) -> str | None:
    """The DOI in a publisher or doi.org URL (lowercased), or None: `.../doi/full/10.1002/rse2.203` -> `10.1002/rse2.203`."""
    match = _DOI.search(unquote(url))
    if not match:
        return None
    doi = match.group(0).rstrip("./")
    while (suffix := next((x for x in _DOI_SUFFIXES if doi.lower().endswith(x)), None)) is not None:
        doi = doi[: -len(suffix)]
    return doi.lower()


class OpenAlexResolver:
    """A URL with a DOI (publisher pages, doi.org links) -> the paper's open-access copies: every open-access
    location OpenAlex knows, PDFs before landing pages, the "best" one first.

    The best copy is often the publisher's own PDF, which may serve a bot-check page while a repository or author
    copy listed under `locations` has the real text, so all of them are proposed. DOI lookups are free (no key);
    arXiv DOIs are left to `ArxivResolver`. One request per DOI; a failed lookup is `[]`.
    """

    URL = "https://api.openalex.org/works/doi:"
    USER_AGENT = "fact-assessor (+https://github.com/NISH1001/fact-assessor)"

    def __init__(self, timeout: float = 5.0, max_concurrent: int = 5) -> None:
        self.timeout = timeout
        self._slots = asyncio.Semaphore(max_concurrent)  # OpenAlex asks for at most 10 requests/s
        self._http: httpx.AsyncClient | None = None

    async def resolve(self, url: str) -> list[str]:
        doi = doi_in(url)
        if doi is None or arxiv_id(url) is not None:
            return []
        try:
            async with self._slots:
                await self.start()
                response = await self._http.get(self.URL + doi)  # type: ignore[union-attr]
            if response.status_code != 200:
                return []
            return self._copies(response.json())
        except Exception:  # timeouts, connection errors, bad JSON
            return []

    @staticmethod
    def _copies(work: dict[str, Any]) -> list[str]:
        copies = [work.get("best_oa_location") or {}] + [loc for loc in work.get("locations") or [] if loc.get("is_oa")]
        urls = [c.get("pdf_url") for c in copies] + [c.get("landing_page_url") for c in copies]
        urls.append((work.get("open_access") or {}).get("oa_url"))
        return list(dict.fromkeys(u for u in urls if u))

    async def start(self) -> None:
        if self._http is None:
            self._http = httpx.AsyncClient(timeout=self.timeout, headers={"User-Agent": self.USER_AGENT})

    async def stop(self) -> None:
        if self._http is not None:
            await self._http.aclose()
            self._http = None
