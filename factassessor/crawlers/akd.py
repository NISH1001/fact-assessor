"""AKDDoiCrawler: a paper's full text from AKD Scholar Search by its DOI (`GET /doi/{doi}`), no web fetch."""

from __future__ import annotations

import httpx

from factassessor.crawlers._base import Crawler
from factassessor.crawlers.predicates import Fetch
from factassessor.pipeline import Predicate
from factassessor.resolvers import doi_in
from factassessor.search.akd import akd_settings
from factassessor.utils import http_pool


class AKDDoiCrawler(Crawler):
    """url -> the paper's text from AKD Scholar Search's full-text index (S2ORC), for any url with a DOI in it: the
    doi.org links `AKDScholarSearcher` returns, publisher pages. ~0.1s, no paywalls or bot checks (two thirds of
    AKD's doi.org hits failed to crawl from the publisher). A url without a DOI, or a DOI the index doesn't hold
    (404), is a `Fetch` without a page, so put it first in a cascade: `CascadedCrawler(AKDDoiCrawler(), HTTPXCrawler())`.
    The one exception to never raising: a rejected key (401/403) raises the HTTP error, which `keys.auth_error`
    turns into InvalidAPIKeyError (AKD's host comes from AKD_SCHOLAR_SEARCH_URL), stopping the check.
    """

    def __init__(self, url: str | None = None, api_key: str | None = None, timeout: float = 10.0,
                 max_connections: int = 32, accept: Predicate | None = None) -> None:  # connections: utils.http_pool
        super().__init__(accept)
        self.url, self.api_key = akd_settings(url, api_key, needed_by="AKD Scholar Search (AKDDoiCrawler)")
        self.timeout = timeout
        self.max_connections = max_connections
        self._http: httpx.AsyncClient | None = None

    async def start(self) -> None:
        if self._http is None:
            self._http = http_pool(self.timeout, self.max_connections)

    async def stop(self) -> None:
        if self._http is not None:
            await self._http.aclose()
            self._http = None

    async def crawl(self, url: str) -> Fetch:
        doi = doi_in(url)
        if doi is None:
            return self.mark(Fetch(url=url, error="no DOI"))
        await self.start()
        try:
            response = await self._http.get(f"{self.url}/doi/{doi}", headers={"Authorization": f"Bearer {self.api_key}"})
        except httpx.HTTPError as exc:
            return self.mark(Fetch(url=url, error=type(exc).__name__))
        if response.status_code in (401, 403):  # a rejected key: raised, so keys.auth_error stops the check
            response.raise_for_status()           # instead of every hit quietly falling back to the web
        f = Fetch(url=url, status=response.status_code, content_type="application/json")
        if response.status_code == 200:
            paper = response.json()
            text = (paper.get("text") or "").strip()
            if text:
                f = f.model_copy(update={"page": {"url": url, "title": paper.get("title") or "", "text": text},
                                         "words": len(text.split())})
        return self.mark(f)
