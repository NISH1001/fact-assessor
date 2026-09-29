"""SerperSearcher: Google results via Serper (web search, or Google Scholar with SearchType.SCIENCE)."""

from __future__ import annotations

import os
from typing import Any

import httpx

from factassessor.search._base import Searcher, SearchType, hedged

SERPER_URL = "https://google.serper.dev/search"
SERPER_SCHOLAR_URL = "https://google.serper.dev/scholar"


class SerperSearcher(Searcher):
    """Google results via Serper (needs SERPER_API_KEY).

    `hedge_after`: Serper's p50 is ~0.8s but outliers hit 3s+, so a request that hasn't answered by then is
    raced against a duplicate and the first reply wins.
    """

    def __init__(
        self,
        api_key: str | None = None,
        num: int = 10,
        timeout: float = 5.0,
        hedge_after: float | None = 1.2,
        search_type: SearchType | str = SearchType.GENERAL,
    ) -> None:
        self.api_key = api_key or os.environ.get("SERPER_API_KEY")
        self.search_type = SearchType(search_type)  # SCIENCE: Google Scholar, 1 credit per query like web search
        self.num = num
        self.timeout = timeout
        self.hedge_after = hedge_after
        self._http: httpx.AsyncClient | None = None

    async def search(self, query: str) -> list[dict[str, Any]]:
        if not self.api_key:
            raise RuntimeError("SERPER_API_KEY is not set (pass api_key= or add it to .env)")
        if self.hedge_after is None:
            return await self._request(query)
        return await hedged(lambda: self._request(query), self.hedge_after)

    async def start(self) -> None:
        if self._http is None:
            self._http = httpx.AsyncClient(timeout=self.timeout)

    async def stop(self) -> None:
        if self._http is not None:
            await self._http.aclose()
            self._http = None

    async def _request(self, query: str) -> list[dict[str, Any]]:
        await self.start()
        url = SERPER_SCHOLAR_URL if self.search_type is SearchType.SCIENCE else SERPER_URL
        response = await self._http.post(url, headers={"X-API-KEY": self.api_key}, json={"q": query, "num": self.num})
        response.raise_for_status()
        return [
            # Scholar results may carry a free PDF (pdfUrl): point the hit there, so crawling reads the paper
            {"url": r.get("pdfUrl") or r["link"], "title": r.get("title") or "", "snippet": r.get("snippet") or ""}
            for r in response.json().get("organic", [])
        ]
