"""SerperSearcher: Google results via Serper (web search, or Google Scholar with SearchType.SCIENCE)."""

from __future__ import annotations

import os
from typing import Any

import httpx

from factassessor.keys import require_key

from factassessor.search._base import BLOCKED_DOMAINS, Searcher, SearchType, hedged

SERPER_URL = "https://google.serper.dev/search"
SERPER_SCHOLAR_URL = "https://google.serper.dev/scholar"
GOOGLE_QUERY_WORDS = 32  # Google ignores everything past 32 words; a `-site:` operator counts as one


class SerperSearcher(Searcher):
    """Google results via Serper (needs SERPER_API_KEY).

    `exclude`: hosts appended to every web query as `-site:` operators (the blocked domains by default), so Google
    fills those result slots with usable sources instead of us paying for hits that `not_blocked()` then drops
    (measured: 3 of 10 hits blocked -> 0, same credit, the surviving hits in the same order). As many as fit next
    to the query under Google's 32-word cap, in the tuple's order, so the leakiest hosts go first; keep
    `not_blocked()` in the chain as the guarantee. Google Scholar queries are sent as they are.

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
        exclude: tuple[str, ...] = BLOCKED_DOMAINS,
    ) -> None:
        self.api_key = require_key("SERPER_API_KEY", api_key, needed_by="Serper search (SerperSearcher)",
                                   instead="To search without a key, run SearXNG yourself: FactAssessor(searcher=SearxngSearcher(url)).")
        self.search_type = SearchType(search_type)  # SCIENCE: Google Scholar, 1 credit per query like web search
        self.num = num
        self.timeout = timeout
        self.hedge_after = hedge_after
        self.exclude = exclude
        self._http: httpx.AsyncClient | None = None

    def query(self, query: str) -> str:
        """The query as sent: with `-site:` exclusions, as many as fit under Google's word cap (web search only)."""
        if self.search_type is SearchType.SCIENCE or not self.exclude:
            return query
        room = max(0, GOOGLE_QUERY_WORDS - len(query.split()))
        return " ".join([query, *(f"-site:{host}" for host in self.exclude[:room])])

    async def search(self, query: str) -> list[dict[str, Any]]:
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
        response = await self._http.post(url, headers={"X-API-KEY": self.api_key}, json={"q": self.query(query), "num": self.num})
        response.raise_for_status()
        return [
            # Scholar results may carry a free PDF (pdfUrl): point the hit there, so crawling reads the paper
            {"url": r.get("pdfUrl") or r["link"], "title": r.get("title") or "", "snippet": r.get("snippet") or ""}
            for r in response.json().get("organic", [])
        ]
