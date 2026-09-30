"""SearxngSearcher: your own SearXNG instance (no API key)."""

from __future__ import annotations

import asyncio
from typing import Any

import httpx

from factassessor.search._base import Searcher, SearchType, hedged


class SearxngSearcher(Searcher):
    """Your own SearXNG instance (no API key): `SearxngSearcher("http://localhost:8080")`.

    Public instances don't work for this: in our check, 0 of 25 healthy ones served JSON (rate limits, bot
    blocking, JSON disabled). Run one yourself with JSON enabled (`search.formats: [html, json]` in settings.yml),
    e.g. `docker run -p 8080:8080 searxng/searxng`.

    `categories` / `engines`: which of SearXNG's sources to search, e.g. `categories=["science"]` for its scholarly
    engines (Google Scholar, arXiv, Semantic Scholar, PubMed...). The default ("general") never uses them; on
    scientific claims, science nearly doubled finding the source paper (20% -> 38% of claims on a scientific eval set).
    """

    def __init__(
        self,
        base_url: str,
        num: int = 10,
        timeout: float = 5.0,
        hedge_after: float | None = 1.2,
        categories: list[str] | None = None,
        engines: list[str] | None = None,
        search_type: SearchType | str = SearchType.GENERAL,
        max_concurrent: int | None = 4,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.num = num
        self.search_type = SearchType(search_type)
        if categories is None and self.search_type is SearchType.SCIENCE:
            categories = ["science"]  # explicit categories win
        self.categories, self.engines = categories, engines
        self.timeout = timeout
        self.hedge_after = hedge_after
        # SearXNG's engines suspend a client that bursts ("Suspended: too many requests", CAPTCHAs): a text's 15 claims
        # searching at once got mostly empty results. 4 in flight kept every engine answering over 1,700 searches.
        self.max_concurrent = max_concurrent
        self._slots = asyncio.Semaphore(max_concurrent) if max_concurrent else None
        self._http: httpx.AsyncClient | None = None

    async def search(self, query: str) -> list[dict[str, Any]]:
        if self._slots is None:
            return await self._search(query)
        async with self._slots:
            return await self._search(query)

    async def _search(self, query: str) -> list[dict[str, Any]]:
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
        params = {"q": query, "format": "json"}
        if self.categories:
            params["categories"] = ",".join(self.categories)
        if self.engines:
            params["engines"] = ",".join(self.engines)
        response = await self._http.get(f"{self.base_url}/search", params=params)
        response.raise_for_status()
        return [
            {"url": r["url"], "title": r.get("title") or "", "snippet": r.get("content") or ""}
            for r in response.json().get("results", [])[: self.num]
        ]
