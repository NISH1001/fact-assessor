"""AKDScholarSearcher: AKD Scholar Search, BM25 over 27.7M full-text papers (S2ORC), with the passages that match."""

from __future__ import annotations

import os
from typing import Any

import httpx

from factassessor.keys import require_key
from factassessor.search._base import Hit, Searcher, hedged
from factassessor.utils import http_pool


def akd_settings(url: str | None, api_key: str | None, needed_by: str) -> tuple[str, str]:
    """The server's base URL and key: the arguments, else AKD_SCHOLAR_SEARCH_URL / AKD_SCHOLAR_SEARCH_API_KEY."""
    base = (url or os.environ.get("AKD_SCHOLAR_SEARCH_URL", "")).strip().rstrip("/")
    if not base:
        raise ValueError(f"{needed_by} needs the server's URL: set AKD_SCHOLAR_SEARCH_URL or pass url=")
    key = require_key("AKD_SCHOLAR_SEARCH_API_KEY", api_key, needed_by=needed_by, instead="Or search the web: SerperSearcher().")
    return base, key


class AKDScholarSearcher(Searcher):
    """AKD Scholar Search (`GET /search`): the papers that best match the query (BM25 over title and body), each
    with the passages where the query's words occur. A hit is one paper `{"url", "title", "snippet"}`: the url
    is its doi.org link (else an open-access copy), the snippet its best `max_matches` passages joined, so the
    judge has paper text before any crawl; the crawler reads the paper from its link like any Scholar hit.

    The index returns a paper once per matching chunk (10 results held 7 papers in our probe), so `top_k` asks for
    twice `num` and repeats are merged into the first. Answers in ~0.1-0.2s, so no hedging by default.
    """

    def __init__(
        self,
        url: str | None = None,
        api_key: str | None = None,
        num: int = 10,
        context_words: int = 40,  # words on either side of each passage (server default 20)
        max_matches: int = 3,  # passages per paper in the snippet
        timeout: float = 10.0,
        hedge_after: float | None = None,
        max_connections: int = 32,  # keep-alive connections, reused (utils.http_pool)
    ) -> None:
        self.url, self.api_key = akd_settings(url, api_key, needed_by="AKD Scholar Search (AKDScholarSearcher)")
        self.num = num
        self.context_words = context_words
        self.max_matches = max_matches
        self.timeout = timeout
        self.hedge_after = hedge_after
        self.max_connections = max_connections
        self._http: httpx.AsyncClient | None = None

    async def search(self, query: str) -> list[Hit]:
        if self.hedge_after is None:
            return await self._request(query)
        return await hedged(lambda: self._request(query), self.hedge_after)

    async def start(self) -> None:
        if self._http is None:
            self._http = http_pool(self.timeout, self.max_connections)

    async def stop(self) -> None:
        if self._http is not None:
            await self._http.aclose()
            self._http = None

    async def _request(self, query: str) -> list[Hit]:
        await self.start()
        params = {"q": query, "top_k": min(100, 2 * self.num), "index": "all",  # title + body
                  "context_words": self.context_words, "max_matches": self.max_matches}
        response = await self._http.get(f"{self.url}/search", params=params, headers={"Authorization": f"Bearer {self.api_key}"})
        response.raise_for_status()  # a 401/403 from this server: keys.auth_error names AKD_SCHOLAR_SEARCH_API_KEY
        papers: dict[str, dict[str, Any]] = {}  # url -> hit, passages merged across repeats, in rank order
        for d in response.json().get("docs", []):
            url = d.get("link") or (f"https://doi.org/{d['doi']}" if d.get("doi") else "")
            if not url:
                continue  # nothing to cite
            hit = papers.setdefault(url, {"url": url, "title": d.get("title") or "", "passages": []})
            for m in d.get("matches") or [{"content": d.get("content") or ""}]:
                passage = m["content"].strip().strip("…").strip()
                if passage and passage not in hit["passages"]:
                    hit["passages"].append(passage)
        return [
            {"url": h["url"], "title": h["title"], "snippet": " … ".join(h["passages"][: self.max_matches])}
            for h in list(papers.values())[: self.num]
        ]
