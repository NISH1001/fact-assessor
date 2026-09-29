"""DuckDuckGoSearcher: DuckDuckGo via the `ddgs` package (no API key)."""

from __future__ import annotations

import asyncio
from typing import Any

from factassessor.search._base import Searcher


class DuckDuckGoSearcher(Searcher):
    """DuckDuckGo via the `ddgs` package: no API key, no setup (`fact-assessor[ddg]`).

    Slower and less predictable than Serper (0.7-3.3s per query in our check vs ~0.8s) and it's an unofficial
    client, so heavy use can get rate-limited. Good for trying things out; use Serper for production.
    """

    def __init__(self, num: int = 10, client_factory: Any = None, retry_after: float = 1.0) -> None:
        self.num = num
        self.retry_after = retry_after
        self._client_factory = client_factory  # tests pass a fake; default: ddgs.DDGS

    async def search(self, query: str) -> list[dict[str, Any]]:
        results = await asyncio.to_thread(self._search_sync, query)  # ddgs is synchronous
        if results is None:  # ddgs says "no results", which from DuckDuckGo usually means throttling: retry once
            await asyncio.sleep(self.retry_after)
            results = await asyncio.to_thread(self._search_sync, query) or []
        return [{"url": r["href"], "title": r.get("title") or "", "snippet": r.get("body") or ""} for r in results]

    def _search_sync(self, query: str) -> list[dict[str, Any]] | None:
        """Hits, or None when ddgs reports no results (it raises instead of returning [])."""
        factory = self._client_factory
        if factory is None:
            from ddgs import DDGS as factory
        try:
            return factory().text(query, max_results=self.num) or []
        except Exception as exc:
            if "no results" in str(exc).lower():
                return None
            raise
