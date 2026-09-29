"""Searchers: query -> hits `{"url", "title", "snippet"}`, in rank order. A step.

`Searcher` is the role: implement `search(query)`. Compose: `SerperSearcher() >> not_blocked() >> Take(5)`.
"""

from factassessor.search._base import BLOCKED_DOMAINS, Searcher, SearchType, hedged, is_blocked, not_blocked
from factassessor.search.documents import DocumentSearcher
from factassessor.search.duckduckgo import DuckDuckGoSearcher
from factassessor.search.searxng import SearxngSearcher
from factassessor.search.serper import SERPER_SCHOLAR_URL, SERPER_URL, SerperSearcher

__all__ = [
    "Searcher", "SearchType", "SerperSearcher", "SearxngSearcher", "DuckDuckGoSearcher", "DocumentSearcher",
    "not_blocked", "is_blocked", "hedged", "BLOCKED_DOMAINS", "SERPER_URL", "SERPER_SCHOLAR_URL",
]
