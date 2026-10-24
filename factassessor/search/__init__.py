"""Searchers: query -> hits `{"url", "title", "snippet"}`, in rank order. A step.

`Searcher` is the role: implement `search(query)`. Compose: `SerperSearcher() >> not_blocked() >> Take(5)`.
"""

from factassessor.search._base import BLOCKED_DOMAINS, Hit, HitOrder, Ranking, Searcher, SearchType, hedged, is_blocked, merge, not_blocked
from factassessor.search.akd import AKDScholarSearcher
from factassessor.search.documents import DocumentSearcher
from factassessor.search.duckduckgo import DuckDuckGoSearcher
from factassessor.search.searxng import SearxngSearcher
from factassessor.search.serper import SERPER_SCHOLAR_URL, SERPER_URL, SerperSearcher

__all__ = [
    "Searcher", "SearchType", "SerperSearcher", "SearxngSearcher", "DuckDuckGoSearcher", "DocumentSearcher", "AKDScholarSearcher",
    "not_blocked", "is_blocked", "hedged", "Hit", "Ranking", "HitOrder", "merge", "BLOCKED_DOMAINS", "SERPER_URL", "SERPER_SCHOLAR_URL",
]
