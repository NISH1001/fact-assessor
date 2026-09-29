from dotenv import find_dotenv, load_dotenv

# API keys (SERPER_API_KEY, OPENAI_API_KEY, ...) from .env in the cwd or a parent; never overrides real env vars.
load_dotenv(find_dotenv(usecwd=True))

from factassessor import kg  # noqa: E402
from factassessor.assessor import FactAssessor
from factassessor.atomizer import Atomizer, LLMAtomizer
from factassessor.claim_filters import ClaimFilter, GlinerClaimFilter, LayaClaimFilter
from factassessor.crawlers import Crawl4AICrawler, Crawler, FallbackCrawler, HTTPXCrawler, NoCrawler
from factassessor.judges import GlinerJudge, Judge, LayaJudge, LLMJudge
from factassessor.pipeline import Chain, Filter, FlatMap, Map, Pred, Scan, Step, Take, TakeUntil, collect, last, once
from factassessor.resolvers import ArxivResolver, CompositeResolver, OpenAlexResolver, Resolver
from factassessor.schema import (
    Atom,
    AtomResult,
    CheckResult,
    ClaimFound,
    ClaimVerified,
    Done,
    Event,
    Evidence,
    fact_score,
)
from factassessor.search import (
    BLOCKED_DOMAINS,
    DocumentSearcher,
    DuckDuckGoSearcher,
    Searcher,
    SearchType,
    SearxngSearcher,
    SerperSearcher,
    is_blocked,
    not_blocked,
)
from factassessor.verify import Policy, Verify, WeightedPolicy

__all__ = [
    # the ready-made pipeline
    "FactAssessor",
    # roles (base types) and their implementations
    "Atomizer", "LLMAtomizer",
    "ClaimFilter", "LayaClaimFilter", "GlinerClaimFilter",
    "Searcher", "SerperSearcher", "DuckDuckGoSearcher", "SearxngSearcher", "DocumentSearcher", "SearchType",
    "Crawler", "Crawl4AICrawler", "HTTPXCrawler", "FallbackCrawler", "NoCrawler",
    "Resolver", "ArxivResolver", "OpenAlexResolver", "CompositeResolver",
    "Judge", "LayaJudge", "GlinerJudge", "LLMJudge",
    "Policy", "WeightedPolicy",
    "Verify",
    # composition
    "Step", "Chain", "Map", "FlatMap", "Filter", "Take", "Scan", "TakeUntil", "Pred", "once", "collect", "last",
    # helpers
    "not_blocked", "is_blocked", "BLOCKED_DOMAINS", "fact_score", "kg",
    # data
    "Atom", "Evidence", "AtomResult", "CheckResult", "ClaimFound", "ClaimVerified", "Done", "Event",
]
