from importlib.metadata import PackageNotFoundError, version

from dotenv import find_dotenv, load_dotenv

try:
    __version__ = version("fact-assessor")  # from pyproject.toml, so the two never drift
except PackageNotFoundError:  # running from a source tree that isn't installed
    __version__ = "0.0.0"

# API keys (SERPER_API_KEY, OPENAI_API_KEY, OPENROUTER_API_KEY, ...) from .env in the cwd or a parent; never
# overrides real env vars.
load_dotenv(find_dotenv(usecwd=True))

from factassessor import kg  # noqa: E402
from factassessor.assessor import FactAssessor
from factassessor.atomizer import Atomizer, LLMAtomizer
from factassessor.claim_filters import ClaimFilter, DecisionClaimFilter
from factassessor.keys import MissingAPIKeyError
from factassessor.crawlers import Crawl4AICrawler, Crawler, CascadedCrawler, Fetch, HTTPXCrawler, ImpitCrawler, NoCrawler
from factassessor.decisions import (Answer, DecisionPacking, DecisionRequest, DecisionResponse, DecisionRunner, GlinerRunner, LayaRunner,
                                    LLMRunner, OpenAIDecisionRunner, Question, SystemOneRunner)
from factassessor.judges import DecisionJudge, Judge
from factassessor.pipeline import Cache, Chain, Filter, FlatMap, Map, Predicate, Scan, Step, Take, TakeUntil, collect, last, once
from factassessor.rankers import BM25Ranker, EmbeddingRanker, HybridRanker, Ranker
from factassessor.resolvers import ArxivResolver, CompositeResolver, OpenAlexResolver, PMCResolver, Resolver
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
    "ClaimFilter", "DecisionClaimFilter",
    "Searcher", "SerperSearcher", "DuckDuckGoSearcher", "SearxngSearcher", "DocumentSearcher", "SearchType",
    "Crawler", "Crawl4AICrawler", "HTTPXCrawler", "ImpitCrawler", "CascadedCrawler", "NoCrawler", "Fetch", "MissingAPIKeyError",
    "Resolver", "ArxivResolver", "OpenAlexResolver", "PMCResolver", "CompositeResolver",
    "Ranker", "BM25Ranker", "EmbeddingRanker", "HybridRanker",
    "Judge", "DecisionJudge",
    "Policy", "WeightedPolicy",
    "Verify",
    # decision runners (the model behind the claim filter and the judge) and their request/response types
    "DecisionRunner", "LayaRunner", "SystemOneRunner", "OpenAIDecisionRunner", "DecisionPacking", "LLMRunner", "GlinerRunner",
    "DecisionRequest", "DecisionResponse", "Question", "Answer",
    # composition
    "Step", "Chain", "Map", "FlatMap", "Filter", "Take", "Scan", "TakeUntil", "Cache", "Predicate", "once", "collect", "last",
    # helpers
    "not_blocked", "is_blocked", "BLOCKED_DOMAINS", "fact_score", "kg",
    # data
    "Atom", "Evidence", "AtomResult", "CheckResult", "ClaimFound", "ClaimVerified", "Done", "Event",
]
