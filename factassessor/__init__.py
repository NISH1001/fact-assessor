from dotenv import find_dotenv, load_dotenv

# API keys (SERPER_API_KEY, OPENAI_API_KEY, ...) from .env in the cwd or a parent; never overrides real env vars.
load_dotenv(find_dotenv(usecwd=True))

from factassessor import kg  # noqa: E402
from factassessor.assessor import FactAssessor
from factassessor.atom_filter import LayaCheckworthy
from factassessor.atomizer import Atomizer, LLMAtomizer
from factassessor.crawl import Crawl4AICrawler, Crawler, FallbackCrawler, HTTPXCrawler
from factassessor.evidence_judge import Judge, LayaJudge
from factassessor.laya import LayaRunner
from factassessor.pipeline import Chain, Filter, FlatMap, Map, Pred, Scan, Step, Take, TakeUntil, collect, last, once
from factassessor.schema import Atom, AtomResult, CheckResult, ClaimFound, ClaimVerified, Done, Event, Evidence, fact_score
from factassessor.search import (
    BLOCKED_DOMAINS,
    DuckDuckGoSearcher,
    Searcher,
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
    "Searcher", "SerperSearcher", "DuckDuckGoSearcher", "SearxngSearcher",
    "Crawler", "Crawl4AICrawler", "HTTPXCrawler", "FallbackCrawler",
    "Judge", "LayaJudge",
    "Policy", "WeightedPolicy",
    "LayaCheckworthy", "Verify", "LayaRunner",
    # composition
    "Step", "Chain", "Map", "FlatMap", "Filter", "Take", "Scan", "TakeUntil", "Pred", "once", "collect", "last",
    # helpers
    "not_blocked", "is_blocked", "BLOCKED_DOMAINS", "fact_score", "kg",
    # data
    "Atom", "Evidence", "AtomResult", "CheckResult", "ClaimFound", "ClaimVerified", "Done", "Event",
]
