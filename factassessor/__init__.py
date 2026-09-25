from dotenv import find_dotenv, load_dotenv

# API keys (SERPER_API_KEY, OPENAI_API_KEY, ...) from .env in the cwd or a parent; never overrides real env vars.
load_dotenv(find_dotenv(usecwd=True))

from factassessor.aggregate import build_graph, fact_score  # noqa: E402
from factassessor.assessor import FactAssessor
from factassessor.atom_filter import LayaCheckworthy
from factassessor.atomizer import Atomizer
from factassessor.crawl import Crawl4ai
from factassessor.evidence_judge import LayaJudge
from factassessor.laya import LayaRunner
from factassessor.pipeline import Chain, Filter, FlatMap, Map, Pred, Scan, Step, Take, TakeUntil, collect, last, once
from factassessor.schema import Atom, AtomResult, CheckResult, ClaimFound, ClaimVerified, Done, Event, Evidence
from factassessor.search import BLOCKED_DOMAINS, Serper, is_blocked, not_blocked
from factassessor.verify import Verify, WeightedPolicy

__all__ = [
    # the ready-made pipeline
    "FactAssessor",
    # components
    "Atomizer", "LayaCheckworthy", "Serper", "Crawl4ai", "LayaJudge", "Verify", "WeightedPolicy", "LayaRunner",
    # composition
    "Step", "Chain", "Map", "FlatMap", "Filter", "Take", "Scan", "TakeUntil", "Pred", "once", "collect", "last",
    # helpers
    "not_blocked", "is_blocked", "BLOCKED_DOMAINS", "fact_score", "build_graph",
    # data
    "Atom", "Evidence", "AtomResult", "CheckResult", "ClaimFound", "ClaimVerified", "Done", "Event",
]
