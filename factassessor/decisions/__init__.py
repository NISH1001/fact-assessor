"""Decision runners: the model layer under the claim filter and the judge.

A decision model reads a `state` (named text fields) and answers typed `questions` about it with probabilities; it
generates no text. Several models answer that same request, one module each:
TypeSafe's Jev over HTTP (`systemone.SystemOneRunner`, the default; also a remote `python -m laya.serve`), OpenAI's
Decisions API (`openai.OpenAIDecisionRunner`), Laya in-process (`laya.LayaRunner`), any chat LLM asked to pick an
option (`llm.LLMRunner`), and GLiNER2.5-decide (`gliner.GlinerRunner`). `DecisionJudge` and
`DecisionClaimFilter` are written once on top of the `DecisionRunner` protocol, so the model is one argument, and
the filter and the judge can run on different ones.

A runner never sees claims, pages or evidence: only requests. Batching is its job, not the caller's: `predict`
takes a list, concurrent callers' requests are merged into shared model calls (`Batcher`), cut at `batch_size`
and capped in flight, so 100 incoming requests become a few GPU passes or HTTP calls, never 100 parallel anything.

`types` has the request, the answer and the role; `utils` what the runners share (`Batcher`, `pack`,
`post_packed`, `post_with_retries`). Every name imports from here too: `from factassessor.decisions import LayaRunner`.
"""

from factassessor.decisions.types import Answer, DecisionPacking, DecisionRequest, DecisionResponse, DecisionRunner, Question
from factassessor.decisions.utils import Batcher, pack, post_packed, post_with_retries
from factassessor.decisions.systemone import SystemOneRunner
from factassessor.decisions.openai import OpenAIDecisionRunner
from factassessor.decisions.llm import LLMRunner
from factassessor.decisions.laya import LayaRunner
from factassessor.decisions.gliner import GlinerRunner

__all__ = [
    "Question", "DecisionRequest", "Answer", "DecisionResponse", "DecisionRunner", "DecisionPacking",
    "Batcher", "pack", "post_packed", "post_with_retries",
    "SystemOneRunner", "OpenAIDecisionRunner", "LLMRunner", "LayaRunner", "GlinerRunner",
]
