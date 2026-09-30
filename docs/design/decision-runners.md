# Decision runners: the model layer (design, 2026-09-30)

Status: built 2026-09-30 (steps 1-6; `ARCHITECTURE.md` describes the result). Two departures from the plan
below: `LLMRunner` and `GlinerRunner` landed in one commit, and step 6 made `Judge` and `Policy` Protocols while
`Atomizer`, `Searcher`, `Crawler` and `ClaimFilter` stayed `Step` base classes (without ABCs): what those four
give an implementer is the streaming adapter around the one method, and a Protocol can't carry that without
requiring `__call__` of every conforming object. `SystemOneRunner` packs one `predict` call (one claim's
passages) per request by default, as a list field with `evidence[i]` references (Jev's documented "batch every
question about the same state"); `Packing.ALL` mixes every caller in flight, which measured well on a 4-answer
sample but isn't documented behaviour. Step 7 (the Jev replay, then the overfetch live run) is next.

## Why

Three components ask a decision model a typed question and read probabilities back: the claim filter ("what kind
of statement is this?"), the judge ("does this passage support the claim?"), and, in effect, the GLiNER and LLM
variants of both. Today each variant is its own class (`LayaJudge`, `GlinerJudge`, `LLMJudge`, `LayaClaimFilter`,
`GlinerClaimFilter`), and each re-implements passage selection and request building for its model. Swapping the
model means swapping the component; running the filter on one model and the judge on another means two unrelated
classes. Meanwhile TypeSafe's Jev (a hosted decision model of the same kind as Laya) answers the identical request
shape over HTTP, and laya ships a server speaking the same wire protocol.

The fix is one more layer, below the roles: a **runner** answers `(state, questions)` requests with probabilities,
and the filter and the judge are written once, on top of it.

## The system

```
LAYER 1  PIPELINE   (assessor.py, verify.py)   the only code that knows the order of things
   atoms  = atomizer >> claim_filter >> Take(n_atoms)
   verify = Verify(searcher, resolver, crawler, judge, policy)     per claim, all claims at once
   result = atoms >> verify  ->  CheckResult (verdicts, fact_score, knowledge graph on demand)

LAYER 2  ROLES      (Protocols, @runtime_checkable, one method each; Step is an optional mixin for >>)
   Atomizer.atomize(text)        -> atoms          LLMAtomizer
   ClaimFilter.score(atom)       -> P(factual)     DecisionClaimFilter(runner)
   Searcher.search(query)        -> hits           Serper, SearXNG, DuckDuckGo, Document
   Resolver.resolve(url)         -> urls           Arxiv, OpenAlex, Composite
   Crawler.crawl(url)            -> page           HTTPX, Crawl4AI, Fallback, NoCrawler
   Judge.judge(claim, docs)      -> evidence       DecisionJudge(runner, ranker, chunk, passages_per_page)
   Ranker.top(claim, chunks, k)  -> chunks         BM25, Embedding, Hybrid
   Policy.settled / verdict      -> verdict        WeightedPolicy

LAYER 3  RUNNERS    (decisions.py)
   DecisionRunner (Protocol): batch_size: int; async predict(requests) -> responses
   LayaRunner        in-process Laya on the local GPU; merges requests from every caller into 32-row passes (default)
   SystemOneRunner   HTTP, Jev's System One protocol: OpenRouter's Jev, or a remote `python -m laya.serve`
   LLMRunner         any chat model via pydantic-ai (OpenAI, OpenRouter, Ollama, an OpenAI-compatible endpoint)
   GlinerRunner      GLiNER2.5-decide via ONNX on the CPU

HELPERS  extract.py (bytes -> text)   passages.py (clean, normalize, word_windows, chunk, BM25)
```

### The runner contract

```python
class Question(BaseModel):
    type: Literal["choice", "noul", "score"]   # the wire protocol's word; not a reserved word in Python
    instructions: str
    criteria: dict[str, str] = {}              # choice: label -> description

class DecisionRequest(BaseModel):
    state: str | dict[str, Any]                # e.g. {"evidence": ..., "claim": ...}
    questions: dict[str, Question]
    model: str | None = None

class Answer(BaseModel):
    choice: str | None = None
    probabilities: dict[str, float]            # every label, sums to 1
    confidence: float | None = None

class DecisionResponse(BaseModel):
    answers: dict[str, Answer]                 # the request's keys
    usage: dict[str, float] = {}               # tokens, cost, when the server reports them

@runtime_checkable
class DecisionRunner(Protocol):
    batch_size: int                            # items per model call: rows per GPU pass, questions per HTTP call...
    async def predict(self, requests: list[DecisionRequest]) -> list[DecisionResponse]: ...
```

- **One method.** `predict` takes a list; batching is the runner's job, never the caller's. A runner merges requests
  from all concurrent callers, cuts them into `batch_size` pieces and caps its own concurrency, so 100 incoming
  requests become 4 GPU passes (Laya) or 3 HTTP calls (Jev), never 100 parallel anything. `batch_size` is an
  attribute set in each runner's own `__init__`, not a `predict` argument: a per-call size would defeat the merging.
- **No tokenizer on the protocol.** Passages are cut by words in the judge (90 words, about Laya's 128 tokens, well
  inside the ~300-token state budget); a token-exact chunker stays available as `DecisionJudge(chunk=...)` for those
  who want it with Laya. The tokenizer is Laya's private business.
- Our runners subclass the Protocol explicitly (`class LayaRunner(DecisionRunner)`) so the type checker verifies
  them; external runners conform structurally, and `isinstance(x, DecisionRunner)` works for both. Plain classes
  with a short `__init__`, no dataclasses, no abstract base classes.
- **`SystemOneRunner`** packs a judge call's requests (same claim, one passage varying) into one HTTP request with
  one question per passage; the server evaluates them in parallel. URL from `OPENROUTER_DECISIONS_URL` or
  `https://openrouter.ai/api/v1/systemone`; key from `OPENROUTER_API_KEY`; a Laya server needs neither. Verified
  on 2026-09-30: OpenRouter's `/api/v1/systemone` and `/api/alpha/decisions` and a local `laya.serve` all accept
  the Laya-shaped request and return the same answer shape; jev-1.13 costs ~$0.0000176 per call (418 input tokens).

### Rules that keep it small

1. The pipeline depends only on roles; a role never imports a backend by name except as its default.
2. A runner never sees pipeline concepts (claims, pages, evidence): only `state` and `questions`.
3. A new model for the same question is a runner (~40 lines). A new *way* of judging (an NLI cross-encoder with
   its own preprocessing, a rule checker, an agent) is a new `Judge`, and may or may not use runners.
4. Every default is a number from the eval, written next to it (`strong=0.7`, `passages_per_page=3`, SearXNG's
   4 requests in flight, `batch_size=32`).
5. Delete, don't alias: `LLMJudge`, `GlinerJudge`, `GlinerClaimFilter` and the interim `DecisionAPIJudge` go away
   as their runners land. No `utils.py`.
6. Not built until measured on the target hardware: a Laya ONNX runner (laya's ONNX path is CPU/CUDA only, no MPS),
   a `Chunker` role (a function argument is enough), question types beyond `choice`.

### Mixing models

```python
laya = LayaRunner()                      # local, fast: "is this a factual claim?"
gpt  = LLMRunner("openai:gpt-6-luna")    # the judge, where the hard decisions are
FactAssessor(claim_filter=DecisionClaimFilter(laya), judge=DecisionJudge(gpt, passages_per_page=3))
```

`FactAssessor()` with no arguments shares one `LayaRunner` between filter and judge (the model loads once).
An LLM's probabilities are self-reported, not calibrated like Laya's or Jev's; its `strong` threshold has to be
set from a replay before it is recommended.

## How the eval sits on it

`scripts/eval_atoms.py` is the same pipeline with three substitutions and a recorder: FactReasoner's labelled
atoms from the JSONL as the claims (our atomizer runs only for timing and the text's source query, so scores are
apples-to-apples); the real searcher and crawler on a live run, wrapped to cache hits and pages, and cached
replacements on a replay; `DecisionJudge(runner=...)` chosen by `--judge laya | decision`, with `--passages`,
`--ranker`, `--strong` as the knobs. Each kind of question maps to one layer:

- retrieval (overfetch, Serper, resolvers, science search): a live run, pages cached;
- judging (passages, ranker, threshold, which runner): a replay on identical evidence, no network;
- throughput: the runner. Laya at ~47 rows/s on an M-series Mac is the floor for local replays (~20 min for
  100 answers at top-3); `SystemOneRunner` on Jev runs the same replay in minutes for ~$0.50. The final
  apples-to-apples numbers are re-run on Laya with one flag.

## Measured facts this design rests on (2026-09-29/30)

- Laya's per-row cost on MPS is the replay floor: the batch window (5 -> 50ms) and replaying 4 answers at once
  changed nothing; the runner fix (drain everything pending after each pass) gave 13%.
- The local Laya server (`laya.serve`) serializes requests behind a lock, one forward pass per HTTP request
  (180ms for one row): in-process `LayaRunner` batches across claims and is the faster choice on one machine;
  the server is for a separate GPU box or non-Python clients.
- Laya checkpoints: english (421M, ModernBERT-large, `max_len` 512, encoder positions 8,192), multilingual
  (322M, 1,024 tokens, up to 8,192, ~2.2x faster), typed-decisions (1,024). 512/128-token passages stay the
  default: on the frozen benchmark one 128-token passage matched three ~290-token ones at a third of the time;
  top-3 *separate* passages gave +0.10 F1. Longer windows are one replay away, not assumed.
- Jev on OpenRouter: jev-1.13, 32k-token window (jev-router 1M), $0.042 per million input tokens, output free;
  one call ~0.5s; many questions per request.

## Codebase changes, file by file

No aliases and no compatibility shims: the project is alpha, and every rename is done everywhere in one commit
(`perl -pi -e 's/\bOld\b/New/g'` over `factassessor tests docs README.md notebooks/*.py`, the way `Pred` became
`Predicate`).

| Step | New | Changed | Deleted |
|---|---|---|---|
| 1 runners | `factassessor/decisions.py` (schema, `DecisionRunner`, `SystemOneRunner`), `tests/test_decisions.py` (packing, splitting at `batch_size`, 429 backoff, cost, protocol conformance) | `factassessor/__init__.py` exports | `factassessor/judges/decision_api.py`, `tests/test_decision_api_judge.py` |
| 2 Laya conforms | | `factassessor/laya.py`: `class LayaRunner(DecisionRunner)`, `predict()` (was `predict_batch`), typed requests in and responses out, `batch_size` attribute; `tests/test_laya_runner.py` | |
| 3 judge + filter | `factassessor/judges/decision.py` (`DecisionJudge`), `factassessor/claim_filters/decision.py` (`DecisionClaimFilter`) | `judges/__init__.py`, `claim_filters/__init__.py`, `assessor.py` (defaults: one shared `LayaRunner`), `verify.py` (nothing but names), `scripts/eval_atoms.py` (`make_judge` builds `DecisionJudge(runner=...)`), README, ARCHITECTURE, WALKTHROUGH, decisions.md, notebooks; tests renamed (`test_laya_judge.py` -> `test_decision_judge.py`, `test_claim_filter.py`) | `judges/laya.py`, `claim_filters/laya.py` (their bodies move) |
| 4 LLM runner | `LLMRunner` in `decisions.py` (+ tests: prompt rendering, JSON parsing, batching per prompt) | `scripts/eval.py` (`llm` variant -> `DecisionJudge(LLMRunner(...))`), README judge table | `judges/llm.py`, `tests/test_llm_judge.py` |
| 5 GLiNER runner | `GlinerRunner` in `gliner.py` (criteria -> labels) + tests | `scripts/eval.py` (`gliner` variant), `scripts/compare_judges.py`, `compare_claim_filters.py`, README | `judges/gliner.py`, `claim_filters/gliner.py`, their tests (cases move to the runner tests) |
| 6 roles as Protocols | `tests/test_roles.py` | `judges/_base.py` (`Judge`) and `verify.py` (`Policy`): `@runtime_checkable` Protocols with their one/two methods; `atomizer.py`, `claim_filters/_base.py`, `search/_base.py`, `crawlers/_base.py`: ABC and `abstractmethod` dropped, the role method raises `NotImplementedError` naming itself; ARCHITECTURE roles table | |
| 7 eval | | `scripts/eval_atoms.py --judge laya \| decision \| llm`, `--judge-model`; replay of `web-42` on Jev | |

What does not change: the pipeline (`Verify`'s chain, `FactAssessor.stream`), the data models in `schema.py`,
`Ranker`, `Resolver`, the crawlers, the searchers, `passages.py` (it gains nothing: `word_windows` is already
there), the eval's metric and caches. `FactAssessor()` with no arguments behaves exactly as before.

Public API after the refactor (what a user imports): `FactAssessor`; the roles; `DecisionJudge`,
`DecisionClaimFilter`; the runners `LayaRunner`, `SystemOneRunner`, `LLMRunner`, `GlinerRunner`; everything else
unchanged. `LayaJudge`, `LayaClaimFilter`, `LLMJudge`, `GlinerJudge`, `GlinerClaimFilter` stop existing.

## Build order (each step test-first, its own commit)

1. `decisions.py`: typed schema, `DecisionRunner`, `SystemOneRunner` (packing, splitting at `batch_size`, 429
   backoff, cost).
2. `LayaRunner.predict` and `batch_size` conformance; the laya-shaped dicts become the typed models.
3. `DecisionJudge` (was `LayaJudge`) and `DecisionClaimFilter` (was `LayaClaimFilter`) with word chunking by
   default; renames across README, docs, notebooks; `DecisionAPIJudge` removed.
4. `LLMRunner`, absorbing `LLMJudge`.
5. `GlinerRunner`, absorbing `GlinerJudge` and `GlinerClaimFilter`.
6. Roles as Protocols (`Step` stays a mixin for chaining); `ARCHITECTURE.md` updated.
7. The Jev replay of the 100-answer evidence; then the eval work continues (overfetch live run, in-domain).
