# FactAssessor as a streaming, composable pipeline

Status: draft for review · 2026-09-24

## Why

Today every step except the atomizer, filter, and judge is a private method on `FactAssessor`, and the
orchestration (per-atom concurrency, early exit, hedging) is hand-written around them. We want:

1. **Every step swappable** behind a small interface, the way `Atomizer`, `AtomFilter`, and `LayaJudge` already are.
2. **Functional composition**: `Atomizer() >> Filter(...) >> Search(...) >> Verify(...) >> Aggregate()`.
3. **Streaming**: an atom moves to the next step the moment it exists. Atom 1 can be judged while atom 4 is
   still being searched. The only point where everything waits is the final aggregation.
4. **Component-level concerns stay at the component**: hedging, timeouts, retries, and later caching wrap a
   searcher or crawler; the pipeline doesn't know about them.
5. **No regressions**: keep every latency win (hedged search, early exit while crawling, pages judged on
   arrival, Laya micro-batching), and `FactAssessor().assess(text)` / `assess_sync(text)` keep working unchanged.

Non-goals for now: caching, distributed execution, batching many texts in one call.

## Core idea: steps over async streams

A step transforms a stream of items into a stream of items. Authors don't write stream plumbing; they write
one of three shapes and the runner lifts it:

| Shape | Author writes | Used by |
|---|---|---|
| `Map` | `async def process(item) -> Out \| None` (return `None` to drop) | Filter, Search, Verify |
| `FlatMap` | `async def process(item) -> AsyncIterator[Out]` (0..n outputs) | Atomizer (text → atoms) |
| `Fold` | `async def fold(items: AsyncIterator[In]) -> Result` | Aggregate |

```python
class Step(Protocol[In, Out]):
    def __call__(self, items: AsyncIterator[In]) -> AsyncIterator[Out]: ...
```

Runner semantics for `Map` / `FlatMap`:

- Items are processed **concurrently** (bounded by `concurrency`, default unbounded for I/O steps) and
  **emitted in completion order**. Order is restored at the end by atom id.
- `a >> b` returns a `Pipeline`, which is itself a step, so pipelines nest.
- **Cancellation flows upstream**: closing a stream (early exit, timeout, the UI moving on) cancels in-flight
  work in every step feeding it.

Why this doesn't reintroduce stage barriers: `>>` connects *streams*, not *batches*. Nothing waits for a step to
finish all items before the next step starts; only a `Fold` consumes the whole stream.

## What flows through

One record per atom, enriched as it moves:

```python
class Claim(BaseModel):
    atom: Atom                      # id, text (self-contained), span in the input
    hits: list[Hit] = []            # set by Search
    evidence: list[Evidence] = []   # set by Verify
    verdict: Verdict | None = None  # set by Verify
    confidence: float = 0.0
    error: str | None = None        # any step can mark a claim failed instead of raising
```

`AtomResult` / `CheckResult` stay as the public outputs (built from `Claim`s by `Aggregate`).

## Components and their interfaces

| Step | Wraps component | Component interface | Default |
|---|---|---|---|
| `Atomizer` (FlatMap: text → claims) | — | itself | LLM, pydantic-ai; **streams atoms as the model writes them** |
| `Filter(pred)` (Map, may drop) | predicate | `async (Claim) -> bool` | `LayaCheckworthy(laya, threshold=0.4)` |
| `Take(n)` | — | — | caps atoms per text (replaces `n_atoms`) |
| `Search(searcher)` (Map) | `Searcher` | `async search(query) -> list[Hit]` | `SerperSearcher(blocked_domains=...)` |
| `Verify(judge, crawler, policy)` (Map) | `Judge`, `Crawler`, `VerdictPolicy` | see below | `LayaJudge`, `Crawl4aiCrawler`, `WeightedPolicy` |
| `Aggregate(graph=...)` (Fold) | graph builder | `(claims) -> dict` | source→atom support/refute graph + fact score |

```python
class Searcher(Protocol):
    async def search(self, query: str) -> list[Hit]: ...

class Crawler(Protocol):
    async def crawl(self, url: str) -> Page | None: ...           # None = failed/timed out, never raises

class Judge(Protocol):
    async def judge(self, claim: str, docs: list[Hit | Page]) -> list[Evidence]: ...

class VerdictPolicy(Protocol):
    def settled(self, evidence: list[Evidence]) -> bool: ...        # early exit test
    def verdict(self, evidence: list[Evidence]) -> tuple[Verdict, float]: ...
```

Filters compose: `Filter(LayaCheckworthy() & NotQuestion())`, `|`, `~` on predicates.

### Verify: the only step with internal branching

```
judge(snippets) ── settled? ── yes ──────────────────────────────▶ verdict
                        └─ no ─▶ crawl all hits concurrently
                                 as each page lands: judge it, add evidence
                                 settled? ─ yes ─▶ cancel remaining crawls ─▶ verdict
                                 all pages done ─────────────────────────────▶ verdict
```

This is today's `_verify`, moved into a step with its collaborators injected. A per-claim deadline
(`Verify(..., timeout=...)`) guarantees every claim resolves, so `Aggregate` never waits on a stuck atom.

### Wrappers (component level)

```python
Hedged(SerperSearcher(), after=1.2)        # race a duplicate if slow; first reply wins
Timeout(Crawl4aiCrawler(), seconds=2.5)
Retry(SerperSearcher(), attempts=2)
# later: Cached(Crawl4aiCrawler())
```

Each wrapper implements the same protocol as what it wraps, so pipelines never change to add them.

### Laya

`LayaRunner` stays the one shared model with the micro-batcher. `LayaCheckworthy` and `LayaJudge` both take it.
Streaming puts more atoms in flight at once, which gives the micro-batcher more to merge.

## Public API

```python
from factassessor import FactAssessor

fa = FactAssessor()                       # the default pipeline, same knobs as today
result = await fa.assess(text)            # CheckResult (unchanged); assess_sync for blocking code
async for atom_result in fa.astream(text):   # new: each AtomResult the moment its atom is verified
    ...

# custom
from factassessor.pipeline import Take
from factassessor import Atomizer, Filter, LayaCheckworthy, Search, SerperSearcher, Hedged, Verify, LayaJudge, Crawl4aiCrawler, Aggregate, LayaRunner

laya = LayaRunner()
pipeline = (
    Atomizer()
    >> Filter(LayaCheckworthy(laya))
    >> Take(8)
    >> Search(Hedged(SerperSearcher(), after=1.2))
    >> Verify(judge=LayaJudge(laya), crawler=Crawl4aiCrawler(timeout=2.5))
    >> Aggregate()
)
async with pipeline:                      # aload/aclose every component (browser, HTTP pool, Laya)
    result = await pipeline.run(text)
```

## Module layout

```
factassessor/
  pipeline.py        Step, Map, FlatMap, Fold, Pipeline (>>), Take, the runner
  schema.py          Atom, Claim, Hit, Page, Evidence, AtomResult, CheckResult
  atomizer.py        Atomizer (LLM, streaming)
  filters.py         Filter, predicate combinators, LayaCheckworthy
  search.py          Searcher, SerperSearcher, Search step
  crawl.py           Crawler, Crawl4aiCrawler
  wrappers.py        Hedged, Timeout, Retry
  evidence_judge.py  Judge, LayaJudge (+ passages.py for chunking/BM25/cleaning)
  verify.py          Verify step, VerdictPolicy, WeightedPolicy
  aggregate.py       Aggregate step, fact score, graph
  laya.py            LayaRunner
  assessor.py        FactAssessor: builds the default pipeline from today's constructor args
```

## Behaviour changes to decide

1. **`n_atoms` semantics.** Today: the *n highest-scoring* atoms. Streaming: `Take(n)` keeps the *first n* that
   pass the filter, because waiting to rank them all would be a barrier. Proposal: first-n (atoms are usually
   in text order, and n is a safety cap, not a ranking).
2. **Streaming atomizer.** pydantic-ai can stream structured output; we emit an atom once its list item is
   complete. Needs a short spike to confirm partial output is reliable with gpt-5.6-luna. Fallback: emit all atoms
   when the call finishes (today's behaviour), still under the same interface.
3. **Result order.** `astream` yields in completion order; `assess` sorts by atom id.

## Testing

- **Runner**: concurrency, completion-order emission, drop (`None`), flat-map fan-out, `Take`, cancellation
  propagating upstream, `Fold`.
- **Components**: existing tests move with their code (`test_search` → `SerperSearcher`, `test_crawl` →
  `Crawl4aiCrawler`, `test_verdicts` → `verify` + `aggregate`); wrappers get their own (hedging test already exists).
- **Verify**: existing early-exit tests (snippets settle; pages settle and remaining crawls are cancelled).
- **End to end** with fakes for every component, plus a live smoke run on the Nepal and mixed examples to
  confirm verdicts and that latency is no worse than the current ~5–6s.

## Rollout

One branch (`feat/streaming-pipeline`), in order: runner + tests → move components behind interfaces (no
behaviour change, tests green) → `Verify` + `Aggregate` steps → `FactAssessor` rebuilt on the pipeline →
streaming atomizer (after the spike) → notebooks updated to show `astream`.
