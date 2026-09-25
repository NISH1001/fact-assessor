# The functional pipeline: design and implementation

How `factassessor` composes: what a step is, how `>>` works, how items stream and run concurrently, how
stopping early cancels work, and where (and why) the code is deliberately not pure.

## 1. The one idea: a step is a function from stream to stream

```python
class Step:
    def __call__(self, items: AsyncIterator[Any]) -> AsyncIterator[Any]: ...
```

Everything in the pipeline is this shape. The atomizer turns a stream of texts into a stream of atoms; a searcher
turns a stream of queries into a stream of hits; a crawler turns URLs into pages. Because every step has the
same shape, any step can follow any other, and a chain of steps is itself a step:

```python
atoms    = LLMAtomizer() >> LayaClaimFilter() >> Take(8)          # a Step: texts -> the atoms worth checking
searcher = SerperSearcher() >> not_blocked() >> Take(5)           # a Step: queries -> hits
result   = await collect(searcher(once("NASA was founded in 1958.")))
```

`once(x)` makes a one-item stream (the usual way to feed a chain); `collect(stream)` gathers one into a list.
A chain is a **value**: build it once, pass it around, extend it (`searcher >> OnePerSite()`), hand it to
`FactAssessor(searcher=...)`. Nothing runs until something reads the stream: chains are **lazy**.

## 2. `>>` is composition

`Chain` (`pipeline.py`) holds a list of steps and applies them in order:

```python
class Chain(Step):
    def __init__(self, *steps):  # nested chains are flattened: (a >> b) >> c == a >> (b >> c)
        self.steps = [s for step in steps for s in (step.steps if isinstance(step, Chain) else [step])]

    def __call__(self, items):
        for step in self.steps:
            items = step(items)   # each step wraps the previous step's stream; nothing runs yet
        return items
```

`Step.__rshift__` / `__rrshift__` build the chain, passing the other side through `as_step`, which is what lets a
chain contain plain functions and conditions:

| In a chain | Becomes | Example |
|---|---|---|
| a `Step` | itself | `SerperSearcher()` |
| a `Pred` (a condition) | `Filter(pred)` | `not_blocked()`, `official & ~is_video` |
| any other callable | `Map(fn)` | `lambda hit: hit["url"]` |

Why `>>` and not `>` or `|`: Python treats `a > b > c` as a chained comparison, `(a > b) and (b > c)`, which
would silently drop the first step. `|` already means "or" and is used for combining conditions (section 5).
`>>` reads as "then", has no such trap, and binds tighter than `|`/`&`, so `a >> (p | q) >> b` parses as it
looks.

## 3. The combinators

| Combinator | Shape | Notes |
|---|---|---|
| `Map(fn)` | 1 → 1 (or 0 if `fn` returns `None`) | `fn` sync or async |
| `FlatMap(fn)` | 1 → n | `fn(item)` is an async generator; the atomizer and searchers are FlatMaps inside |
| `Filter(pred)` | 1 → 1 or 0 | dropped atoms are reported as `skipped` (section 7) |
| `Take(n)` | first n, then stop | closes the stream upstream |
| `Scan(fn, initial)` | running total | yields `fn(total, item)` after every item |
| `TakeUntil(pred)` | up to and including the first match, then stop | closes the stream upstream |
| `last(stream)` | stream → its final item | pairs with `Scan`: the final running total |

Each is a few lines. For example:

```python
class Scan(Step):
    async def __call__(self, items):
        state = self.initial
        try:
            async for item in items:
                state = self.fn(state, item)
                yield state
        finally:
            await _aclose(items)   # if we're closed early, close whatever feeds us too
```

## 4. Streaming and concurrency: `_concurrently`

This is the one piece of real machinery; every async `Map`/`FlatMap`/`Filter` goes through it.

```
            items (upstream stream)
                  │
          feed() task: pulls items one at a time
                  │   (waits on a semaphore when `concurrency` workers are busy: backpressure)
                  ├──────────────┬──────────────┐
             work(item 1)   work(item 2)   work(item 3)     one task per item, all running at once
                  │              │              │
                  └──────► asyncio.Queue ◄──────┘           each output is put here the moment it exists
                                 │
                     consumer: `yield` from the queue       ← the step's output stream
```

- **Concurrent:** every item gets its own task, so a slow item (a dead website) never holds up a fast one.
- **Completion order:** outputs are yielded as they finish, not in input order. That's what lets the next step
  start on item 1 while item 5 is still running, and lets a page be judged the moment its crawl lands.
- **Backpressure:** with `concurrency=N`, the feeder only pulls a new item when a worker is free, so memory stays
  bounded however fast upstream produces.
- **Errors:** an exception in a worker is put on the queue and raised to the consumer.
- **Sync functions take a shortcut:** if `fn`/`pred` is synchronous there is nothing to overlap, so `Map`/`Filter`
  run it inline (`_in_order`), keeping input order and skipping the tasks. That's why
  `SerperSearcher() >> not_blocked() >> Take(5)` keeps Google's ranking.

## 5. Conditions: `Pred` and `&`, `|`, `~`

`Pred` wraps a condition so it combines:

```python
official = Pred(lambda hit: hit["url"].endswith((".gov", ".edu")))
searcher = SerperSearcher() >> (not_blocked() & official) >> Take(5)
```

`&` and `|` build a new `Pred` that **short-circuits** (the right side only runs if the left doesn't decide) and
evaluates each side at most once. If either side is async, the combined condition is async; otherwise it stays a
plain sync function (so it keeps the fast, ordered path in section 4). `~` negates. A `Pred` is a condition on
**one item**; to run it over a stream, put it in a chain or wrap it in `Filter`.

## 6. Stopping early cancels work

Closing a stream runs the `finally` blocks down the chain: `_concurrently` cancels its feeder and every worker
task still running, then closes its own input. So when `Take(3)` has its three items, or `TakeUntil` sees a
settled claim, or a consumer `break`s, or a timeout fires, **every unfinished crawl, search, and judgment above it
is cancelled**. Nothing keeps running (or holding a browser tab) for a result nobody will read.

`Verify` (the per-claim step) is built from exactly these pieces:

```python
evidence = await self.judge.judge(claim, hits)            # snippets first: often enough on their own
if hits and not self.policy.settled(evidence):
    gather_evidence = (
        self.crawler                                      # every hit crawled at once; pages in finish order
        >> judge_page                                     # a bare async function: a Map; each page judged on arrival
        >> Scan(operator.add, evidence)                   # running total of the evidence
        >> TakeUntil(self.policy.settled)                 # stop at the first total that settles the claim...
    )                                                     # ...which cancels the crawls still in flight
    evidence = await last(gather_evidence(urls(hits)), default=evidence)
```

Read top to bottom: crawl, judge each page, keep a running total, stop when settled. The early exit that took the
Nepal check from ~8.9s to ~5–6s is the `TakeUntil` line.

## 7. Where it's deliberately not pure

Pure functions would be simpler to reason about, but a fact-checker has to touch the world. The impurity is
contained in a few, named places:

- **Resources live in components.** A browser (`Crawl4AICrawler`), an HTTP pool (`SerperSearcher`), and a model
  (the Laya runtime) are objects with state. Each step declares them: `start()` / `stop()` for its own, and anything with
  `aload` / `aclose` it holds. `Step.aload()` walks the chain (`_nodes`) and starts every resource exactly once, even
  when it's shared. Models are shared further, per process: one Laya per device and one GLiNER per variant, however
  many components use them, so nothing has to be passed around.
- **`skipped` atoms use a side channel.** A `Filter` drops items; to report which *atoms* were dropped without
  changing every step's output type, `Filter` appends them to a context variable (`pipeline.dropped`) that
  `FactAssessor.stream` sets per run. Context variables are per task, so concurrent checks don't mix.
- **`FactAssessor` is a facade object.** It builds the default chain from familiar arguments, owns the resources and
  their lifecycle, and runs a chain as a stream of events (`ClaimFound` → `ClaimVerified` → `Done`). `assess` is
  `stream` read to the end; `assess_sync` runs it on a background event loop.
- **Pydantic models are mutable,** but the code treats them as values (`atom.model_copy(update=...)`).

## 8. Roles: one method to implement

Each component has a role, a base type that supplies the step behaviour so an implementation only writes its
one method:

| Role | Implement | Step behaviour from the base | Implementations |
|---|---|---|---|
| `Atomizer` | `atomize(text) -> list[Atom]` | FlatMap: text → atoms | `LLMAtomizer` |
| `ClaimFilter` | `score(atom) -> float` | Map: keeps atoms scoring ≥ threshold, reports the rest as skipped | `LayaClaimFilter`, `GlinerClaimFilter` |
| `Searcher` | `search(query) -> list[hit]` | FlatMap: query → hits | `SerperSearcher`, `DuckDuckGoSearcher`, `SearxngSearcher` |
| `Crawler` | `crawl(url) -> page or None` | Map: urls → pages, concurrent, finish order | `Crawl4AICrawler` |
| `Judge` | `judge(claim, docs) -> list[Evidence]` | (called by `Verify`) | `LayaJudge`, `GlinerJudge`, `LLMJudge` |
| `Policy` | `settled(ev)`, `verdict(ev)` | (called by `Verify`) | `WeightedPolicy` |

```python
class Crawler(Step, ABC):
    @abstractmethod
    async def crawl(self, url): ...                 # yours: fetch one page, None on failure

    def __call__(self, urls):
        return Map(self.crawl)(urls)                # ours: every URL at once, pages as they finish
```

Performance cost of all of this: none measurable. The combinators are microseconds per item; a check spends its
seconds on network calls, crawling, and model passes. (A/B vs the pre-refactor code: within network noise.)

## 9. Map of the code

| File | What |
|---|---|
| `pipeline.py` | `Step`, `Chain`, `Map`, `FlatMap`, `Filter`, `Take`, `Scan`, `TakeUntil`, `Pred`, `as_step`, `_concurrently`, lifecycle walk |
| `atomizer.py`, `search.py`, `crawl.py`, `evidence_judge.py`, `gliner.py`, `verify.py` | roles and implementations |
| `claim_filter.py` | `ClaimFilter` (role: `score(atom)`), `LayaClaimFilter` |
| `assessor.py` | `FactAssessor`: default chain, `stream` / `assess` / `assess_sync`, lifecycle |
| `schema.py` | data types; `CheckResult.fact_score` is computed from the atoms |
| `kg.py` | knowledge graph, built on demand from a result |
