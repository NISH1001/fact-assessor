# fact-assessor, step by step

A guided tour from the three-line version to building your own pipeline. It's the same path as the notebook, which
runs every step live:

```bash
uv run marimo edit --no-sandbox notebooks/fa_walk.py
```

Example outputs below are from real runs (M-series Mac); live web results vary between runs. For how the machinery
works inside (streams, concurrency, cancellation), see [design/functional.md](design/functional.md).

## 1. The three-line version

```python
from factassessor import FactAssessor

async with FactAssessor() as fa:
    result = await fa.assess("It was believed that Nepal's earthquake in 2017 of 7.8 magnitude scale caused "
                             "massive damage. Total lives lost were 1 million people.")

result.fact_score                                     # 0.5
[(a.verdict, a.atom.text) for a in result.atoms]
# contested  Nepal's earthquake occurred in 2017.        (it was 2015)
# supported  Nepal's earthquake had a magnitude of 7.8.
# supported  Nepal's earthquake caused massive damage.
# refuted    Nepal's earthquake killed 1 million people.
```

`FactAssessor()` builds the default pipeline; `assess_sync(text)` is the same for non-async code.

## 2. Building blocks

Everything is a **step**: it takes an async stream of items and yields an async stream of items. Steps chain with
`>>` ("then"). On plain numbers:

```python
from factassessor import Filter, FlatMap, Map, Pred, Scan, Take, TakeUntil, collect, once

async def numbers():
    for n in range(10):
        yield n

await collect((Filter(lambda n: n % 2 == 0) >> Map(lambda n: n * 10) >> Take(3))(numbers()))   # [0, 20, 40]
```

| Step | Does |
|---|---|
| `Map(fn)` | one item in, one out (return `None` to drop it) |
| `FlatMap(fn)` | one item in, many out (the atomizer works like this: text → atoms) |
| `Filter(pred)` | keep items where `pred(item)` is true |
| `Take(n)` | first n items, then stop and cancel unfinished work upstream |
| `Scan(fn, initial)` | running total: `fn(total, item)` after each item |
| `TakeUntil(pred)` | items up to and including the first one where `pred` holds, then stop |

`once(x)` is a one-item stream (the usual way to feed a chain); `collect(stream)` gathers a stream into a list.

**Conditions and shorthand.** A `Pred` is a condition that combines with `&` (and), `|` (or), `~` (not). In a chain a
`Pred` filters and a plain function transforms, so you rarely write `Filter(...)` / `Map(...)`:

```python
even, big = Pred(lambda n: n % 2 == 0), Pred(lambda n: n > 5)
await collect(Filter(even & big)(numbers()))                        # [6, 8]
await collect((~even >> (lambda n: n * 10))(numbers()))             # [10, 30, 50, 70, 90]
await collect((Scan(lambda t, n: t + n, 0) >> TakeUntil(lambda t: t >= 10))(numbers()))   # [0, 1, 3, 6, 10]
```

**Async functions run concurrently**, and each result is passed on the moment it's ready:

```python
async def slow_square(n):
    await asyncio.sleep(n / 10)        # 3 -> 0.3s, 1 -> 0.1s, 2 -> 0.2s
    return n * n

await collect(Map(slow_square)(three_then_one_then_two()))   # [1, 4, 9] in 0.30s (not 0.6s; fastest first)
```

That's what makes the pipeline fast: a slow website never holds up a fast one, and a claim moves on as soon as its
own work is done. Sync functions keep input order (useful for ranked search results).

## 3. Each component on its own

Each component has a **role** (a base type) and implementations; to make your own you implement one method.

| Role | You implement | Built in |
|---|---|---|
| `Atomizer` | `atomize(text)` | `LLMAtomizer` |
| `ClaimFilter` | `score(atom)` → P(factual claim) | `LayaClaimFilter`, `GlinerClaimFilter` |
| `Searcher` | `search(query)` | `SerperSearcher`, `DuckDuckGoSearcher`, `SearxngSearcher` |
| `Crawler` | `crawl(url)` | `Crawl4AICrawler`, `HTTPXCrawler`, `FallbackCrawler` |
| `Judge` | `judge(claim, docs)` | `LayaJudge`, `GlinerJudge`, `LLMJudge` |
| `Policy` | `settled(evidence)`, `verdict(evidence)` | `WeightedPolicy` |

Model-backed components share their model automatically: `LayaClaimFilter` and `LayaJudge` use one copy of Laya,
and `GlinerClaimFilter` and `GlinerJudge` one copy of GLiNER. Nothing to pass around.

### 3a. `LLMAtomizer`: text → atoms

One LLM call splits the text into single-fact claims, each understandable on its own; each remembers the `span` of
the words it came from (for highlighting):

```python
atoms = await LLMAtomizer().atomize(text)
```

| atom | from the text |
|---|---|
| Nepal's earthquake occurred in 2017. | Nepal's earthquake in 2017 |
| Nepal's earthquake had a magnitude of 7.8. | Nepal's earthquake in 2017 of 7.8 magnitude scale |
| Nepal's earthquake caused massive damage. | It was believed that Nepal's earthquake … caused massive damage. |
| Nepal's earthquake killed 1 million people. | Total lives lost were 1 million people. |

### 3b. `LayaClaimFilter`: atoms → factual claims

Scores each atom's `claim_score` (P(it's a factual claim)); atoms below `threshold` (0.4) are dropped, as opinions,
greetings, and questions:

```python
claim_filter = LayaClaimFilter(threshold=0.4)
await claim_filter.score(atom)     # 0.92 for "Nepal's earthquake occurred in 2017.", ~0.05 for "I think this is sad."
```

| atom | Laya | GLiNER |
|---|---|---|
| Nepal's earthquake occurred in 2017. | 0.92 | 0.67 |
| Nepal's earthquake had a magnitude of 7.8. | 0.90 | 0.71 |
| Nepal's earthquake caused massive damage. | 0.95 | 0.64 |
| Nepal's earthquake killed 1 million people. | 0.97 | 0.75 |
| *(4 atoms)* | 154 ms | 503 ms |

On the 19-statement benchmark both get 17/19, but Laya's misses keep opinions (one wasted search) while GLiNER's drop
real claims (never checked), and GLiNER is slower, so Laya is the default.

### 3c. `SerperSearcher`: query → search hits

Google results as `{"url", "title", "snippet"}`, in rank order:

```python
hits = await SerperSearcher(num=10).search("Nepal's earthquake occurred in 2017.")
```

No API key? `DuckDuckGoSearcher()` works without one (slower), or `SearxngSearcher(url)` with your own SearXNG.

### 3d. `Crawl4AICrawler`: url → page

A headless browser fetches a page and returns clean plain text (no links, citations, or menus); a failed or slow page
is `None`:

```python
page = await Crawl4AICrawler(timeout=2.5).crawl(hits[0]["url"])   # {"url", "title", "text"} or None
```

The same 5 search hits through each crawler:

| crawler | pages read | all 5 at once |
|---|---|---|
| `Crawl4AICrawler` (browser) | 4/5 | 3.1s |
| `HTTPXCrawler` (plain HTTP, no JavaScript) | 3/5 | 1.5s |
| `FallbackCrawler(HTTPXCrawler(), Crawl4AICrawler())` | 5/5 | 0.7s (browser already warm) |

On 20 fresh URLs: browser 16/20 in 4.8s, HTTPX 11/20 in 1.0s, fallback 16/20 in 2.9s.

### 3e. `LayaJudge` + `WeightedPolicy`: evidence → verdict

The judge labels each passage *supports* / *refutes* / *not_enough_info* for a claim (pages are cut to their most
relevant passage first); the policy weighs strong evidence (≥ 0.7) into one verdict and decides when there's enough
to stop looking:

```python
judge, policy = LayaJudge(), WeightedPolicy(strong=0.7, early_exit=0.9)
evidence = await judge.judge(claim, hits[:5] + [page])
policy.verdict(evidence)       # ("contested", 0.51)
policy.settled(evidence)       # True if 2+ passages agree at >= 0.9 and none strongly disagree
```

The same evidence for "Nepal's earthquake occurred in 2017." (it was 2015) through each judge:

| judge | verdict | time | passage labels |
|---|---|---|---|
| `LayaJudge` | contested | 0.43s | refutes 0.96, refutes 0.58, supports 0.94, supports 0.95, refutes 0.75, not_enough_info 0.73 |
| `LLMJudge` (gpt-6-luna) | **refuted** | 1.69s | refutes 1.00, refutes 0.99, not_enough_info 0.99, refutes 0.99, refutes 0.99, not_enough_info 0.95 |
| `GlinerJudge` | unverified | 1.53s | every label below 0.5, so nothing counts as strong |

Laya is fooled by pages about a *different*, real 2017 Nepal earthquake; the LLM judge isn't. On the 15-case judge
benchmark: `LLMJudge` 15/15, `LayaJudge` 13/15, `GlinerJudge` 12/15; Laya is ~10x faster per pair.

## 4. Chaining components

Components are steps, so they chain with the building blocks. A condition filters **whatever flows at that point**:
atoms after the claim filter, search hits after the searcher.

```python
long_enough = Pred(lambda atom: len(atom.text) > 15)
atomizer_chain = LLMAtomizer() >> LayaClaimFilter(threshold=0.4) >> long_enough >> Take(8)
await collect(atomizer_chain(once(text)))                # the atoms worth checking

searcher_chain = SerperSearcher() >> not_blocked() >> Take(5)   # drop social/video sites, keep the top 5
await collect(searcher_chain(once(claim)))

await collect(crawler(urls))                             # every page crawled at once, yielded as each finishes
```

`Verify` runs one claim start to finish: judge the snippets; if they don't settle it, crawl every hit at once, judge
each page as it lands, and stop (cancelling the remaining crawls) once the policy says settled. It's built from the
same blocks: `crawler >> judge_page >> Scan(add, evidence) >> TakeUntil(policy.settled)`.

```python
result = await Verify(searcher_chain, crawler, judge, policy).verify(atom)   # one AtomResult
```

## 5. Putting it together

Two ways to give `FactAssessor` a claim filter:

```python
# 1. separate arguments: FactAssessor chains atomizer >> claim_filter >> Take(n_atoms) itself
fa = FactAssessor(
    atomizer=LLMAtomizer(),
    claim_filter=LayaClaimFilter(threshold=0.4),     # or GlinerClaimFilter(), or None for no filter
    searcher=SerperSearcher() >> not_blocked() >> Take(5),
    crawler=Crawl4AICrawler(),
    judge=LayaJudge(),                               # or LLMJudge(), GlinerJudge()
)

# 2. a chain that already filters (section 4): say so with claim_filter=None
fa = FactAssessor(atomizer=atomizer_chain, claim_filter=None, searcher=searcher_chain, crawler=crawler, judge=judge)
```

`stream` shows each claim as it's found and each verdict the moment it settles:

```python
async for event in fa.stream(text):
    if event.type == "claim_found":        # event.atom: show "checking…" at event.atom.span
        ...
    elif event.type == "claim_verified":   # event.result: verdict, confidence, evidence
        ...
    elif event.type == "done":             # event.result: the CheckResult (fact_score, all atoms, skipped)
        ...
```

`assess` is `stream` read to the end.

## 6. The knowledge graph

A view of the result, built on demand (~0.1ms), not a pipeline step:

```python
from factassessor import kg

graph = kg.build(result)        # {"nodes": [...], "edges": [...]}, plain dicts
kg.to_mermaid(graph)            # a Mermaid diagram
```

```
sentence ──contains──► claim ◄──supports 0.95 / refutes 0.88── passage ◄──published── source
claim ──same_sentence── claim
```

For the Nepal text: 2 sentences, 4 claims, 17 passages, 11 sources; 10 supports and 9 refutes edges; 3
`same_sentence` links (the three claims from the first sentence); 2 passages used by more than one claim. One passage,
*"July 2, 2017 Earthquake Information of Nepal · Magnitude 4.9"*, shows why "2017" was hard: it's a different, real
2017 earthquake.

## 7. Writing your own

- **A function or a `Pred`**: drop it straight into a chain (sync or async). Enough for most things.
- **A role subclass**: implement its one method; streaming, concurrency, and chaining come from the base.

```python
from factassessor import ClaimFilter, Crawler

class LengthFilter(ClaimFilter):
    async def score(self, atom):
        return 1.0 if len(atom.text) > 15 else 0.0       # P(factual claim)

class MyCrawler(Crawler):
    async def crawl(self, url):
        ...                                               # return {"url", "title", "text"}, or None on failure

FactAssessor(claim_filter=LengthFilter(), crawler=MyCrawler())
```

- **A `Step` subclass**: implement `__call__(items)` as an async generator, for steps that keep state across items:

```python
from urllib.parse import urlparse
from factassessor import Step

class OnePerSite(Step):
    """Search hits -> at most one per website (more independent sources)."""

    async def __call__(self, hits):
        seen = set()
        async for hit in hits:
            site = urlparse(hit["url"]).netloc.removeprefix("www.")
            if site not in seen:
                seen.add(site)
                yield hit

FactAssessor(searcher=SerperSearcher() >> not_blocked() >> OnePerSite() >> Take(5))
```
