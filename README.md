# fact-assessor

Fast, async-first fact assessment for any piece of text: a sentence, a paragraph, a model's answer.
It splits the text into atomic claims, finds web evidence for each one, and returns a verdict per claim,
an overall fact score, and a knowledge graph linking claims to their sources.

```python
from factassessor import FactAssessor, kg

async with FactAssessor() as fa:
    result = await fa.assess("Nepal's earthquake in 2017 of 7.8 magnitude caused massive damage. Total lives lost were 1 million people.")

for atom in result.atoms:
    print(atom.verdict, atom.atom.text)
# contested  Nepal's earthquake occurred in 2017.        (it was 2015; see Roadmap: per-detail checks)
# supported  Nepal's earthquake had a magnitude of 7.8.
# supported  Nepal's earthquake caused massive damage.
# refuted    Nepal's earthquake killed 1 million people.

result.fact_score   # 0.5
kg.build(result)    # knowledge graph: sentences -> claims <- evidence passages (supports / refutes, prob)
```

It's built for interactive use (select text in a UI, see verdicts in seconds): every step is async, all claims
are checked concurrently, and most of the work is I/O that overlaps.

> Status: alpha. The API will change as the pipeline becomes composable (see [Roadmap](#roadmap)).

## How it works

```
text
 └─ LLMAtomizer ──────── one LLM call: atomic, self-contained claims ("Total lives lost…" → "The Nepal earthquake killed…")
     └─ DecisionClaimFilter ─ Laya, local: drop opinions, greetings, questions
         └─ search ───── Serper (Google), every claim in parallel; slow requests are hedged
             └─ judge ── Laya, local: does each snippet support / refute the claim?
                 ├─ settled → done (no crawling)
                 └─ not yet → [resolve: where can each hit be read in full? (optional: arXiv, OpenAlex)]
                              crawl pages (crawl4ai) in parallel, judge each page as it lands,
                              stop as soon as the evidence settles the claim
 └─ aggregate ─────────── verdict per claim → fact score   (knowledge graph: kg.build(result), on demand)
```

| Step | What does it | Where it runs |
|---|---|---|
| Atomize + decontextualize | `LLMAtomizer`: [pydantic-ai](https://ai.pydantic.dev) → `openai:gpt-5.6-luna` (reasoning as low as the model allows) | API, ~2s |
| Claim filter | `DecisionClaimFilter`: one `choice` decision per atom (is this a factual claim?) on a decision runner, [Laya](https://github.com/NandhaKishorM/laya) by default | local (MPS / CUDA / CPU) |
| Search | [Serper](https://serper.dev); social media and video sites excluded in the query (`-site:`, so Google fills those slots) and filtered after | API, ~1s |
| Resolve (optional) | `CompositeResolver(ArxivResolver(), OpenAlexResolver())`: a paper's full text from its free copies | network, ~0.2s/paper |
| Crawl | [crawl4ai](https://github.com/unclecode/crawl4ai), one shared headless browser, cleaned plain text; or `HTTPXCrawler` (HTML and PDF) | network, ~1s/page |
| Rank passages | `Ranker`: which chunks of a page the judge sees. `BM25Ranker` (default, word overlap); `HybridRanker` adds 8M-parameter static embeddings for paraphrase (`fact-assessor[embed]`) | local, ms |
| Evidence judge | `DecisionJudge`: claim and evidence in one Unicode form (`ha⁻¹` = `ha−1`), pages cut into 90-word windows, the ranker's top passages per page, one decision each on the same runner | local |
| Verdicts, score, graph | strong evidence weighed per side: `supported` / `refuted` / `contested` / `unverified` | local |

Every box above is a swappable, chainable step (`SerperSearcher() >> not_blocked() >> Take(5)`), and the whole
thing streams: a claim starts searching the moment it's found, each page is judged the moment its crawl lands,
and results come out as each claim settles. See [Compose your own pipeline](#compose-your-own-pipeline).

Laya is a non-autoregressive decision model (a Jev-style encoder that classifies instead of generating), so the
filter and the judge are single forward passes. They ask it through a **decision runner** (`DecisionRunner`), the
one place a model is wired in: `LayaRunner` (the default) merges every request that arrives within a few
milliseconds, from any claim or page, into one batch on the local GPU; `SystemOneRunner` sends the same requests
to TypeSafe's Jev on OpenRouter instead (no GPU). `FactAssessor()` gives the filter and the judge one shared
`LayaRunner`; see [Decision runners](#decision-runners-the-model-behind-the-filter-and-the-judge).

**Latency** (M-series Mac, MPS, warm): ~4–6s for a 2–5 claim paragraph, most of it network. The atomizer call,
search, and crawling dominate; Laya passes take 40–150ms each. The first call in a fresh process also loads Laya
and starts the browser (several seconds, ~30s the very first time); call `await fa.aload()` up front to pay that
before the user is waiting. Evidence comes from the live web, so verdicts on borderline claims can vary between runs.

## Quick start: the notebook, one line

Needs [uv](https://docs.astral.sh/uv/), a [Serper](https://serper.dev) key, and an OpenAI key. No clone needed:

```bash
export SERPER_API_KEY=... OPENAI_API_KEY=...        # or put them in a .env in the current folder
uvx --from crawl4ai crawl4ai-setup                  # one time: installs the headless browser used for crawling
uvx marimo edit --sandbox https://raw.githubusercontent.com/NISH1001/fact-assessor/main/notebooks/fa.py
```

Type or paste text, press **Fact-check**, and get the fact score, a knowledge graph of claims and sources, and
every piece of evidence. The first run downloads Laya's weights (~0.8 GB) and warms up; after that a check takes
a few seconds.

## Install

As a dependency of your project:

```bash
uv add git+https://github.com/NISH1001/fact-assessor
uv run crawl4ai-setup          # one time: headless browser for crawling
```

Optional extras: `fact-assessor[gliner]` (the GLiNER2.5-decide judge, via onnxruntime) and `fact-assessor[ddg]`
(DuckDuckGo search, no API key), e.g. `uv add "fact-assessor[gliner,ddg] @ git+https://github.com/NISH1001/fact-assessor"`.

For development:

```bash
git clone https://github.com/NISH1001/fact-assessor && cd fact-assessor
uv sync
uv run crawl4ai-setup
cp .env.example .env           # then fill in the keys
uv run marimo edit --no-sandbox notebooks/fa.py         # pick searcher / crawler / judge, see the code, run it
uv run marimo edit --no-sandbox notebooks/fa_walk.py    # guided walk: building blocks -> custom pipeline (also docs/WALKTHROUGH.md)
```

| Variable | Used by |
|---|---|
| `SERPER_API_KEY` | web search ([serper.dev](https://serper.dev)) |
| `OPENAI_API_KEY` | the atomizer (swap in any [pydantic-ai model](https://ai.pydantic.dev/models/), see below) |

A `.env` in the working directory (or a parent) is loaded when `factassessor` is imported; variables already set
in the environment win.

## Usage

### Check a text

```python
import asyncio
from factassessor import FactAssessor

async def main():
    async with FactAssessor() as fa:
        result = await fa.assess("Marie Curie won the Nobel Prize in Physics in 1903. The Eiffel Tower is 500 meters tall.")
    print(f"fact score {result.fact_score:.0%} in {result.latency_ms / 1000:.1f}s")
    for a in result.atoms:
        print(f"{a.verdict:10s} {a.confidence:.2f}  {a.atom.text}")

asyncio.run(main())
# example output (evidence comes from the live web, so borderline verdicts can vary between runs):
# fact score 50% in 4.4s
# supported  0.97  Marie Curie won the Nobel Prize in Physics in 1903.
# refuted    0.96  The Eiffel Tower is 500 meters tall.
```

In Jupyter or marimo, `await` works at the top level: `result = await fa.assess(text)`.

Not in async code? `assess_sync` blocks and returns the same result:

```python
from factassessor import FactAssessor

with FactAssessor() as fa:                     # closes the browser and HTTP pool on exit
    result = fa.assess_sync("NASA was founded in 1958.")
    result = fa.assess_sync("Python was created by James Gosling.")   # reuses the warm assessor
```

It runs on a background event loop owned by the assessor, so it also works where a loop is already running
(Jupyter), and repeated calls stay fast. Pick one style per instance: `assess` or `assess_sync`, not both.
(`acheck` is an alias for `assess`.)

### Keep one assessor alive

Creating a `FactAssessor` is cheap, but the first check loads Laya and starts a browser. In a notebook, app, or
service, create one, warm it up once, and reuse it for every check:

```python
fa = FactAssessor(n_atoms=8)
await fa.aload()              # load Laya + start the browser now (~3s), not on the first user request
...
result = await fa.assess(text)   # reuse for every check
...
await fa.aclose()             # on shutdown: closes the browser and HTTP pool (or use `async with`)
```

### Stream results

`stream` yields each claim as it's found and each verdict the moment it settles, so a UI can show progress
instead of waiting for the slowest claim:

```python
async for event in fa.stream(text):
    if event.type == "claim_found":        # event.atom: underline event.atom.span as "checking…"
        ...
    elif event.type == "claim_verified":   # event.result: an AtomResult (verdict, confidence, evidence)
        ...
    elif event.type == "done":             # event.result: the full CheckResult (score, all atoms)
        ...
```

`assess` is `stream` read to the end. Events are Pydantic models (`ClaimFound`, `ClaimVerified`, `Done`), so
they serialize straight to JSON for a websocket or server-sent events.

### Read the result

```
CheckResult
├── fact_score      supported / (supported + refuted + contested); None if nothing was checkable
├── latency_ms
├── atoms           list[AtomResult], in text order
│   ├── atom        Atom: text (self-contained claim), span (char offsets into your input), claim_score (P(factual claim))
│   ├── verdict     "supported" | "refuted" | "contested" | "unverified"
│   ├── confidence  0..1
│   ├── evidence    list[Evidence]: url, title, text (passage), source ("snippet" | "page"), label, prob
│   └── error       set if this claim failed (e.g. search down); the rest of the check still completes
├── skipped         list[Atom]: opinions, greetings, questions the filter dropped
└── (graph)         not stored: kg.build(result) builds it on demand, see "Knowledge graph" below
```

Everything is a Pydantic model, so `result.model_dump()` / `model_dump_json()` gives JSON for a UI. To highlight
claims in the original text, use each atom's `span`:

```python
for a in result.atoms:
    start, end = a.atom.span
    print(f"[{a.verdict}] {text[start:end]!r}")
```

### Knowledge graph

`kg.build(result)` turns a result into a graph (plain dicts, JSON-ready); `kg.to_mermaid(graph)` draws it. It's a view
of the result, not part of the pipeline, and takes ~0.1ms, so build it when you show it.

```
sentence ──contains──► claim ◄──supports 0.95 / refutes 0.88── passage ◄──published── source
claim ──same_sentence── claim
```

| Node | What |
|---|---|
| `sentence` | a sentence of your text (its `span`); claims point back to the sentence they came from |
| `claim` | an atom, with `verdict`, `confidence`, `span` |
| `passage` | the evidence text a judge decided on, with `url`, `site`; one node even when several claims used it |
| `source` | the site (`en.wikipedia.org`) |

Evidence below the `strong` threshold (0.7) and `not_enough_info` are left out by default
(`kg.build(result, strong=0.7, include_not_enough_info=False)`).

### Configure

All keyword arguments to `FactAssessor`:

| Argument | Default | Meaning |
|---|---|---|
| `n_atoms` | 5 | max claims checked per text |
| `top_k` | 5 | search results per claim |
| `atomizer_model` | `openai:gpt-5.6-luna` | any pydantic-ai model string |
| `device` | `auto` | Laya device: cuda → mps → cpu |
| `claim_threshold` | 0.4 | min `claim_score` (P(factual claim)) to check an atom; low on purpose, since a dropped real claim is never checked |
| `early_exit_conf` | 0.9 | 2+ passages this sure (and none against) settle a claim without more crawling |
| `strong_evidence` | 0.7 | min probability for a passage to count toward a verdict |
| `crawl_timeout` | 2.5 | seconds per page (a hard limit; a page that takes longer is dropped and the claim goes on without it) |
| `search_hedge_after` | 1.2 | if a search hasn't answered by then, send the same request again and use whichever reply comes first (fixes Serper's occasional 3s+ outliers; only slow searches cost a second credit; `None` turns it off) |
| `blocked_domains` | social + video | hosts never used as evidence (subdomains included): dropped after search and, with Serper, excluded in the query; `()` to allow all |
| `timeout` | 15 | per-claim deadline; a claim still running then is decided on the evidence judged so far (`error="timeout"`) |
| `max_concurrent_claims` | the judge's | claims checked at once; a claim's `timeout` starts when it gets its turn. The judge takes it from its runner: no limit on Laya, Jev and LLMs; 3 on GLiNER. `None` = no limit |
| `max_concurrent_crawls` | 10 | pages the browser crawler loads at once (shared by all claims) |
| `search_timeout` | 5 | seconds per Serper request |
| `laya_model` | `english` | Laya checkpoint of the default runner (shared by the filter and the judge): `english`, `multilingual`, `typed-decisions` |
| `serper_api_key` | `SERPER_API_KEY` | Serper key (from `.env` or the environment if not given) |

### Compose your own pipeline

Every component is a step, and steps chain with `>>` ("then"). In a chain, a **`Predicate`** (a condition) filters
whatever flows at that point (atoms after the atomizer, search hits after the searcher) and a **plain function**
transforms it. Conditions combine with `&` (and), `|` (or), `~` (not).

```python
from urllib.parse import urlparse
from factassessor import Crawl4AICrawler, FactAssessor, DecisionClaimFilter, Predicate, SerperSearcher, Take, not_blocked

official = Predicate(lambda hit: urlparse(hit["url"]).netloc.endswith((".gov", ".edu")))   # a condition
long_enough = Predicate(lambda atom: len(atom.text) > 15)

fa = FactAssessor(
    claim_filter=DecisionClaimFilter(threshold=0.5) >> long_enough,
    searcher=SerperSearcher(num=20) >> (not_blocked() & official) >> Take(5),
    crawler=Crawl4AICrawler(timeout=4.0),
)
result = await fa.assess("NASA was founded in 1958. The Eiffel Tower is 500 meters tall.")
# supported   NASA was founded in 1958.              (nasa.gov, eisenhowerlibrary.gov)
# unverified  The Eiffel Tower is 500 meters tall.   (no .gov/.edu sources)
```

`Cache(step)` wraps any step or sub-chain in a TTL cache (10 minutes by default): the same input is processed once,
concurrent duplicates share the work in flight, failures aren't kept. `FactAssessor` uses it around the searcher so a
text's claims share one search for the text's source query; `Cache(searcher_chain)` or `Cache(crawler)` work the same
way by hand.

What you can pass. Each component has a **role**. `Judge`, `Policy`, `Resolver`, `Ranker` and `DecisionRunner` are
Protocols: any object with the method works, no subclassing needed. `Atomizer`, `Searcher`, `Crawler` and
`ClaimFilter` are `Step` base classes: subclass one, implement its method, and streaming, concurrency, and chaining
come for free.

| Argument | Role | You implement | Default |
|---|---|---|---|
| `atomizer=` | `Atomizer` (or a chain starting with one) | `atomize(text) -> list[Atom]` | `LLMAtomizer()` |
| `claim_filter=` | `ClaimFilter` (or any step; `None` = no filter) | `score(atom) -> P(factual claim)` | `DecisionClaimFilter(threshold=0.4)` on Laya; `DecisionClaimFilter(runner)` for another model |
| `searcher=` | `Searcher` (or a chain) | `search(query) -> list[hit]`, hits `{"url", "title", "snippet"}` | `SerperSearcher() >> not_blocked() >> Take(top_k)` |
| `resolver=` | `Resolver` (a Protocol) | `resolve(url) -> list[str]`: where the hit can be read in full, best first | none; `CompositeResolver(ArxivResolver(), OpenAlexResolver())` for papers |
| `crawler=` | `Crawler` | `crawl(url) -> page or None`, pages `{"url", "title", "text"}` | `Crawl4AICrawler(timeout=2.5)`; also `HTTPXCrawler`, `FallbackCrawler` |
| `judge=` | `Judge` (a Protocol) | `judge(claim, docs) -> list[Evidence]` | `DecisionJudge()` on Laya; `DecisionJudge(LLMRunner())`, `DecisionJudge(SystemOneRunner())`, `DecisionJudge(GlinerRunner())` for other models. Takes `ranker=` (a `Ranker`: `top(claim, chunks, k)`), default `BM25Ranker()`; `HybridRanker()` mixes in embeddings |
| `policy=` | `Policy` (a Protocol) | `settled(evidence)`, `verdict(evidence) -> (verdict, confidence)` | `WeightedPolicy()` |

A new crawler, for example, is just:

```python
from factassessor import Crawler

class MyCrawler(Crawler):
    async def crawl(self, url):
        ...  # fetch; return {"url", "title", "text"} or None on failure

fa = FactAssessor(crawler=MyCrawler())
```

Building blocks:

| | Does |
|---|---|
| `Map(fn)` or a bare `fn` in a chain | one in, one out (return `None` to drop) |
| `Filter(pred)` or a `Predicate` in a chain | keep items where the condition holds |
| `FlatMap(fn)` | one in, many out |
| `Take(n)` | first n items, then stop |
| `Scan(fn, initial)` | running total: yields `fn(total, item)` after each item |
| `TakeUntil(pred)` | items up to and including the first one where `pred` holds, then stop |
| `Step` | subclass it: `__call__(items)` as an async generator (+ `start`/`stop` if it holds a resource) |

Functions and conditions can be sync or async; async ones run concurrently and pass results on as they finish, sync
ones keep order. Stopping early (`Take`, `TakeUntil`, a timeout) cancels unfinished work upstream. `Verify` itself is
built from these: `crawler >> judge_page >> Scan(add, evidence) >> TakeUntil(policy.settled)`: crawl every hit,
judge each page as it lands, keep a running total, stop (cancelling the rest) once the claim is settled.

Another LLM for atomization (install its extra first, e.g. `uv add "pydantic-ai-slim[anthropic]"`):
`LLMAtomizer("anthropic:claude-haiku-4-5", model_settings={})`.

How the composition works under the hood (streams, `>>`, concurrency, cancellation, where it isn't pure):
[docs/design/functional.md](docs/design/functional.md).

### Other crawlers, judges, and searchers

Every built-in option, one line per role. Mix freely; anything not passed keeps its default.

```python
from factassessor import (
    FactAssessor, LLMAtomizer,                                      # atomizer
    DecisionClaimFilter,                                            # claim filter
    SerperSearcher, DuckDuckGoSearcher, SearxngSearcher, not_blocked, Take,  # searcher
    Crawl4AICrawler, HTTPXCrawler, FallbackCrawler,                 # crawler
    DecisionJudge,                                                  # judge
    LayaRunner, SystemOneRunner, LLMRunner, GlinerRunner,           # the model behind the filter and the judge
    WeightedPolicy,                                                 # policy
)

FactAssessor(
    atomizer=LLMAtomizer("openai:gpt-6-luna"),                       # any pydantic-ai model
    claim_filter=DecisionClaimFilter(threshold=0.4),                 # Laya; or DecisionClaimFilter(GlinerRunner()), or None (no filter)
    searcher=SerperSearcher() >> not_blocked() >> Take(5),           # or DuckDuckGoSearcher(), SearxngSearcher(url)
    crawler=Crawl4AICrawler(timeout=2.5),                            # or HTTPXCrawler(), FallbackCrawler(HTTPXCrawler(), Crawl4AICrawler())
    judge=DecisionJudge(),                                           # Laya; or DecisionJudge(LLMRunner()), DecisionJudge(SystemOneRunner()), DecisionJudge(GlinerRunner())
    policy=WeightedPolicy(strong=0.7, early_exit=0.9),
)
```

Keep `>> not_blocked() >> Take(top_k)` when you pass your own searcher: `FactAssessor` adds it only around its
default Serper searcher. The same components chain by hand too (see "Compose your own pipeline" above).

**Faster crawling.** `HTTPXCrawler` fetches pages with a plain HTTP request (no browser, no JavaScript);
`FallbackCrawler` tries crawlers in order and keeps the first page with text:

```python
from factassessor import Crawl4AICrawler, FallbackCrawler, HTTPXCrawler

FactAssessor(crawler=HTTPXCrawler())                                          # fastest; misses JavaScript pages
FactAssessor(crawler=FallbackCrawler(HTTPXCrawler(), Crawl4AICrawler()))      # fast first, browser only if needed
```

| Crawler (same 20 live search-result URLs) | Pages read | All 20 at once | Per page (median) |
|---|---|---|---|
| `Crawl4AICrawler` (default) | 16/20 | 4.8s | 2.14s |
| `HTTPXCrawler` | 11/20 | 1.0s | 0.15s |
| `FallbackCrawler(HTTPX, browser)` | 16/20 | 2.9s | 1.15s |

**Scientific papers: read in full, even behind bot protection.** Publishers like Wiley and IOP block headless
browsers (0 of 15 paper pages crawled on a scientific eval set, even with a 10s timeout), and an arXiv or publisher
link usually lands on an abstract. Give the pipeline a resolver: a step before the crawler that finds where each
hit can be read in full (`async resolve(url) -> list[str]`, the `Resolver` protocol). The crawler is unchanged:
it just crawls those locations in order (`fact-assessor[pdf]` for PDFs):

```python
from factassessor import (ArxivResolver, CompositeResolver, Crawl4AICrawler, FallbackCrawler, HTTPXCrawler,
                          HybridRanker, DecisionJudge, OpenAlexResolver)

FactAssessor(
    source_query=True,                          # the atomizer also writes one search for the text's source document
    resolver=CompositeResolver(ArxivResolver(), OpenAlexResolver()),   # papers: free full-text copies first
    crawler=FallbackCrawler(HTTPXCrawler(), Crawl4AICrawler()),        # plain HTTP (HTML and PDF), browser only if needed
    judge=DecisionJudge(passages_per_page=3),       # 3 passages per page: +0.10 F1 on the paper eval, ~3s more per text
)

DecisionJudge(passages_per_page=3, ranker=HybridRanker())   # BM25 + 8M static embeddings (fact-assessor[embed]); measured:
                                                        # no gain over BM25 at top-3 on the paper eval, kept as an option
```

`source_query`: a claim about a detail inside a paper rarely finds the paper by itself, while the whole text usually
does; the atomizer writes that query in the same call as the atoms (no added latency), every claim searches with it
too, and it is searched once per text (`FactAssessor` wraps its searcher in `Cache`).

### Decision runners: the model behind the filter and the judge

`DecisionClaimFilter` and `DecisionJudge` don't know which model answers them. They build a request (a `state`
such as `{"evidence": passage, "claim": claim}` and a typed `question` with its options) and hand it to a
`DecisionRunner`: one method, `predict(requests) -> responses`, with probabilities per option. Batching is the
runner's job: it merges the requests of every claim, page and component in flight and sends them in full batches.

```python
from factassessor import DecisionClaimFilter, DecisionJudge, FactAssessor, LayaRunner, SystemOneRunner

laya = LayaRunner()                       # in-process Laya (the default): 32-row passes on the local GPU
jev = SystemOneRunner()                   # TypeSafe's Jev on OpenRouter (OPENROUTER_API_KEY): no GPU, ~0.5s a call,
                                          # a claim's passages in one call (up to 40), 16 calls in flight, $0.042 per 1M input tokens
FactAssessor(claim_filter=DecisionClaimFilter(laya), judge=DecisionJudge(jev, passages_per_page=3))
FactAssessor(judge=DecisionJudge(SystemOneRunner(url="http://gpu-box:8000/v1/systemone", model="english")))  # a remote `python -m laya.serve`
```

| Runner | Model | Where | Batching |
|---|---|---|---|
| `LayaRunner(model="english")` (default) | Laya: `english`, `multilingual` (~2.2x faster), `typed-decisions` | local GPU / CPU, one model per device per process | requests merged across callers, 32 rows per pass |
| `SystemOneRunner(model="~typesafe/jev-latest")` | Jev (System One protocol), or a `laya.serve` server | OpenRouter, or any URL | one claim's passages per call as a list field (up to 40; Jev's "ask every question about the same state in one request"), 16 calls in flight, 429s retried; `packing="all"` fills calls with every claim in flight instead, `"none"` sends each request alone |

A runner is a `Protocol`: anything with `batch_size` and `async predict(requests)` works, and
`isinstance(x, DecisionRunner)` checks it. `factassessor.decisions` has the request and response models.

```
search -> judge snippets -> resolve (every hit at once) -> crawl each hit's locations in order -> judge each page
```

- `ArxivResolver`: any arXiv link (or arXiv DOI) -> `arxiv.org/html/<id>`, then `arxiv.org/pdf/<id>`; no request.
- `OpenAlexResolver`: a URL with a DOI -> the paper's open-access copies from [OpenAlex](https://openalex.org)
  (one free lookup, no key), PDFs first; doi.org links are left out (they redirect back to the publisher).
- `CompositeResolver(a, b)`: asks every resolver at once and joins their candidates, so a later resolver's copies
  are still tried when an earlier one's are all blocked.

A hit's locations are: the hit itself if it's a direct PDF, then the copies, then the hit's own page. Resolvers only
propose; the crawl checks by fetching, so a 403 or a bot-check page moves on to the next. A copy needs 300 words
(Wiley's and HAL's bot-check pages are ~180); one hit's locations share an 8s deadline (`read_timeout`). The page is
cited under the hit's URL. Ordinary pages cost nothing extra: resolvers answer `[]` for them without a request.

On that set, OpenAlex's copies recovered 87 of 162 failed pages that had a DOI; the rest have no free copy.

End to end the gain is smaller, since many claims settle on search snippets without crawling (Nepal example:
6.1s → 5.2s median; mixed example: about the same). `HTTPXCrawler` has a 1s connect timeout plus a hard 2.5s total
deadline per page (8s once the response is a PDF; httpx's own timeouts are per phase), reads HTML and PDFs
(`fact-assessor[pdf]`), and stops at 3 MB for pages and 20 MB for PDFs. PDFs are parsed in worker processes
(`extract.PDF_WORKERS`, 2): PDFium is C code parsing documents from the web, and a crash there (seen once in a live
run) loses only that PDF instead of the whole pipeline.

**GLiNER2.5-decide judge** ([GLiNER2.5-decide](https://fastino.ai/blog/gliner-2-5-decide-open-weight-decision-model),
another Jev/Laya-style decision model, as ONNX from
[nishparadox/gliner2.5-decide-onnx](https://huggingface.co/nishparadox/gliner2.5-decide-onnx); CPU, no torch):

```python
from factassessor import DecisionJudge, GlinerRunner   # needs fact-assessor[gliner]; downloads ~1.75 GB on first use

fa = FactAssessor(judge=DecisionJudge(GlinerRunner()))   # variant="int8" is 2x faster but much less accurate
```

GLiNER runs on CPU (~7 passages/s on an M3 Max; ONNX Runtime's CoreML path crashes or is slower), so
`GlinerRunner` sets `concurrency=3`: a judge on it checks 3 claims at a time, and the rest wait for a turn instead
of timing out together. End to end on the synthetic set it's far behind Laya: 22% of all claims right vs 92%, at
14.3s per text vs 1.6s (`data/results/`).

**LLM runner** (`LLMRunner`, pydantic-ai structured output; default `openai:gpt-6-luna`, reasoning off): the
questions of a call become numbered items of one prompt, the model picks an option per item with a confidence.
The most accurate judge we measured, ~10x slower per pair than Laya, which matters less than it sounds: pairs run
in parallel alongside searching and crawling (a 3-claim check took 5.5s with it vs 5.3s with Laya).

```python
FactAssessor(judge=DecisionJudge(LLMRunner()))                    # needs OPENAI_API_KEY
FactAssessor(judge=DecisionJudge(LLMRunner("openai:gpt-5.4-mini", batch_size=20)))   # 20 items per call
```

| Judge (15-case benchmark, M-series Mac) | Correct | Time for 15 pairs |
|---|---|---|
| `LLMRunner` gpt-6-luna | **15/15** (3 of 4 runs; 14 in the other) | ~2.1s (API) |
| `LLMRunner` gpt-5.6-luna | 14/15 | ~2.1s (API) |
| `LayaRunner` (default, MPS) | 13/15 | 0.2s |
| `GlinerRunner` fp32 (CPU) | 12/15 | 1.8s |
| `GlinerRunner` int8 (CPU) | 7/15 | 1.0s |

GLiNER's misses were all false "supports" (it let "NASA was founded in 1972" through). The LLM numbers were
measured with the judge's earlier fact-checking prompt; `LLMRunner` renders the same question and options for any
decision, so re-run the benchmark before relying on them. It packs up to 40 items per call and merges concurrent
callers (measured earlier: one packed call was ~0.2s slower than 15 parallel single calls, for a fraction of the
tokens). Reproduce with `uv run --extra gliner python scripts/compare_judges.py`.

**Claim filters** (`FactAssessor(claim_filter=DecisionClaimFilter(GlinerRunner()))`, or `claim_filter=None` to
check every atom; `scripts/compare_claim_filters.py`, 19 statements): on Laya 17/19 in 0.22s, on GLiNER 17/19 in
2.3s. Laya's misses keep two opinions (harmless: one extra search each); GLiNER's drop two real claims (they're
never checked), so Laya stays the default.

**Check against your own documents (in-domain).** `DocumentSearcher` searches given documents instead of the web:
the papers or reports a text was written from. It splits each
document into passages once (BM25 index) and returns the best ones as ordinary hits, so any judge takes them;
`NoCrawler` because there are no pages to fetch:

```python
from factassessor import DocumentSearcher, FactAssessor, NoCrawler

docs = [{"url": "poorter-2016", "title": "Biomass resilience of Neotropical secondary forests", "text": paper_text}]
fa = FactAssessor(searcher=DocumentSearcher(docs), crawler=NoCrawler())
```

On a scientific eval set (16 passage pairs whose source papers we could get as full text), FactAssessor in-domain
scored F1 0.84 on the original passages, at ~2s per passage for the models. The caveat: with evidence from the same
paper, Laya also let 38% of false claims through (claims wrong in one detail read as supported by an on-topic passage).

**Search without an API key:**

```python
from factassessor import DuckDuckGoSearcher, SearxngSearcher, Take, not_blocked

FactAssessor(searcher=DuckDuckGoSearcher() >> not_blocked() >> Take(5))                   # fact-assessor[ddg]
FactAssessor(searcher=SearxngSearcher("http://localhost:8080") >> not_blocked() >> Take(5))  # your own SearXNG
```

`SearxngSearcher` keeps at most 4 requests in flight (`max_concurrent`): SearXNG's upstream engines suspend a client
that bursts, and a text's 15 claims searching at once came back with almost no hits. The last claim of such a text
waits ~3s for a slot; Serper needs no such cap.

DuckDuckGo needs no setup but is slower (0.7–3.3s per query vs ~0.8s for Serper) and unofficial, so heavy use can
get rate-limited. Public SearXNG instances don't work for this (none of 25 healthy ones served JSON in our check);
run your own. A `settings.yml` that works:

```yaml
use_default_settings: true
server:
  secret_key: "<openssl rand -hex 32>"
  limiter: false        # local, single user (the limiter needs Valkey)
search:
  formats: [html, json] # SearxngSearcher reads JSON
engines:                # more general engines, so one throttling doesn't leave a single source
  - {name: google, disabled: false}
  - {name: bing, disabled: false}
  - {name: yahoo, disabled: false}
  - {name: qwant, disabled: false}
```

```bash
docker run -d --name searxng -p 127.0.0.1:8080:8080 -v "$PWD/searxng:/etc/searxng" searxng/searxng
```

With that, Google, Bing, and Yahoo answered most queries (30-47 results each); Brave and DuckDuckGo rate-limit or
show CAPTCHAs under load, which the others cover.

## Development

```bash
uv run pytest            # fast, offline: every network/model call is faked
```

Layout:

```
factassessor/
  assessor.py        FactAssessor: builds the default chain; stream / assess / assess_sync; lifecycle
  pipeline.py        Step, >>, Map, FlatMap, Filter, Take, Scan, TakeUntil, Predicate: the streaming runner
  atomizer.py        Atomizer (role), LLMAtomizer: text -> atoms
  decisions.py       DecisionRunner (role, a Protocol), DecisionRequest / DecisionResponse, Batcher,
                     SystemOneRunner (Jev over HTTP), LLMRunner (any chat model)
  laya.py            LayaRunner (default): in-process Laya, one model per device per process, micro-batching, batch cap
  gliner.py          GlinerRunner (optional extra): GLiNER2.5-decide via ONNX, one model per (model, variant) per process
  claim_filters/     atoms -> the factual claims
    _base.py         ClaimFilter (role)
    decision.py      DecisionClaimFilter (default): one decision per atom on any runner
  search/            query -> hits
    _base.py         Searcher (role), SearchType (general / science), not_blocked, hedging
    serper.py        SerperSearcher (Google web search, or Google Scholar)
    searxng.py       SearxngSearcher (self-hosted)
    duckduckgo.py    DuckDuckGoSearcher (no key)
    documents.py     DocumentSearcher (given documents: in-domain checks)
  crawlers/          url -> clean page text
    _base.py         Crawler (role), FallbackCrawler, NoCrawler
    browser.py       Crawl4AICrawler (headless browser, JavaScript)
    plain_http.py    HTTPXCrawler (plain HTTP, fast; HTML and PDF)
  resolvers.py       Resolver (role, a Protocol): url -> where to read it in full; Arxiv, OpenAlex, Composite
  rankers.py         Ranker (role, a Protocol): which chunks of a page the judge sees; BM25 (default), Embedding, Hybrid
  extract.py         document -> text: extract() (PDF or HTML, by type or bytes), pdf_text, html_text
  verify.py          Verify (per claim: snippets, resolve + crawl if needed, early exit), Policy (role), WeightedPolicy
  judges/            evidence -> stance per passage
    _base.py         Judge (role)
    decision.py      DecisionJudge (default): one decision per (claim, passage) on any runner
  kg.py              knowledge graph (kg.build, kg.to_mermaid), built on demand from a result
  passages.py        page cleaning, normalization, word windows and token-exact chunking, BM25
  utils.py           sentence spans; locate(query, source): the sentence of a text that best matches a piece of it
  schema.py          Atom, Evidence, AtomResult, CheckResult (fact_score computed from its atoms), stream events
```

Benchmarks: `scripts/compare_judges.py` compares judges on `data/judge_cases.json`, `scripts/compare_claim_filters.py`
claim filters on `data/claim_cases.json`, and `scripts/eval.py` is the end-to-end harness: synthetic texts
(`data/eval_texts.jsonl`, built from `data/fact_pairs.json`), evidence recorded once (DuckDuckGo by default), and
laya / gliner / llm runs on it, with a comparison and plots in `data/results/` (`uv run python scripts/eval.py --help`).
`scripts/eval_atoms.py` scores a labelled long-form set atom by atom with the SciELF paper's metric (per-answer
precision / recall / F1, macro-averaged; live run with cached evidence, then free replays of other judging settings),
which is how the comparison against FactReasoner in [issue #1](https://github.com/NISH1001/fact-assessor/issues/1) is
produced; that dataset is not in the repo.

## Roadmap

- **Streaming atomizer**: emit claims while the LLM is still writing them (the pipeline already streams from there
  on), so the first verdict arrives ~1s sooner.
- **Default crawler**: consider `FallbackCrawler(HTTPXCrawler(), Crawl4AICrawler())` plus the resolvers as the
  default once more end-to-end runs confirm it's faster.
- **Overfetch as the default**: `FactAssessor(overfetch=1.0)` keeps twice as many hits as pages and judges the
  first `top_k` that turn out readable (`Take` after the crawl, the rest cancelled), so paywalled or blocked hits
  don't leave a claim short of evidence; off (`0.0`) until the paper eval measures it.
- **Search queries for long claims**: a self-contained claim ("The 2025 Scientific Reports study of aboveground
  biomass across Connecticut forests used 67 explanatory variables") is right for the judge but a poor web query
  (engines match the generic words). A keyword step in front of the searcher, keeping numbers, names and technical
  terms, is the planned fix; the judge still sees the full claim.
- **Offline atomizer** (TODO): an `Atomizer` without an LLM, e.g. spaCy sentence/clause splitting plus coreference.
  An early prototype did this in ~5ms but kept multi-fact sentences together ("Nepal's 2017 earthquake of 7.8
  magnitude" is three facts), which let wrong details through; useful as a no-API fallback.
- **GLiNER judge accuracy**: tune the label descriptions / instruction on the judge benchmark; try it on CUDA.
- Per-detail checks in the judge (ask Laya about each date/number in a claim in the same forward pass), so a
  claim that is right except for one detail comes out refuted rather than contested.
- Component-level wrappers: retries and hedging (`Cache(step)` exists: a TTL cache around any step, used so a
  text's claims share one search for the text's source query).
- LLMAtomizer: keep opinions marked as opinions. Unwrapping hedges currently also strips "I think", so
  "I think pizza is the best food" becomes a plain claim; the default filter catches it, a custom one may not.

How the pieces fit, in detail (roles, data flow, concurrency, failure handling, timeouts):
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md). Step-by-step tour with example output: [docs/WALKTHROUGH.md](docs/WALKTHROUGH.md).
Design notes: [docs/design/streaming-pipeline.md](docs/design/streaming-pipeline.md). Why things are the way they
are (benchmarks, trade-offs): [docs/design/decisions.md](docs/design/decisions.md).

## Acknowledgements

Inspired by factuality pipelines such as [FActScore](https://github.com/shmsw25/FActScore) and SAFE; this project is an
independent, latency-focused implementation. Built on [Laya](https://github.com/NandhaKishorM/laya), [crawl4ai](https://github.com/unclecode/crawl4ai),
[pydantic-ai](https://ai.pydantic.dev), and [Serper](https://serper.dev).

## License

MIT
