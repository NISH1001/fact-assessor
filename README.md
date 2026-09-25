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
     └─ LayaCheckworthy ─ Laya, local: drop opinions, greetings, questions
         └─ search ───── Serper (Google), every claim in parallel; slow requests are hedged
             └─ judge ── Laya, local: does each snippet support / refute the claim?
                 ├─ settled → done (no crawling)
                 └─ not yet → crawl pages (crawl4ai) in parallel, judge each page as it lands,
                              stop as soon as the evidence settles the claim
 └─ aggregate ─────────── verdict per claim → fact score   (knowledge graph: kg.build(result), on demand)
```

| Step | What does it | Where it runs |
|---|---|---|
| Atomize + decontextualize | `LLMAtomizer`: [pydantic-ai](https://ai.pydantic.dev) → `openai:gpt-5.6-luna` (reasoning off) | API, ~2s |
| Check-worthiness filter | `LayaCheckworthy`: [Laya](https://github.com/NandhaKishorM/laya) `choice` decision | local (MPS / CUDA / CPU) |
| Search | [Serper](https://serper.dev), social media and video sites filtered out | API, ~1s |
| Crawl | [crawl4ai](https://github.com/unclecode/crawl4ai), one shared headless browser, cleaned plain text | network, ~1s/page |
| Evidence judge | `LayaJudge`: pages chunked with Laya's own tokenizer, best BM25 passage per page | local |
| Verdicts, score, graph | strong evidence weighed per side: `supported` / `refuted` / `contested` / `unverified` | local |

Every box above is a swappable, chainable step (`LLMAtomizer() >> LayaCheckworthy() >> Take(8)`), and the whole
thing streams: a claim starts searching the moment it's found, each page is judged the moment its crawl lands,
and results come out as each claim settles. See [Compose your own pipeline](#compose-your-own-pipeline).

Laya is a non-autoregressive decision model (a Jev-style encoder that classifies instead of generating), so the
filter and the judge are single forward passes. Every Laya request that arrives within a few milliseconds, from
any claim or page, is merged into one batch by `LayaRunner`.

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
uv run marimo edit --no-sandbox notebooks/fa_walk.py    # guided walk: building blocks -> custom pipeline
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
│   ├── atom        Atom: text (self-contained claim), span (char offsets into your input), checkworthiness
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
| `checkworthy_threshold` | 0.4 | min P(factual claim) to keep an atom |
| `early_exit_conf` | 0.9 | 2+ passages this sure (and none against) settle a claim without more crawling |
| `strong_evidence` | 0.7 | min probability for a passage to count toward a verdict |
| `crawl_timeout` | 2.5 | seconds per page (a hard limit; a page that takes longer is dropped and the claim goes on without it) |
| `search_hedge_after` | 1.2 | if a search hasn't answered by then, send the same request again and use whichever reply comes first (fixes Serper's occasional 3s+ outliers; only slow searches cost a second credit; `None` turns it off) |
| `blocked_domains` | social + video | hosts never used as evidence (subdomains included); `()` to allow all |
| `timeout` | 15 | per-claim deadline; a claim still running then comes back `unverified` |

### Compose your own pipeline

Every component is a step, and steps chain with `>>` ("then"). In a chain, a **`Pred`** (a condition) filters
whatever flows at that point (atoms after the atomizer, search hits after the searcher) and a **plain function**
transforms it. Conditions combine with `&` (and), `|` (or), `~` (not).

```python
from urllib.parse import urlparse
from factassessor import Crawl4AICrawler, FactAssessor, LayaCheckworthy, LayaRunner, LLMAtomizer, Pred, SerperSearcher, Take, not_blocked

official = Pred(lambda hit: urlparse(hit["url"]).netloc.endswith((".gov", ".edu")))   # a condition
long_enough = Pred(lambda atom: len(atom.text) > 15)

laya = LayaRunner()                                  # one Laya model, shared by the filter and the judge
fa = FactAssessor(
    laya=laya,
    atomizer=LLMAtomizer() >> LayaCheckworthy(laya, threshold=0.5) >> long_enough >> Take(10),
    searcher=SerperSearcher(num=20) >> (not_blocked() & official) >> Take(5),
    crawler=Crawl4AICrawler(timeout=4.0),
)
result = await fa.assess("NASA was founded in 1958. The Eiffel Tower is 500 meters tall.")
# supported   NASA was founded in 1958.              (nasa.gov, eisenhowerlibrary.gov)
# unverified  The Eiffel Tower is 500 meters tall.   (no .gov/.edu sources)
```

What you can pass. Each component has a **role** (a base type): subclass it and implement one method, and
streaming, concurrency, and chaining come for free.

| Argument | Role | You implement | Default |
|---|---|---|---|
| `atomizer=` | `Atomizer` (or a chain starting with one) | `atomize(text) -> list[Atom]` | `LLMAtomizer() >> LayaCheckworthy(laya) >> Take(n_atoms)` |
| `searcher=` | `Searcher` (or a chain) | `search(query) -> list[hit]`, hits `{"url", "title", "snippet"}` | `SerperSearcher() >> not_blocked() >> Take(top_k)` |
| `crawler=` | `Crawler` | `crawl(url) -> page or None`, pages `{"url", "title", "text"}` | `Crawl4AICrawler(timeout=2.5)`; also `HTTPXCrawler`, `FallbackCrawler` |
| `judge=` | `Judge` | `judge(claim, docs) -> list[Evidence]` | `LayaJudge(laya)` |
| `policy=` | `Policy` | `settled(evidence)`, `verdict(evidence) -> (verdict, confidence)` | `WeightedPolicy()` |
| `laya=` | `LayaRunner` | | one per assessor, shared by filter and judge |

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
| `Filter(pred)` or a `Pred` in a chain | keep items where the condition holds |
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

End to end the gain is smaller, since many claims settle on search snippets without crawling (Nepal example:
6.1s → 5.2s median; mixed example: about the same). `HTTPXCrawler` has a 1s connect timeout plus a hard 2.5s total
deadline (httpx's own timeouts are per phase), reads only HTML, and stops at 3 MB.

**GLiNER2.5-decide judge** ([GLiNER2.5-decide](https://fastino.ai/blog/gliner-2-5-decide-open-weight-decision-model),
another Jev/Laya-style decision model, as ONNX from
[nishparadox/gliner2.5-decide-onnx](https://huggingface.co/nishparadox/gliner2.5-decide-onnx); CPU, no torch):

```python
from factassessor.gliner import GlinerJudge     # needs fact-assessor[gliner]; downloads ~1.75 GB on first use

fa = FactAssessor(judge=GlinerJudge())            # variant="int8" is 2x faster but much less accurate
```

| Judge (15-case benchmark, M-series Mac) | Correct | Time for 15 pairs |
|---|---|---|
| `LayaJudge` (default, MPS) | 13/15 | 0.27s |
| `GlinerJudge` fp32 (CPU) | 12/15 | 2.2s |
| `GlinerJudge` int8 (CPU) | 7/15 | 1.0s |

GLiNER's misses were all false "supports" (it let "NASA was founded in 1972" through), so Laya stays the default.
Reproduce with `uv run --extra gliner python benchmarks/compare_judges.py`.

**Search without an API key:**

```python
from factassessor import DuckDuckGoSearcher, SearxngSearcher, Take, not_blocked

FactAssessor(searcher=DuckDuckGoSearcher() >> not_blocked() >> Take(5))                   # fact-assessor[ddg]
FactAssessor(searcher=SearxngSearcher("http://localhost:8888") >> not_blocked() >> Take(5))  # your own SearXNG
```

DuckDuckGo needs no setup but is slower (0.7–3.3s per query vs ~0.8s for Serper) and unofficial, so heavy use can
get rate-limited. Public SearXNG instances don't work for this (none of 25 healthy ones served JSON in our check);
run your own with JSON enabled (`docker run -p 8888:8080 searxng/searxng`, then add `json` to `search.formats`).

## Development

```bash
uv run pytest            # fast, offline: every network/model call is faked
```

Layout:

```
factassessor/
  assessor.py        FactAssessor: builds the default chain; stream / assess / assess_sync; lifecycle
  pipeline.py        Step, >>, Map, FlatMap, Filter, Take, Scan, TakeUntil, Pred: the streaming runner
  atomizer.py        Atomizer (role), LLMAtomizer: text -> atoms
  atom_filter.py     LayaCheckworthy: atoms -> factual atoms
  search.py          Searcher (role), SerperSearcher, DuckDuckGoSearcher, SearxngSearcher, not_blocked, hedging
  crawl.py           Crawler (role), Crawl4AICrawler, HTTPXCrawler, FallbackCrawler: url -> clean pages
  verify.py          Verify (per claim: snippets, crawl if needed, early exit), Policy (role), WeightedPolicy
  evidence_judge.py  Judge (role), LayaJudge
  gliner.py          GlinerJudge: GLiNER2.5-decide via ONNX (optional extra)
  kg.py              knowledge graph (kg.build, kg.to_mermaid), built on demand from a result
  laya.py            LayaRunner: shared model, micro-batching, batch cap
  passages.py        page cleaning, token-exact chunking, BM25
  schema.py          Atom, Evidence, AtomResult, CheckResult (fact_score computed from its atoms), stream events
```

Benchmarks: `benchmarks/compare_judges.py` compares judges on `benchmarks/judge_cases.py`.

## Roadmap

- **Streaming atomizer**: emit claims while the LLM is still writing them (the pipeline already streams from there
  on), so the first verdict arrives ~1s sooner.
- **Default crawler**: consider `FallbackCrawler(HTTPXCrawler(), Crawl4AICrawler())` as the default once more
  end-to-end runs confirm it's faster.
- **Offline atomizer** (TODO): an `Atomizer` without an LLM, e.g. spaCy sentence/clause splitting plus coreference.
  An early prototype did this in ~5ms but kept multi-fact sentences together ("Nepal's 2017 earthquake of 7.8
  magnitude" is three facts), which let wrong details through; useful as a no-API fallback.
- **GLiNER judge accuracy**: tune the label descriptions / instruction on the judge benchmark; try it on CUDA.
- Per-detail checks in the judge (ask Laya about each date/number in a claim in the same forward pass), so a
  claim that is right except for one detail comes out refuted rather than contested.
- Component-level wrappers: caching, retries, timeouts.
- LLMAtomizer: keep opinions marked as opinions. Unwrapping hedges currently also strips "I think", so
  "I think pizza is the best food" becomes a plain claim; the default filter catches it, a custom one may not.

Design notes: [docs/design/streaming-pipeline.md](docs/design/streaming-pipeline.md). Why things are the way they
are (benchmarks, trade-offs): [docs/design/decisions.md](docs/design/decisions.md).

## Acknowledgements

Inspired by factuality pipelines such as [FActScore](https://github.com/shmsw25/FActScore), SAFE, and
[IBM's FactReasoner](https://github.com/IBM/FactReasoner); this project is an independent, latency-focused
implementation. Built on [Laya](https://github.com/NandhaKishorM/laya), [crawl4ai](https://github.com/unclecode/crawl4ai),
[pydantic-ai](https://ai.pydantic.dev), and [Serper](https://serper.dev).

## License

MIT
