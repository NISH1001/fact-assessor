# Architecture

How FactAssessor is put together: the roles, how data flows between them, what runs concurrently, and how
failures and deadlines are handled. For the why behind each choice (benchmarks, trade-offs) see
[design/decisions.md](design/decisions.md); for a guided tour with real output see [WALKTHROUGH.md](WALKTHROUGH.md).
The model layer (`DecisionRunner`) under the judge and the claim filter is designed in
[design/decision-runners.md](design/decision-runners.md), with the build order at its end.

## Goals

1. **Same job as a long-form fact checker** (atomize a text, check every claim against evidence, score it), at a
   fraction of the latency: seconds per passage, not minutes.
2. **Speed first.** Every step is async, every claim is checked at once, work streams through the pipeline instead
   of waiting for whole stages, and stopping early cancels what's no longer needed.
3. **Every step swappable** behind a small interface (a *role*). Concerns of one component (caching, retries,
   hedging, timeouts) live in that component or a wrapper around it, never in the orchestrator.
4. **Local models for classification.** Laya (a non-autoregressive decision model) filters claims and judges
   evidence in single forward passes; the only LLM call is the atomizer. The model sits behind one small role
   (`DecisionRunner`), so the same filter and judge run on Jev over HTTP, or another model, with one argument.

## The pipeline

```
text
 │
 ├─ 1. ATOMIZE    Atomizer: text -> atoms (atomic, self-contained claims)        LLMAtomizer: one LLM call
 │                (source_query=True: the same call also writes one search for the text's source document,
 │                 carried by every atom of the text)
 ├─ 2. FILTER     ClaimFilter: drop what isn't a factual claim; Take(n_atoms)     DecisionClaimFilter
 │
 └─ 3. VERIFY     per claim, every claim at once (Verify)
      │
      ├─ a. SEARCH     Searcher -> not_blocked -> Take(top_k)     hits {url, title, snippet}
      │                the claim, and the text's source query if it has one, concurrently; hits merged, the claim's first
      ├─ b. SNIPPETS   Judge the snippets; Policy.settled? ──────────────────────────────► g
      │                (not settled: gather more evidence, streamed)
      ├─ c. RESOLVE    Resolver (optional): hit url -> locations [direct PDF, free copies…, the hit's page]
      ├─ d. CRAWL      Crawler: each hit's locations in order, first readable wins -> page {url, title, text}
      ├─ e. PASSAGES   (inside the judge) chunk -> BM25 -> top passages per page
      ├─ f. JUDGE      Judge each page the moment it lands; running total of evidence
      │                Policy.settled? -> stop, cancel every resolve / crawl still running
      └─ g. POLICY     Policy.verdict(evidence) -> supported | refuted | contested | unverified
 │
 ▼
CheckResult: per claim {verdict, confidence, evidence}, fact_score, skipped atoms; knowledge graph on demand
```

`FactAssessor` (`factassessor/assessor.py`) wires the default chain: `atoms = atomizer >> claim_filter >>
Take(n_atoms)`, then `Verify(searcher, crawler, judge, policy, resolver=...)`. Everything it builds can be passed
in instead.

## Data

All passed between steps as plain dicts or pydantic models (`factassessor/schema.py`):

| Name | Shape | Made by | Used by |
|---|---|---|---|
| atom | `Atom(id, text, span, source_query)`: `span` is the sentence of the input text the claim was made from (found locally, `utils.locate`); `source_query` (optional) is one search for the document the text came from, the same for every atom of a text | atomizer | filter, verify |
| hit | `{"url", "title", "snippet"}` | searcher | snippet judge, resolver, crawler |
| page | `{"url", "title", "text"}`: clean plain text; `url` is always the **hit's** URL, even when the text came from a copy | crawler (via the read stage) | page judge |
| evidence | `Evidence(url, title, text, source="snippet" \| "page", label, prob)`: one judged passage | judge | policy |
| result | `AtomResult(atom, verdict, confidence, evidence, error)` | verify | output |
| check | `CheckResult(text, atoms, skipped, latency_ms)`, `fact_score` computed from the atoms | `FactAssessor` | caller, `kg.build` |
| events | `ClaimFound` (a claim is being checked), `ClaimVerified` (one verdict), `Done` (the result) | `FactAssessor.stream` | UIs |

`label` is `supports`, `refutes` or `not_enough_info`; `prob` is the judge's probability for that label.
`fact_score = supported / (supported + refuted + contested)`; unverified claims don't count either way.

## Roles

A role is a small interface. The built-in implementations are just the ones we measured best; anything with the
same method works. `Judge`, `Policy`, `Resolver`, `Ranker` and `DecisionRunner` are `Protocol`s: any object with
the method is one (`isinstance` checks structurally), no subclassing needed. `Atomizer`, `Searcher`, `Crawler`
and `ClaimFilter` are `Step` base classes, because what they give you is the streaming adapter (`__call__`, `>>`,
lifecycle) around your one method; no abstract base classes anywhere.

| Role | Interface | Implementations | File |
|---|---|---|---|
| `Atomizer` | `atomize(text) -> list[Atom]` | `LLMAtomizer` (pydantic-ai; falls back to sentences if the LLM fails) | `atomizer.py` |
| `ClaimFilter` | `score(atoms)` -> P(factual claim) | `DecisionClaimFilter` (one decision per atom, on any runner) | `claim_filters/` |
| `Searcher` | `search(query) -> list[hit]` | `SerperSearcher` (web or Google Scholar), `SearxngSearcher` (self-hosted; general or science engines), `DuckDuckGoSearcher`, `DocumentSearcher` (given documents: in-domain checks) | `search/` |
| `Resolver` (Protocol) | `resolve(url) -> list[str]` | `ArxivResolver`, `OpenAlexResolver`, `CompositeResolver` | `resolvers.py` |
| `Crawler` | `crawl(url) -> page \| None` | `Crawl4AICrawler` (browser), `HTTPXCrawler` (plain HTTP; HTML and PDF), `CascadedCrawler` (waterfall), `NoCrawler` | `crawlers/` |
| `Judge` (Protocol) | `judge(claim, docs) -> list[Evidence]` | `DecisionJudge` (one decision per (claim, passage), on any runner) | `judges/` |
| `Ranker` (Protocol) | `top(claim, chunks, k) -> list[str]` | `BM25Ranker` (default), `EmbeddingRanker` (model2vec), `HybridRanker` | `rankers.py` |
| `Policy` (Protocol) | `settled(evidence)`, `verdict(evidence)` | `WeightedPolicy` | `verify.py` |
| `DecisionRunner` (Protocol) | `predict(requests) -> responses` (a `state` and typed `questions` in, probabilities per option out); `batch_size` | `LayaRunner` (default; in-process), `SystemOneRunner` (Jev's System One protocol over HTTP: OpenRouter or a `laya.serve` server), `LLMRunner` (any pydantic-ai chat model), `GlinerRunner` (ONNX, CPU) | `decisions.py`, `laya.py`, `gliner.py` |

The runner is the layer below the roles: `DecisionClaimFilter` and `DecisionJudge` are written once on top of it
and never name a model; a runner never sees claims, pages or evidence, only requests. Batching is the runner's
job (`decisions.Batcher`): the requests of every caller in flight are merged, cut at `batch_size` and capped in
flight, so 100 requests become a few passes or calls. `FactAssessor()` gives the filter and the judge one shared
`LayaRunner`.

Shared helpers, not roles: `extract.py` (document bytes -> text), `passages.py` (cleaning, normalization,
word windows, chunking, BM25), `utils.py` (sentence spans; `locate(query, source)`, the sentence of a text that
best matches a piece of it, how the atomizer finds where a claim came from), `kg.py` (knowledge graph).

**Who knows what.** Resolvers know about *documents* (DOIs, arXiv ids, where free copies live) and never fetch
them. Crawlers know about *fetching one URL* (HTTP or a browser) and turn what they get into text with the shared
`extract()`; they never know a URL was resolved. Judges know about *stance* and never fetch anything.

## The streaming engine

`factassessor/pipeline.py`. A `Step` turns an async stream of items into an async stream of items; `a >> b` chains
them. In a chain, a plain function is a `Map` and a `Predicate` (a condition) is a `Filter`.

| Step | Does |
|---|---|
| `Map(fn)` | each item -> one item (dropped if `fn` returns None), **concurrently**, emitted as each finishes |
| `FlatMap(fn)` | each item -> 0..n items, concurrently |
| `Filter(pred)` | keep items where `pred` holds; conditions combine with `&`, `\|`, `~` |
| `Scan(fn, init)` | running total: emits `state = fn(state, item)` after every item |
| `Take(n)`, `TakeUntil(pred)` | stop after n items / at the first item where `pred` holds, **cancelling everything upstream** |
| `Cache(step, ttl, maxsize)` | a TTL cache (cachetools) around any step or sub-chain: the same input is processed once per `ttl`, concurrent duplicates share the one in flight, failures aren't kept; `x >> Cache(y >> z) >> a` |
| `once(x)`, `collect(stream)`, `last(stream)` | a one-item source; all items; the final item |

Two properties make the whole pipeline fast:

- **Nothing waits for a whole stage.** Claim 1 can be judging its pages while claim 5 is still searching.
- **Stopping cancels upstream.** When `TakeUntil(policy.settled)` fires, every resolve, crawl and download still
  running for that claim is cancelled.

**Lifecycle.** `aload()` / `aclose()` on any step walk everything reachable from it (`Step.parts()`: its attributes
that are steps or have `aload` / `aclose`) and start or stop each once: the browser, HTTP pools, the OpenAlex
client, the model runners. `FactAssessor` exposes them as `aload` / `aclose` and `async with`.

## Verify, step by step

`factassessor/verify.py`, per claim:

**a. Search.** The claim text is the query. `not_blocked()` drops social media, forums and video sites
(`BLOCKED_DOMAINS`); the searcher over-fetches (`num = 2 * top_k`) so blocked hits don't leave it short. With
Serper the same hosts are also excluded in the query itself (`-site:` operators, as many as fit under Google's
32-word cap, leakiest first), so Google fills those slots with usable sources instead of hits we pay for and drop;
`not_blocked()` stays as the guarantee.
When the atom carries a `source_query` (the atomizer's search for the document the text came from), it is searched
at the same time and its hits are appended after the claim's own, each URL once. A claim about a detail inside a
paper rarely finds the paper by itself (about a quarter of claims on scientific passages); the whole text usually
does. The source query is identical for every claim of a text, so `FactAssessor` wraps the searcher in `Cache`: one
real search, the other claims wait for it.

**b. Snippets first.** The judge gets the snippets as they are. If `Policy.settled` holds (2+ passages agree at
≥ `early_exit` 0.9 and none disagrees at ≥ `strong` 0.7), the claim is done: no resolving, no crawling. General
knowledge claims usually settle here; specific scientific ones rarely do.

**c. Resolve** (only with `resolver=`). Every hit is resolved at once (`Map`). `resolvers.locations(url, candidates)`
orders where to read it:

1. the hit itself, if it's a direct PDF link (exactly the document search matched);
2. the resolvers' candidates (arXiv HTML then PDF; OpenAlex's open-access copies, PDFs before landing pages;
   doi.org links left out, since they only redirect back to the publisher);
3. the hit's own page (an ordinary page is only this).

`CompositeResolver` asks all its resolvers at once and joins their lists, so a later resolver's copies are still
tried when an earlier one's are all blocked. An ordinary page costs nothing: resolvers answer `[]` without a request.

**d. Crawl (the read stage).** Every hit is read at once; within one hit, `read_first` tries its locations **in
order** and keeps the first readable one ("first that works", which is also polite to hosts). A copy claims to be
the full document, so it needs `min_copy_words` (300; bot-check pages are ~180); the hit's own page is taken as
the crawler returns it. A hit has no clock of its own: each fetch has the crawler's limit, and the claim deadline
caps the rest (a per-hit 8s clock used to count the wait for a crawler connection too, and under load it cancelled
crawls before they started; see design/decisions.md). The page comes back under the hit's URL.
Without a resolver, the crawler just crawls each hit's URL.

**e. Passages** (inside the judge, through its `Ranker`: `BM25Ranker` by default; `HybridRanker` mixes in static
embeddings to catch paraphrase). Claim and page text are put in one Unicode form (`normalize_text`: NFKC,
minus signs as `-`, so `ha⁻¹` and `ha−1` match). The page is cut into 90-word windows (about 128 Laya tokens, the
size that measured best; 25% overlap; `DecisionJudge(chunk=...)` swaps the cutter), the text's own words, so every
runner sees the same passages, and the ranker picks the top `passages_per_page` (1 by default; 3 measured +0.10 F1
on the paper eval).

**f. Judge.** Each page is judged the moment it lands; `Scan` keeps the running total of evidence, and
`TakeUntil(policy.settled)` stops the claim as soon as it's settled.

**g. Policy** (`WeightedPolicy`). Only strong evidence counts (prob ≥ `strong`, 0.7, and not `not_enough_info`).
With `S` and `R` the summed probabilities of strong supports and refutations:

| Condition | Verdict | Confidence |
|---|---|---|
| no strong evidence | `unverified` | 0 |
| `S >= 2R` | `supported` | max support prob |
| `R >= 2S` | `refuted` | max refute prob |
| otherwise | `contested` | `max(S, R) / (S + R)` |

The 2x margin means one stray "refutation" (a related but different fact) doesn't flip several supports.

## Concurrency

| Level | How | Limit |
|---|---|---|
| claims | `Verify` is a `Map` over atoms | the judge's `concurrency`, taken from its runner (none for Laya, Jev and LLMs; 3 for GLiNER); `max_concurrent_claims` overrides. A claim's timeout starts when it gets its slot. |
| hits of a claim | resolve and read are each a `Map` | all hits at once (`top_k`, plus the source query's) |
| the same query from several claims | `Cache` around the searcher | one real search per query per 10 minutes; the rest share it |
| resolvers of a hit | `CompositeResolver`: `asyncio.gather` | all at once |
| locations of a hit | `read_first`: one at a time, in order | deliberate: first that works |
| `LayaRunner` | one model and one thread per device per process; every request arriving within 5ms (any claim, page, or component) is merged into one pass, and everything arriving during a pass forms the next | 32 rows per forward pass |
| `SystemOneRunner` (Jev) | one claim's passages per HTTP call: one state with the passages as a list, one question per passage, up to 40 (`packing=DecisionPacking.ALL` fills calls with every caller's requests in flight instead, `DecisionPacking.NONE` sends each alone); 429s and 5xx retried with backoff | 16 calls in flight |
| SearXNG | `SearxngSearcher(max_concurrent=4)`: its upstream engines suspend a client that bursts | 4 requests in flight |
| browser | one shared headless browser | `max_concurrent_crawls` (10) pages at once, all claims |
| plain HTTP | one shared connection pool | 20 at once |
| OpenAlex | one client | 5 at once (OpenAlex asks for ≤ 10/s) |
| PDF parsing | `PDF_WORKERS` (2) worker processes | one PDF per worker at a time |
| GLiNER (optional) | one model, 2 worker threads | |

## Deadlines

| What | Default | On expiry |
|---|---|---|
| one claim | `timeout` 15s | decided on the evidence judged so far (snippets, pages that landed), `error="timeout"`; crawls still queued or running are cancelled |
| search request | 5s (Serper, SearXNG general); slow requests hedged after 1.2s | no hits |
| browser page | `crawl_timeout` 2.5s | page dropped |
| HTTP page | 2.5s total (1s to connect); 8s once the response is a PDF | page dropped |
| OpenAlex lookup | 5s | no candidates |
| PDF parse | 30s | worker killed and replaced; that PDF dropped |

## Failure handling

Nothing in a claim's path raises to the caller. Each failure degrades to less evidence, never to a crash:

| Failure | Where it's absorbed | Result |
|---|---|---|
| atomizer LLM error | `LLMAtomizer` | the text split into sentences instead |
| search error | `Verify.verify` | claim `unverified`, `error` set |
| resolver error | `CompositeResolver` / `Verify._resolve` | no candidates; the hit's own page is still read |
| 403, 404, bot-check page, timeout on a location | `read_first` / crawler | next location |
| no location readable | read stage | the hit adds no page; snippets still count |
| PDFium crash (segfault) | PDF worker process | that PDF lost; worker replaced |
| claim over its deadline | `Verify.verify` | verdict on the evidence so far |

**Stub detection.** A page is not a document when it's too short: `HTTPXCrawler` needs `min_words` (100; on 2,748
crawled pages, those under 100 words were login walls, "Loading…" shells and browser checks), and a resolved copy
needs 300. Words, not characters: links and markup leftovers inflate character counts.

## Reading documents

`factassessor/extract.py`, shared by every crawler and the eval:

- `extract(body, content_type)` decides the format from the content type or the bytes themselves (`%PDF`,
  `<html`; repositories often label PDFs `application/octet-stream`), and returns `(title, clean text)` or None.
- **PDF**: PDFium via pypdfium2 (22ms per paper, no words broken across lines), in worker processes.
- **HTML**: BeautifulSoup + lxml without scripts, styles, menus, headers, footers, forms; MathML formulas become
  their TeX once (arXiv's HTML papers).
- `passages.clean_text` then removes markup leftovers, citations, table pipes and navigation lines.

## Configuration

`FactAssessor()` defaults target general text; a scientific setup adds scholarly search and the resolvers:

| | Default | Scientific / low-cost eval setup |
|---|---|---|
| atomizer | `openai:gpt-5.6-luna` (reasoning as low as the model allows), no source query | `openai:gpt-6-luna` (same atoms, half the price; gpt-5-nano can't atomize), `source_query=True` |
| search | Serper (web) | self-hosted SearXNG |
| resolver | none | `CompositeResolver(ArxivResolver(), OpenAlexResolver())` |
| crawler | `Crawl4AICrawler(timeout=2.5)` | `CascadedCrawler(HTTPXCrawler(), Crawl4AICrawler())` |
| judge / policy | `DecisionJudge()`, `WeightedPolicy(strong=0.7, early_exit=0.9)` | same |

```python
from factassessor import (ArxivResolver, CompositeResolver, Crawl4AICrawler, FactAssessor, CascadedCrawler,
                          HTTPXCrawler, OpenAlexResolver, SearxngSearcher, Take, not_blocked)

fa = FactAssessor(
    atomizer_model="openai:gpt-6-luna",
    source_query=True,
    searcher=SearxngSearcher("http://localhost:8080", num=10) >> not_blocked() >> Take(5),
    resolver=CompositeResolver(ArxivResolver(), OpenAlexResolver()),
    crawler=CascadedCrawler(HTTPXCrawler(), Crawl4AICrawler()),
)
```

## Extending

- **A source of free copies**: a class with `async resolve(url) -> list[str]` (return `[]` for URLs that aren't
  yours, never raise), added to `CompositeResolver`. Give it `aload` / `aclose` if it holds a client.
- **A way to fetch**: subclass `Crawler`, implement `crawl(url)`, use `extract()` for the bytes, return None on
  failure. Combine with `CascadedCrawler`.
- **A search backend**: subclass `Searcher`, implement `search(query)`; chain `>> not_blocked() >> Take(k)`.
- **A passage ranker**: a class with `top(claim, chunks, k)` (best first; `k=None` for all, ranked), passed as
  `ranker=` to any judge. `HybridRanker(alpha=)` already mixes BM25 with embeddings.
- **A judge**: subclass `Judge`, implement `judge(claim, docs)` (snippets have `snippet`, pages have `text`); set
  `concurrency` if it can't take every claim at once.
- **Caching, retries, hedging**: a wrapper with the same interface around the component, not code in `Verify`.
  `Cache(step)` is the ready-made one for any step; `hedged` in `search/_base.py` for slow requests.

## Known limitations

- **Retrieval is the bottleneck on scientific claims.** When search doesn't return the source paper, most claims
  stay unverified. Self-contained claims and the text's source query help (see Search above), but a passage that
  never names its paper can still fail to find it, and claims about details inside a paper never do on their own.
- **One passage per page** (BM25 top 1): the matching sentence is often the second or third best.
- **Same paper, several hits**: copies of one source count as independent evidence (no dedupe yet).
- **Landing pages** (repository portals) pass the word minimum with only an abstract.
- **Hosts that block bots** (Cloudflare, some publishers) leave only the snippet.
- **Single-detail edits** ("20 flights" when it was 72) tend to come out contested or unverified, not refuted.
- **Long self-contained claims search badly.** The claim text is the web query; a claim that names its study in
  full makes engines match the generic words ("2025", "study", "default"). A keyword step before the searcher is the
  planned fix (the judge keeps the full claim). Shorter claims, like the eval set's own atoms, don't suffer from it.
- **Overfetch is off by default** (`FactAssessor(overfetch=0.0)`): a paywalled or blocked hit is lost rather than
  replaced by the next readable one. With `overfetch=1.0`, twice as many hits are kept and the crawl stage judges
  the first `top_k` that turn out readable (`Verify(pages_per_claim=)`: `Take` after the crawl, cancelling the rest),
  at the cost of up to 2x the crawl requests. Measured on the paper eval before it becomes the default.
