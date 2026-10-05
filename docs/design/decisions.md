# Decisions and measurements

What we tried while building the prototype, what we measured, and what we picked. Numbers are from an M-series
Mac (MPS), Sept 2026. Revisit a decision when its evidence changes.

## Atomization

- **One LLM call does atomize + decontextualize** (`Atomizer`). Sentence/clause splitting (spaCy) kept compound
  claims together: "Nepal's earthquake in 2017 of 7.8 magnitude caused massive damage" is three facts, and the
  judge passed it as *supported* because 2 of 3 were right. Implicit context was also lost ("Total lives lost were
  1 million" got searched on its own and matched WWII pages).
- Prompt rules that mattered: one fact per claim; self-contained; unwrap hedges ("It was believed that X" → X);
  keep values exactly as written (we check, not correct); **name the subject as specifically as the text allows
  (a study by venue, authors, or topic; a place or object by name) and repeat that identifying context in every
  claim that needs it, but never a date or figure that has its own claim** (otherwise "in 2017" leaked into every
  atom and one wrong year failed all of them). Bare references ("the study used 67 variables") can't be searched:
  on the scientific eval, 80% of such atoms never found their source paper. The prompt is rules only, no worked
  examples. On 8 scientific answers, gpt-6-luna and gpt-5.6-luna give the same atoms at reasoning none, low, and
  medium (15-17 per answer vs FactReasoner's 14.4); reasoning only adds time and cost, and at low/medium both
  models once split "67 variables from LiDAR, NAIP and Sentinel-2" into three claims each attributing all 67 to one
  source. gpt-6-luna (none): ~6s per 160-word answer, $0.0003. gpt-5-nano returns whole sentences at minimal
  reasoning and bare "the researchers" claims at low/medium (33s): unusable as the atomizer.
- **The atomizer also writes the text's source query** (`LLMAtomizer(source_query=True)`, `Atom.source_query`),
  searched next to each claim and shared by the text's claims (`Cache` around the searcher: one real search).
  A claim about a detail inside a paper rarely finds the paper: on 8 scientific passages, 24% of claims did (SearXNG
  top 5; FactReasoner's own revised atoms: 21%), and two passages 0 of 18 and 0 of 21. One query per passage found
  the paper for 5 of 8 (Serper: 7 of 8), but only when written like a title (topic, method, place, instruments):
  a query of quoted distinctive numbers found 0 of 8 on SearXNG. Same LLM call as atomization: no added latency.
- Model: **`gpt-5.6-luna`, reasoning `none`** (~2s). Benchmarked on 3 decontextualization cases:

## Passage selection (the paper eval: 100 answers, 1,667 labelled atoms, identical cached evidence)

- **3 passages per page instead of 1: combined F1 0.513 -> 0.618, recall 0.391 -> 0.508, precision unchanged
  (0.897 -> 0.902).** At top-1, 73 numeric true atoms had the fact on the page but in a chunk BM25 didn't rank
  first; top-3 recovers them for ~3s more Laya time per 15-claim text (replay: 5s -> 8s per answer).
- **The hybrid ranker (BM25 + model2vec `potion-base-8M` static embeddings) does not help at top-3**: alpha 0.5
  F1 0.615 / recall 0.504; alpha 0.3 (more embedding weight) 0.606 / 0.493; BM25 0.618 / 0.508. Whenever the fact is
  on the page, BM25's top 3 already holds it; the remaining misses are qualitative atoms with no literal statement
  anywhere and passages Laya rejects. `BM25Ranker` stays the default; `HybridRanker` remains available (optional
  extra `embed`, ~30 MB, sub-millisecond per claim once a page's chunks are cached).
- **Judge threshold `strong` 0.7 -> 0.5 (top-3): combined F1 0.618 -> 0.643, but the corrupted split is flat
  (0.589 -> 0.588) and 5 more false atoms pass (204 -> 199 of 249 caught).** The gain is on the all-true originals:
  saying "supported" more readily, not catching more. 0.7 stays the default.
- Laya's batch merge window (5ms -> 50ms) changed nothing: identical verdicts, 1,190s vs 1,235s for the replay.
  Replaying 4 answers at once was no faster than one at a time (1,293s vs 1,303s): Laya is saturated either way.

  | Model | Warm latency | Quality |
  |---|---|---|
  | gpt-5.6-luna (reasoning none) | ~1.7s | all correct |
  | gpt-6-luna (reasoning none) | ~2.0s | all correct |
  | gpt-5.4-mini (reasoning none) | ~1.0s | garbled a sentence in one run |
  | gpt-5.4-nano / gpt-4.1-nano | ~1.5s (4.1 spikes to 7s) | missed references |
  | gpt-5-nano (minimal) | ~1.0s | unusable |

  Default reasoning made the Luna models slower *and* worse (it rewrote a greeting into a tautological "claim").
- The model returns claims only. It used to quote each claim's source words as well (spans by exact match): that
  doubled the output, and output is what the call's time goes on (176-word answer, reasoning off: 707 tokens in,
  742-1,282 out, 8.8-10.7s; 0 reasoning tokens). Dropped 2026-09-30; spans are now the best-matching sentence by
  BM25 over the text's sentences (`utils.locate`), which is what the UI underlines and the graph links anyway.
- On LLM failure: fall back to sentence atoms rather than failing the check.

## Check-worthiness filter (Laya)

- Laya's `noul` (yes/no) head was near-random for this (6–8/12). A **`choice`** question over
  factual_claim / opinion / question_or_request / social on the `english` checkpoint: **17/19**.
- Keep if P(factual_claim) ≥ **0.4**, not 0.5: dropping a real claim (never checked) costs more than keeping an
  opinion (one wasted search). Plain facts about little-known subjects score near the cutoff (~0.45–0.6).
- ~100ms for a whole batch once warm.

## Evidence judge (Laya)

- Question wording and **state field order** matter: claim-first topped out at 9/15; **evidence first, then claim**
  with "According to `evidence`, is `claim` true or false?" → **13/15**.
- For reference, DeBERTa-v3-base-mnli-fever-anli got 14/16 (135ms/16 pairs). We stayed with Laya by choice (one
  local model family for filter + judge); the judge is swappable if that changes.
- **1 passage × 128 tokens per page** (best BM25 chunk): same accuracy as 3 × ~290 tokens (10/12 gold claims on
  frozen evidence) at 1/3 the judge time (3.2s vs 9.9s for 12 claims, no early exit).
- On MPS the cost is ~pairs × tokens (≈42ms/pair at 290 tokens, 24 at 128, 17 at 64). **fp16/bf16: no gain.**
  Merging micro-batches barely matters (compute-bound). Fewer/shorter pairs is the lever.
- Chunking used Laya's own tokenizer to an exact budget (max_len − head_max_len − claim − margin) until the
  decision-runner refactor (2026-09-30); `DecisionJudge` now cuts 90-word windows (about 128 tokens, well inside
  Laya's ~300-token state budget), the same passages for every runner, and takes a token-exact cutter as `chunk=`.
- Known gap: a claim right except for one detail ("occurred in 2017", real 2015) often comes out *contested* —
  pages mentioning the event in 2017 (anniversaries) read as support. Planned fix: per-detail Laya questions
  (dates/numbers/entities) in the same forward pass.

- **Laya batching under load** (MPS): capping each forward pass at `batch_size=32` rows costs no speed
  (76 pairs: 1,854ms capped or not) and holds memory flat (400 pairs: 7.1 GB → 2.0 GB, and slightly faster).
  Below 16 it slows down. Rows in a batch are padded to the longest, so a pass mixing ~20-token snippets with
  ~128-token passages took 946ms vs 714ms as two passes: `LayaRunner` sorts merged requests by length.
- The 512-token limit is per row, not per batch: each (evidence, claim, question) is its own ≤512-token row,
  and a batch stacks rows (`[n, ≤512]`), so batching many pairs never hits the context limit, only memory.

## Verdicts

- Strong evidence = prob ≥ 0.7, not "not_enough_info". A side wins with ≥ 2× the other side's summed weight;
  otherwise *contested*; no strong evidence → *unverified*. One strong refutation shouldn't flip several supports:
  the judge sometimes calls a related-but-different fact (the 1911 Chemistry prize vs a 1903 Physics claim) a
  refutation.
- Early exit: 2+ passages at ≥ 0.9 on one side and none strongly against.
- Fact score = supported / (supported + refuted + contested).

## Web evidence

- Crawled markdown was ~2/3 links; crawl4ai `ignore_links` + our `clean_text` (citations, emphasis, tables, menu
  bullets) took Wikipedia's Marie Curie page 276k → 84k chars and 441 → 101 chunks.
- Blocked after search, before crawling: facebook, instagram, threads, tiktok, pinterest, twitter/x, reddit, quora
  (reposts, opinions, comments; crawl badly) and youtube (no text). LinkedIn kept (primary source for people/orgs).
  Earlier versions kept twitter/x and reddit; blocked since 2026-09-25. LinkedIn and tumblr blocked since
  2026-09-30 (FactReasoner's list).
- **Pages and OpenAlex lookups cached in memory** (2026-10-01, `utils.cache` on `HTTPXCrawler.crawl`,
  `Crawl4AICrawler.crawl`, `OpenAlexResolver.resolve`; 2,048 pages / 4,096 lookups, 10 minutes, per instance,
  concurrent callers share one call). Half of all hit URLs repeat across a text's claims (they share the source
  query: the same papers), and each repeat used to be fetched, parsed and sometimes rendered again. 10 answers at
  once on cached hits: 162s -> 63s wall clock, ~150s -> 40-63s per answer, crawl busy time 144-158s -> 36-60s.
- **No per-hit read deadline** (2026-09-30). `Verify` used to give one hit 8s for all its locations together,
  measured from the moment the hit was handed to the crawler. The crawler caps connections (20 HTTP, 10 browser),
  so under load that clock counted the wait for a connection: the first live eval at 6 answers in parallel
  (~2,000 hits competing) had 93% of crawls expire unstarted (median crawl 8.3s, against 5.8s and 39% at the limit
  when sequential), only 25% of claims judged any page (89% sequential), and 571 of 1,667 claims came back
  unverified on snippets alone although the eval's after-the-fact cache held a readable page for every one of
  them: F1 0.560 for a run whose replay on the same evidence scores higher. Each fetch already has the crawler's
  limit (2.5s HTML, 8s PDF), a hit has a handful of locations, and the claim deadline caps the rest, so the per-hit
  clock guarded nothing that wasn't guarded and was the only clock measuring the queue. Removed: under load a hit
  waits its turn; latency stretches, evidence isn't dropped.
- **Serper: the blocked hosts are excluded in the query too** (2026-09-30, `SerperSearcher(exclude=)`). Measured
  live: a plain query had 3 of its 10 hits blocked (reddit, facebook, youtube); the same query with 14 `-site:`
  operators had 0, for the same 1 credit, with the 7 shared hits in the same order and the freed slots filled by
  PubMed Central, Britannica and a blog. Before, we crawled 7 pages where we had paid for 10. Google ignores words
  past 32 and an operator counts as one, so as many exclusions as fit go after the claim, leakiest hosts first;
  the claim is never cut. `not_blocked()` stays in the chain: Google honours `-site:` reliably, not as a guarantee.

## Latency

Traced a full check (Nepal example, 4 claims): 8.9s, of which the critical path was one dead site hitting the 6s
crawl timeout, plus one Serper outlier (3.3s vs ~0.8s p50). Fixes → **~5–6s**:
early exit *while* crawling (cancel remaining crawls), crawl timeout 2.5s, hedged Serper requests (duplicate after
1.2s, first reply wins).

Remaining time is mostly the atomizer call (~2s, blocking everything) and crawling. Next: stream atoms out of the
atomizer so claims start searching before the call finishes, and `astream()` so the UI shows the first verdict at
~2–2.5s. See [streaming-pipeline.md](streaming-pipeline.md).

Caching (single-flight + LRU) was prototyped and **removed**: it belongs to components as wrappers
(`Cached(crawler)`), not in the orchestrator.

## Naming

Renamed to **fact-assessor** / `FactAssessor` (unused on PyPI and GitHub) to avoid clashing with an existing
project's name.

## Streaming pipeline refactor

Rebuilt `FactAssessor` on composable steps (`pipeline.py`). A/B against the previous `main`, 4 alternating live
runs each on the same texts: Nepal median 5.9s → 6.5s, mixed 6.8s → 5.0s, with one large outlier on each side;
verdicts unchanged. Read as: no regression beyond network noise. The latency gain from streaming needs the
streaming atomizer (claims currently all appear when the LLM call returns, ~2s in).

## GLiNER under the per-claim timeout

The end-to-end eval had GLiNER decide nothing at the normal 15s per-claim timeout. Measured on recorded evidence
(M3 Max, 14 cores), three causes, each fixed and re-measured:

- **Throughput.** The model is CPU-bound, ~1.2ms per token (~180ms a snippet row, ~250ms a page row), and batching
  doesn't help on CPU (5 rows cost 5x one). One caller at a time left cores idle: 2 callers x 7 intra-op threads
  do 7.4 rows/s vs 5.8 (1 x 14: 5.5; 14 x 1: 7.5, but with 14 inferences' buffers at once).
- **Huge snippets.** Some DuckDuckGo snippets run to 3,000+ tokens (53 of 1,410 over 512) and were scored whole;
  batches pad to their longest row, so one made a 5-snippet call 8.3s instead of 0.9s. They're now cut to their
  best passage, like pages. Short text 15.4s -> 6.0s; medium texts' timeouts 5 of 9 and 5 of 8 -> 0.
- **Every claim at once.** `Verify` started all claims together; a 20-claim text needs ~250 rows (~34s of model
  time), so claims shared the model evenly and all 20 timed out. `Verify(concurrency=)` (default: the judge's
  `concurrency`) makes claims wait for a slot, with the timeout starting at the slot. With any limit, 0 of 20 timed
  out and the text took ~35s; the limit set how soon verdicts arrive: first at 4.8s / 7.0s / 9.6s, half by
  21.3s / 23.7s / 24.4s for 3 / 5 / 8. `GlinerRunner` sets 3; Laya, Jev and LLM runners keep no limit.

Not causes (measured): page splitting (<0.05s a page), the shared tokenizer lock (<0.4s a text), event-loop
blocking (<0.1s lag), and work left over from timed-out claims (none started after its claim timed out; the ~110s
text from the first eval run didn't reproduce).

## Alternative judge: GLiNER2.5-decide (ONNX)

GLiNER (`GlinerRunner`) on nishparadox/gliner2.5-decide-onnx, 15-case benchmark (`scripts/compare_judges.py`), M-series
Mac: fp32 on CPU 12/15 in 2.2s (Laya: 13/15 in 0.27s on MPS); int8 7/15 in 1.0s (the model card's "up to 0.18"
accuracy loss shows); fp32 on CoreML is slower than CPU (CoreML takes only 1,288 of 3,592 graph nodes, so it
keeps switching back to CPU). Wording: evidence-then-claim text 12/15 > claim-then-evidence 11/15 > claim in the
instruction 9/15. The graph accepts padded batches (no CPU speedup, but one call per claim). All of GLiNER's misses
were false "supports", the riskier direction for fact-checking, so Laya stays the default. The encoding
reimplements the repo's `gliner_onnx.py` (we load weights and tokenizer, not its code) and reproduces its scores.

## Search without an API key

Public SearXNG: 0 of 25 instances that searx.space rated healthy returned JSON (429 rate limits, 403/418 bot
blocks, JSON disabled). A self-hosted instance works (`SearxngSearcher`). DuckDuckGo via `ddgs`: 10 relevant results
for 5/5 test queries, 0.7-3.3s each (Serper ~0.8s); an end-to-end check with it gave correct verdicts in 9.2s.

## Crawlers: browser vs plain HTTP

Same 20 live search-result URLs: `Crawl4AICrawler` 16/20 pages in 4.8s (2.14s per page median, 10 concurrent tabs);
`HTTPXCrawler` 11/20 in 1.0s (0.15s per page; it can't read JavaScript-rendered or bot-blocking pages);
`CascadedCrawler(HTTPX, browser)` 16/20 in 2.9s. Extracted text size was the same (~6.2-6.3k chars median).
End to end, 3 alternating rounds: Nepal 6.1s → 5.2s median, mixed 5.8s → 6.0s, same verdicts. The early exit means
crawling often isn't what a check waits on, so the browser stays the default until more runs confirm the gain.

### Crawling under load (2026-10-04)

Under load (600 URLs at once, about 10 texts), crawling was CPU-bound, not network-bound: BeautifulSoup parsing maxed
one core (102%), parsing ran inside the HTTP slot and the 2.5s deadline, so downloads waited a median 16.7s for a slot
and pages that had arrived timed out mid-parse. Raising the HTTP cap made it worse (100 slots: 230 pages read vs 626).

- **Parser**: lxml directly instead of BeautifulSoup on lxml: identical text on 258 crawled pages, 8.5x faster parse.
  `clean_text` skips lines without a letter or digit before any regex: identical output, 5x faster. Together about 7x
  less CPU per page. Parsing now happens after the slot and the deadline are released.
- **Impit in the middle**: impit is an HTTP client with a real browser's TLS fingerprint. httpx with our honest bot
  user agent and impit get through different sites (impit as Chrome even drew JavaScript challenges where httpx got the
  page), so they cascade: on 313 pages httpx couldn't read, impit as Firefox read 62 (as Chrome: 23, the browser: 80
  in 3x the time). Crawlee (which ships impit) itself was slower for us: its scheduler ramps up for large crawls (98s vs
  35s on the same 600 URLs).
- **Browser user agent**: crawl4ai's default claims Chrome 116 on Linux. A current desktop Chrome user agent rescued
  66/64 of 280 pages vs 52/47 (two runs), same speed. Stealth mode added nothing; Firefox read 27 in twice the time,
  WebKit 52 in 30% more time. More tabs (20) rescued fewer: Chromium slows and pages hit the 2.5s timeout.
- **`when=~(StatusIn(404, 410) | ContentType("pdf"))`**: the browser rescued 0 of 95 unreadable PDFs and 0 of 2 404s.
  Every other kind of HTTP failure had some rescues (403 challenge 7%, bare 403 22%, timeout 18%, short HTML 49%), so
  any further rule trades pages for speed. Whether the browser works depends mostly on the site (ResearchGate 0/110,
  ScienceDirect 0/74, PMC 26/27); a site list was judged not worth its upkeep.
- **PDF limit 50 MB** (was 20): a PDF cut short can't be read at all, and theses run 25-35 MB.
- **HTTP cap 50** (was 20, connections only now): on 600 URLs, 319-327 pages by 20s vs 275, but fewer in total
  (335-355 vs 377-383): slow pages that would have arrived late anyway.

Results, 600 URLs at once: 50s → 38s wall, CPU 102% → 55%, 335 → 377 pages, 81 → 140 within 10s. With the
resolver, on 600 hits: 350 read (httpx then browser) vs about 395 (httpx, impit, browser), 17% fewer browser renders.
Accuracy, all 8,481 hits of the Serper eval crawled the same day from the same IP and replayed with Jev: old cascade
F1 0.752, new 0.755 (FactReasoner 0.731). The headline 0.779 was crawled on 2026-09-30; by 2026-10-04 MDPI and IOP
blocked this machine for every crawler (after a day of repeated crawling), which costs both cascades alike. Live, 10
answers at once with a 30s claim deadline: F1 0.692 vs 0.605 for the old crawling (6 at once, 42s). End-to-end
latency is still to be re-measured: a run giving 10 texts at once p50 21.4s / p90 22.6s most likely had the atomizer
falling back to sentences (OpenAI credits ran out), with no LLM call and no source-paper search, so it is not counted.

## LLM judge, claim filters, shared models

The LLM judge (then `LLMJudge` with a fact-checking prompt; now `DecisionJudge(LLMRunner())`, which renders any
question generically) on the 15 judge cases, 4 runs each: gpt-6-luna 15/15 in 3 runs (14 in one), gpt-5.6-luna
14/15; ~2.1-2.3s for 15 pairs either way. One batched call vs 15 parallel calls: 2.34s vs 2.07s median
(gpt-6-luna); `LLMRunner` packs 40 items per call and merges concurrent callers (fewer calls and tokens for
~0.3s). Single runs had gpt-5.4-mini and gpt-5.4-nano at 14/15 and 13-14/15. End to end the judge isn't the
bottleneck (5.5s vs 5.3s with Laya). Re-measure with the generic prompt before relying on these numbers.

Claim filters (19 cases): Laya 17/19 in 0.22s, GLiNER 17/19 in 2.3s; Laya's misses keep opinions, GLiNER's drop real
claims, so Laya stays the default.

Laya's weights are process-wide (one Router per device) and GLiNER's per (model, variant). Since the
decision-runner refactor (2026-09-30) the filter and the judge are handed a `DecisionRunner` (`FactAssessor`
shares one `LayaRunner` between them); the runner's batch queue belongs to the event loop it first runs on, so
use one runner per loop (`assess` or `assess_sync` on a given assessor, not both).
