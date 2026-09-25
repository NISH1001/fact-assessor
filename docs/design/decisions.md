# Decisions and measurements

What we tried while building the prototype, what we measured, and what we picked. Numbers are from an M-series
Mac (MPS), Sept 2026. Revisit a decision when its evidence changes.

## Atomization

- **One LLM call does atomize + decontextualize** (`Atomizer`). Sentence/clause splitting (spaCy) kept compound
  claims together: "Nepal's earthquake in 2017 of 7.8 magnitude caused massive damage" is three facts, and the
  judge passed it as *supported* because 2 of 3 were right. Implicit context was also lost ("Total lives lost were
  1 million" got searched on its own and matched WWII pages).
- Prompt rules that mattered: one fact per claim; self-contained; unwrap hedges ("It was believed that X" → X);
  keep values exactly as written (we check, not correct); **keep subjects identifiable ("the Nepal earthquake",
  never "the earthquake") but don't repeat a detail that has its own claim** (otherwise "in 2017" leaked into every
  atom and contaminated all of them).
- Model: **`gpt-5.6-luna`, reasoning `none`** (~2s). Benchmarked on 3 decontextualization cases:

  | Model | Warm latency | Quality |
  |---|---|---|
  | gpt-5.6-luna (reasoning none) | ~1.7s | all correct |
  | gpt-6-luna (reasoning none) | ~2.0s | all correct |
  | gpt-5.4-mini (reasoning none) | ~1.0s | garbled a sentence in one run |
  | gpt-5.4-nano / gpt-4.1-nano | ~1.5s (4.1 spikes to 7s) | missed references |
  | gpt-5-nano (minimal) | ~1.0s | unusable |

  Default reasoning made the Luna models slower *and* worse ("Hi, paradox is paradox.").
- Structured output keyed by id / exact source quote, so a skipped or merged item only affects itself; spans are
  recovered by locating the quote (case/whitespace-insensitive), falling back to the best-matching sentence.
- On LLM failure: fall back to sentence atoms rather than failing the check.

## Check-worthiness filter (Laya)

- Laya's `noul` (yes/no) head was near-random for this (6–8/12). A **`choice`** question over
  factual_claim / opinion / question_or_request / social on the `english` checkpoint: **17/19**.
- Keep if P(factual_claim) ≥ **0.4**, not 0.5: dropping a real claim (never checked) costs more than keeping an
  opinion (one wasted search). Plain facts about little-known people score near the cutoff (~0.45–0.6).
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
- Chunking uses Laya's own tokenizer to an exact budget (max_len − head_max_len − claim − margin) so Laya never
  silently truncates. The judge keeps a private tokenizer copy under a lock: HF fast tokenizers aren't thread-safe
  ("Already borrowed").
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
- Blocked: facebook, instagram, tiktok, pinterest, threads (reposts, crawl badly), youtube (no text). Kept:
  twitter/x, linkedin (primary sources for people/orgs). Reddit kept for now.

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

`FactReasoner` is IBM's project; renamed to **fact-assessor** / `FactAssessor` (unused on PyPI and GitHub) to
avoid confusion.

## Streaming pipeline refactor

Rebuilt `FactAssessor` on composable steps (`pipeline.py`). A/B against the previous `main`, 4 alternating live
runs each on the same texts: Nepal median 5.9s → 6.5s, mixed 6.8s → 5.0s, with one large outlier on each side;
verdicts unchanged. Read as: no regression beyond network noise. The latency gain from streaming needs the
streaming atomizer (claims currently all appear when the LLM call returns, ~2s in).

## Alternative judge: GLiNER2.5-decide (ONNX)

`GlinerJudge` on nishparadox/gliner2.5-decide-onnx, 15-case benchmark (`benchmarks/compare_judges.py`), M-series
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
`FallbackCrawler(HTTPX, browser)` 16/20 in 2.9s. Extracted text size was the same (~6.2-6.3k chars median).
End to end, 3 alternating rounds: Nepal 6.1s → 5.2s median, mixed 5.8s → 6.0s, same verdicts. The early exit means
crawling often isn't what a check waits on, so the browser stays the default until more runs confirm the gain.

## LLM judge, claim filters, shared models

`LLMJudge` on the 15 judge cases, 4 runs each: gpt-6-luna 15/15 in 3 runs (14 in one), gpt-5.6-luna 14/15;
~2.1-2.3s for 15 pairs either way. One batched call vs 15 parallel calls: 2.34s vs 2.07s median (gpt-6-luna),
so parallel is the default and batching (`window_ms`) is opt-in for fewer calls. Single runs had gpt-5.4-mini and
gpt-5.4-nano at 14/15 and 13-14/15. End to end the judge isn't the bottleneck (5.5s vs 5.3s with Laya).

Claim filters (19 cases): Laya 17/19 in 0.22s, GLiNER 17/19 in 2.3s; Laya's misses keep opinions, GLiNER's drop real
claims, so Laya stays the default.

Laya and GLiNER models are now process-wide (one per device / per model+variant), with the micro-batching queue per
event loop, so components share models without being handed a runner and `assess_sync`'s background loop works.
