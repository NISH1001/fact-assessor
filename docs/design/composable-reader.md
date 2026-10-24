# Composable readers: `FirstOf`, resolvers as steps (design, 2026-10-23)

Status: **deferred** (2026-10-23). Three independent reviews (architecture, correctness, pragmatic) agreed it is
too big for the AKD goal; what was built instead is one crawler, `ResolvingCrawler(resolver, crawler)`
(`factassessor/crawlers/resolving.py`), nested as `CascadedCrawler(AKDDoiCrawler(), ResolvingCrawler(copies, web))`.
Revisit `FirstOf` when per-claim routing (AKD, else Serper) is built. Review findings to fold in then:

- `when` would mean opposite things (CascadedCrawler: "try the next one?" on the failed Fetch; FirstOf: "is this
  output good enough?"): give FirstOf's a different name; don't make FirstOf's behaviour depend on argument types.
- `|` on steps clashes with `|` on Predicates (a predicate in a chain becomes a Filter); prefer the class only.
- Resolver-as-step + ReadFirst only work as a pair (an implicit `(hit_url, [locations])` item); ResolvingCrawler hides it.
- Migration: `FactAssessor(crawler=X)` resolves today (resolver defaults on); the right mapping is
  `reader=resolver >> X`, not `reader=X`. scripts/eval_atoms.py reads `fa.verify.resolver` / `fa.crawler`.
- Lifecycle: a Step's children are found through `parts()`; CompositeResolver keeps them in a list.
- Errors: crawlers/resolvers swallow failures by contract, so a rejected key must be raised at the component (done
  for AKD: 401/403 -> InvalidAPIKeyError) for FirstOf's auth rule to see it; FirstOf(Verify, Verify) gives each
  alternative its own claim timeout (up to 2x).
- Streaming with `when`: held-back outputs only work for single-output alternatives; say what happens to outputs
  before the one passing `when`, and to an alternative that raises after yielding.
- Cancellation: iterate alternatives with aclosing so Take/TakeUntil cancel their crawls; test "no orphan crawl".
- Do the `not_blocked` -> `NotBlocked` rename as its own change.

## Why

Reading a search hit is today two mechanisms that don't compose. Crawlers compose (`CascadedCrawler(HTTPX, impit,
browser)`, a crawler that nests), but the resolver (free copies from arXiv / OpenAlex / PMC) is not a component: it
is an argument of `Verify`, which resolves every hit and reads its locations with private code (`Verify._resolve`,
`Verify._read`, `read_first`). So "read from AKD Scholar Search's full text first, and only if that fails resolve
free copies and crawl the web" has no place to go:

- put `AKDDoiCrawler` first in the crawler cascade, and `Verify` still resolves first and hands the crawler the
  copies (PMC, OpenAlex urls, no DOI) before the hit's own doi.org url, so AKD is mostly bypassed;
- drop the resolver, and papers AKD lacks lose their free copies;
- check AKD inside a resolver wrapper, and every paper is fetched from AKD twice.

The goal is **functional consistency**: every part of the pipeline is a step that composes with `>>`, "or else" is
one generic block usable for any step, and nothing is recomputed: per hit url, AKD runs at most once, the resolver
runs at most once and only when it's needed, and its locations feed every crawler after it.

Measured motivation (tmp/eval_atoms/RESULTS-2026-10-23.md): on the SciELF eval, AKD search (top 20) + AKD full text
+ 30 pages per claim matched Serper on clean answers (macro F1 0.840 vs 0.833) with no Serper credits; the
remaining gaps are papers AKD lacks (resolver + web fallback) and claims AKD can't settle (per-claim routing).

## Goals

- **`FirstOf(a, b, ..., when=None)`**, a generic step combinator, with `a | b` as shorthand: per item, run `a`;
  if it yields nothing useful, run `b` on the same item; and so on. Later alternatives never run for an item an
  earlier one handled. Works for any steps: crawlers, searchers, a resolve-then-read chain, whole `Verify` steps.
- **Resolvers are steps**: url -> its locations (`resolvers.locations(url, candidates)`: a direct-PDF hit, the free
  copies, the url itself). A resolver chains with `>>` like a searcher or crawler; `resolve(url)` stays.
- **`ReadFirst(crawler, min_copy_words=300)`**: locations of one hit -> the first one `crawler` reads, tried one at
  a time in order, returned as the hit's page (cited under the hit url). A copy other than the hit url needs
  `min_copy_words` words (bot-check pages are shorter), as `read_first` does today.
- **`Verify(search, reader, judge, policy, ...)`** reads every hit through one `reader` step (url -> page). Its
  resolver code goes; the default reader expresses today's behaviour.
- **`CascadedCrawler` stays** exactly as it is (the user's call): the crawler fallback with its `when=` failure
  rules. `FirstOf` given only crawlers builds a `CascadedCrawler`, so there is one implementation of "crawler, else
  crawler" and either spelling works.
- **Uppercase for every pipeline step**: `not_blocked()` -> `NotBlocked()`.
  Plain stream helpers that are functions, not steps (`once`, `collect`, `last`), stay lowercase.
- **Clean break** (the user's call): old names and arguments are removed, not aliased; every caller in the repo
  (library, tests, scripts, notebooks, docs) moves to the new ones in this change.
- **Same results**: `FactAssessor()` with defaults gives identical verdicts to today (acceptance test below).

## Non-goals

- New search or judging behaviour. This is a refactor plus one block; AKD setups are what it enables.
- Claim-level routing in the library defaults. `FirstOf(Verify(...), Verify(...), when=...)` must work, but the
  default `FactAssessor` stays single-searcher.
- Changing the stop-early policy, the page cap, streaming, or concurrency model.

## Anti-goals (a failure even if it works)

- **Slower defaults.** Today every hit of a claim is resolved and read at once and each page is judged the moment
  it lands. If the new reader read hits one after another, results would match and runs would crawl.
- **Losing crawler failure rules.** `CascadedCrawler(when=...)` decides from the failed `Fetch` (status, content
  type, body markers) whether the next crawler is worth trying (`NeedsBrowser()`: a JS shell or bot check goes on, a
  404 or paywall doesn't). It stays in `CascadedCrawler`, which `FirstOf` over crawlers builds.
- **Recomputation.** The resolver running for hits AKD read, or once per crawler; AKD asked twice for a url.

## Constraints

- Every step stays swappable behind a small async interface; component concerns (caching, retries) stay in
  wrappers, not in `Verify` (CLAUDE.md).
- Tests offline with fakes; TDD.
- `docs/WALKTHROUGH.md` and `notebooks/fa_walk.py` stay in sync; README, `docs/ARCHITECTURE.md`,
  `docs/design/functional.md` (combinator table), notebooks, scripts and tests move to the new names.

## The design

### Vocabulary after the change

| Block | Meaning |
|---|---|
| `a >> b` | then: `a`'s output stream feeds `b` (unchanged) |
| `FirstOf(a, b, ..., when=None)` / `a \| b` | per item: the first alternative that yields something useful |
| `CascadedCrawler(c1, c2, ..., when=)` | unchanged: url -> the first usable page; `FirstOf` of crawlers is one |
| resolver as a step | url -> [direct PDF?, copies..., url] |
| `ReadFirst(crawler, min_copy_words=300)` | one hit's locations -> its first readable page, cited under the hit url |
| `Map`, `FlatMap`, `Filter`, `Take`, `Scan`, `TakeUntil`, `NotBlocked` | as today (`NotBlocked` = `not_blocked`) |

### Usage

```python
web    = CascadedCrawler(HTTPXCrawler(), ImpitCrawler(), Crawl4AICrawler(), when=NeedsBrowser())   # unchanged
copies = CompositeResolver(ArxivResolver(), OpenAlexResolver(), PMCResolver())
reader = FirstOf(AKDDoiCrawler(), copies >> ReadFirst(web))      # AKD, else resolve once -> first readable copy

verify = Verify(search=AKDScholarSearcher(num=20) >> NotBlocked() >> Take(20),
                reader=reader, judge=DecisionJudge(passage_words=512), policy=WeightedPolicy())
fa = FactAssessor(atoms=LLMAtomizer() >> DecisionClaimFilter() >> Take(max_claims), verify=verify)

# the same blocks, elsewhere
search = FirstOf(AKDScholarSearcher(num=20), SerperSearcher())                       # AKD, else Serper
verify = FirstOf(Verify(akd_search, reader, judge), Verify(SerperSearcher(), reader, judge),
                 when=lambda result: result.verdict != "unverified")                   # Serper only for unsettled claims
fa = FactAssessor(searcher=SerperSearcher(), reader=reader)                            # Serper + AKD full text for DOIs
```

`FactAssessor()` with no arguments builds `reader = copies >> ReadFirst(web)` with today's resolver and cascade.
`FactAssessor(crawler=..., resolver=...)` are replaced by one argument, `FactAssessor(reader=...)`: any step
url -> page, used as is. Callers that passed `crawler=X` (no resolving wanted) pass `reader=X`; callers that
passed both pass `reader=resolver >> ReadFirst(X)`. The usage example `FactAssessor(searcher=SerperSearcher(),
reader=reader)` uses it.

### `FirstOf` semantics

- Per input item, alternatives run in order on that item (`once(item)` into each); the first that yields at least
  one output whose `when` holds wins, and its outputs pass on. `when=None`: any output counts.
- *(Design call)* If no alternative's output passes `when`, the **last alternative's output** passes on (held back
  while the next alternative runs, released if none does better). So `FirstOf(Verify(akd), Verify(serper),
  when=settled)` still reports a claim neither settles, as Serper's unverified result; nothing silently vanishes.
  An alternative that yields nothing at all (a crawler with no page) is just skipped; if all yield nothing, the item
  yields nothing, as a crawler that reads nothing does today.
- *(Design call)* An alternative that raises counts as "nothing" (the next one runs), **except** auth errors
  (`keys.auth_error(exc)` / `InvalidAPIKeyError`) and `asyncio.CancelledError`, which propagate: a rejected key must
  stop the whole check, as `Verify` and `FactAssessor` do today, not quietly fall back on every hit.
- Items are independent and run concurrently, as `Map` does today: `FirstOf` must not serialize items.
- **Crawlers: `CascadedCrawler`, unchanged.** When every alternative is a `Crawler`, `FirstOf(...)` returns
  `CascadedCrawler(..., when=when)`: a `Crawler` (`crawl(url) -> Fetch`) whose `when` is a predicate on the failed
  `Fetch` (the first usable page wins; after a failure `when(fetch)` decides whether the next crawler runs; the
  returned `Fetch` is the crawler it stopped at). `CascadedCrawler`'s code and behaviour don't change.
- Lifecycle: `parts()` returns the alternatives, so `aload`/`aclose` reach them (as `Chain` does).
- `a | b` on two steps builds `FirstOf(a, b)`; `(a | b) | c` flattens to `FirstOf(a, b, c)`. `|` on `Predicate`
  keeps meaning "or" for conditions.

### Resolvers as steps

A `Resolver` base step adapts the one method: `__call__(urls)` yields, per url, its locations
(`locations(url, await self.resolve(url))`; a failing `resolve` counts as no candidates). `CompositeResolver`,
`ArxivResolver`, `OpenAlexResolver`, `PMCResolver` get it; the `Resolver` Protocol (`resolve(url) -> list[str]`)
still describes what an implementer writes.

### `ReadFirst`: carrying the hit url

The locations of one hit must stay together so the page can be cited under the hit url and the copy rule applied.
Builder's choice of representation (a resolver step yielding one `(hit_url, [locations])` item per hit is the
obvious one; `ReadFirst` then tries the list in order with its crawler and returns `{**page, "url": hit_url}`).
Locations of one hit are tried sequentially (stop at the first readable); different hits run concurrently.

### `Verify` changes

- New signature `Verify(search, reader, judge, policy=None, timeout=..., concurrency=..., pages_per_claim=None)`;
  `reader` is any step url -> page. Reading a claim's hits = `reader` over their urls, `Take(pages_per_claim)`,
  judge each page as it lands, stop when settled: the same chain as today with `self.crawler` / `_resolve >> _read`
  replaced by `self.reader`.
- `Verify(searcher, crawler, judge, policy, resolver=...)` is replaced, not kept: callers build the reader
  (`resolver >> ReadFirst(crawler)`, or the crawler alone). `read_first`, `Verify._resolve` / `_read` and the
  `min_copy_words` argument of `Verify` go (the rule lives in `ReadFirst`).

### Renames (clean break, no aliases)

| Old (removed) | New |
|---|---|
| `not_blocked(domains)` | `NotBlocked(domains)` |
| `Verify(searcher, crawler, judge, policy, resolver=, min_copy_words=)` | `Verify(search, reader, judge, policy)` |
| `FactAssessor(crawler=, resolver=)` | `FactAssessor(reader=)` (plus `verify=` if added now) |

## Acceptance

1. All existing tests pass (updated to the new names where they test names, not behaviour).
2. *(Design call)* **Reader equivalence, offline**: before the refactor, write a characterization test with fake
   resolvers and crawlers that records, through today's `Verify(crawler, resolver=...)`, the page, cited url and
   copy-rule decision for a set of hits (direct PDF, good copy, short copy, no copy, failing resolver). After the
   refactor the same test, rewritten to `copies >> ReadFirst(web)`, must give the same results. It covers the code
   that changes (the cached replay below never resolves).
3. **Identical replay**: `scripts/eval_atoms.py replay --tag web-serper --passage-words 512 --passages 3
   --pages-per-claim 10 --judge decision --judge-model typesafe/jev-1.14-20261014` gives the same verdict for every
   atom as the stored `web-serper-j114-w512/results.json.gz` (2026-10-23). It checks `Verify(reader=...)` end to end
   on fixed evidence (~$1.20 of Jev). If any atom differs, rerun the old code once to tell Jev run-to-run variation
   from a real change before treating it as a failure.
4. **Not slower**: a fake-crawler test where every crawl sleeps 0.1s, 10 hits: reading takes ~0.1s, not ~1s.
5. *(Design call)* **No recomputation, per claim**: fakes count calls: a hit AKD reads is never resolved; a
   resolved hit is resolved once; AKD is asked once per url within a claim. Across claims and texts, repeats are a
   caching concern (CLAUDE.md: wrappers, not the orchestrator): `Cache(AKDDoiCrawler())` or a TTL cache inside
   `AKDDoiCrawler` like HTTPX's, measured separately; not part of this change.
6. **Crawlers**: `FirstOf(HTTPX, browser, when=NeedsBrowser())` is a `CascadedCrawler`; it sends a JS shell to the
   browser and a 404 not; all existing `CascadedCrawler` tests pass unchanged.
7. **Errors**: a `FirstOf` whose first alternative raises an auth error propagates it; one that raises anything
   else falls through to the next alternative.

## Left to the builder

- Module layout (`FirstOf` in `pipeline.py`; `ReadFirst` beside crawlers or in `pipeline.py`).
- The hit-url carrying representation for `ReadFirst`.
- How `FirstOf` detects "all alternatives are crawlers" (a type check is the obvious way).
- Whether `FactAssessor(atoms=..., verify=...)` is added now or later (the usage above assumes it).

## Out of this change, next

The AKD eval runs (live, AKD reader; per-claim routing AKD -> Serper) use these blocks once merged; the searcher's
duplicate-paper fix (same paper under two links) and an eval `--max-concurrent` flag are separate small changes.
