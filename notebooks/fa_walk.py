import marimo

__generated_with = "0.25.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import asyncio
    import time

    import marimo as mo

    return asyncio, mo, time


@app.cell
def _(mo):
    mo.md("""
    # fact-assessor, step by step

    A guided tour, from the three-line version to building your own pipeline:

    1. **The three-line version**: `FactAssessor().assess(text)`
    2. **Building blocks**: steps, `>>`, `Map`, `Filter`, `Take` (toy data, no network)
    3. **Each component on its own**: what goes in, what comes out
    4. **Chaining components**: atomizer, searcher, crawler chains
    5. **Putting it together**: a custom `FactAssessor`, streamed live
    6. **Writing your own step**

    Needs `OPENAI_API_KEY` in a `.env` (or the environment). Search uses DuckDuckGo, which needs no key; swap in
    `SerperSearcher()` (faster and steadier, needs `SERPER_API_KEY`) anywhere you see `DuckDuckGoSearcher()`.
    """)
    return


@app.cell
def _():
    text = (
        "Nepal's earthquake in 2017 of 7.8 magnitude scale caused massive damage. "
        "Total lives lost were 1 million people."
    )
    return (text,)


@app.cell
def _(mo):
    mo.md("""
    ## 1. The three-line version

    `FactAssessor()` builds the default pipeline. `assess` returns everything once all claims are checked.
    (`assess_sync` is the same for non-async code.) The default searcher is Serper, which needs `SERPER_API_KEY`;
    here we pass DuckDuckGo instead, which needs no key.
    """)
    return


@app.cell
async def _(Take, mo, text):
    from factassessor import DuckDuckGoSearcher, FactAssessor, not_blocked

    # FactAssessor() alone searches with Serper; this is the same pipeline with DuckDuckGo (no API key)
    async with FactAssessor(searcher=DuckDuckGoSearcher() >> not_blocked() >> Take(5)) as _fa:
        quick = await _fa.assess(text)

    mo.vstack(
        [
            mo.md(f"**Fact score:** {quick.fact_score:.0%} in {quick.latency_ms / 1000:.1f}s"),
            mo.ui.table(
                [{"verdict": a.verdict, "confidence": round(a.confidence, 2), "claim": a.atom.text} for a in quick.atoms],
                selection=None,
            ),
        ]
    )
    return DuckDuckGoSearcher, FactAssessor, not_blocked


@app.cell
def _(mo):
    mo.md("""
    ## 2. Building blocks

    Everything is a **step**: it takes an async stream of items and yields an async stream of items. Steps chain
    with `>>`. Here on plain numbers, so you can see the behaviour without any network:

    | Step | Does |
    |---|---|
    | `Map(fn)` | one item in, one out (return `None` to drop it) |
    | `FlatMap(fn)` | one item in, many out |
    | `Filter(pred)` | keep items where `pred(item)` is true |
    | `Take(n)` | first n items, then stop (and cancel unfinished work upstream) |
    | `Scan(fn, initial)` | running total: `fn(total, item)` after each item |
    | `TakeUntil(pred)` | items up to and including the first one where `pred` holds, then stop |

    `once(x)` makes a one-item stream to feed a chain; `collect(stream)` gathers a stream into a list.
    """)
    return


@app.cell
async def _():
    from factassessor import Filter, FlatMap, Map, Take, collect, once

    async def numbers():
        for n in range(10):
            yield n

    chain = Filter(lambda n: n % 2 == 0) >> Map(lambda n: n * 10) >> Take(3)
    await collect(chain(numbers()))
    return Filter, FlatMap, Map, Take, collect, numbers, once


@app.cell
def _(mo):
    mo.md("""
    **Conditions and shorthand.** A `Predicate` is a condition: combine with `&` (and), `|` (or), `~` (not). In a chain, a
    `Predicate` filters and a plain function transforms, so you rarely need to write `Filter(...)` / `Map(...)` at all.
    `>>` always means "then".
    """)
    return


@app.cell
async def _(Filter, collect, numbers):
    from factassessor import Predicate, Scan, TakeUntil

    even, big = Predicate(lambda n: n % 2 == 0), Predicate(lambda n: n > 5)
    # a Predicate is a condition on ONE item; to run it over a stream, wrap it in Filter (or put it in a chain with >>)
    print("even & big      :", await collect(Filter(even & big)(numbers())))
    print("even | big      :", await collect(Filter(even | big)(numbers())))
    print("~even, then x10 :", await collect((~even >> (lambda n: n * 10))(numbers())))
    print("running totals until >= 10:", await collect((Scan(lambda total, n: total + n, 0) >> TakeUntil(lambda t: t >= 10))(numbers())))
    return (Predicate,)


@app.cell
def _(mo):
    mo.md("""
    **Async functions run concurrently** and each result is passed on the moment it's ready (completion order).
    That's what makes the whole pipeline fast: a slow item never holds up a fast one. Sync functions keep the input
    order (useful for ranked search results).
    """)
    return


@app.cell
async def _(Map, asyncio, collect, time):
    async def slow_square(n):
        await asyncio.sleep(n / 10)  # 3 -> 0.3s, 1 -> 0.1s, 2 -> 0.2s
        return n * n

    async def _three():
        for n in (3, 1, 2):
            yield n

    _t = time.perf_counter()
    _out = await collect(Map(slow_square)(_three()))
    print(f"{_out} in {time.perf_counter() - _t:.2f}s (all at once: ~0.3s, not 0.6s; fastest first)")
    return


@app.cell
async def _(FlatMap, collect, once):
    async def words(sentence):
        for w in sentence.split():
            yield w

    await collect(FlatMap(words)(once("one text becomes many items")))  # the atomizer works like this: text -> atoms
    return


@app.cell
def _(mo):
    mo.md("""
    ## 3. Each component on its own

    The real pipeline uses the same building blocks. Each component has a **role** (a base type) and an
    implementation, and you only ever implement one method to make your own:

    | Role | You implement | Used here |
    |---|---|---|
    | `Atomizer` | `atomize(text)` | `LLMAtomizer` |
    | `ClaimFilter` | `score(atom)` → P(factual claim) | `DecisionClaimFilter` (on any runner) |
    | `Searcher` | `search(query)` | `SerperSearcher` (default), `DuckDuckGoSearcher` (no key, used here), `SearxngSearcher` |
    | `Crawler` | `crawl(url)` | `Crawl4AICrawler` (also `HTTPXCrawler`, `CascadedCrawler`) |
    | `Judge` | `judge(claim, docs)` | `DecisionJudge` (on any runner: `LayaRunner`, `LLMRunner`, `GlinerRunner`, `SystemOneRunner`) |
    | `Policy` | `settled`, `verdict` | `WeightedPolicy` |

    `DecisionClaimFilter` and `DecisionJudge` ask their model through a `DecisionRunner` (`SystemOneRunner()`, Jev over HTTP, by default;
    `LayaRunner()` in-process, used in this walkthrough): `FactAssessor()` gives them one shared runner, and Laya's
    weights load once per process however many runners exist. The next cell just warms it up so the first call
    is quick.
    """)
    return


@app.cell
async def _(time):
    from factassessor import LayaRunner

    _t = time.perf_counter()
    await LayaRunner().aload()  # loads the process-wide Laya model (~3s; ~0.8 GB download the very first time)
    print(f"Laya ready in {time.perf_counter() - _t:.1f}s")
    return


@app.cell
def _(mo):
    mo.md("""
    ### 3a. `LLMAtomizer`: text → atoms

    One LLM call splits the text into single-fact claims, each understandable on its own. Each atom remembers the
    `span` of the words it came from, for highlighting.
    """)
    return


@app.cell
async def _(mo, text):
    from factassessor import LLMAtomizer

    atoms = await LLMAtomizer().atomize(text)
    mo.ui.table([{"id": a.id, "atom": a.text, "from the text": text[a.span[0] : a.span[1]]} for a in atoms], selection=None)
    return LLMAtomizer, atoms


@app.cell
def _(mo):
    mo.md("""
    ### 3b. `DecisionClaimFilter`: atoms → factual claims

    Scores P(factual claim) for each atom and drops opinions, greetings, and questions. It's a step, so it chains
    straight after the atomizer. Here every atom is scored (plus one opinion) to show what would be kept at 0.4.
    """)
    return


@app.cell
async def _(asyncio, atoms, mo):
    from factassessor import DecisionClaimFilter, LayaRunner

    _opinion = atoms[0].model_copy(update={"id": 99, "text": "I think this is the saddest thing ever."})
    _filter = DecisionClaimFilter(LayaRunner(), threshold=0.4)
    _all = [*atoms, _opinion]
    _scores = await asyncio.gather(*(_filter.score(a) for a in _all))  # score(atom) -> P(factual claim)
    mo.ui.table(
        [{"kept at 0.4": p >= _filter.threshold, "claim_score": round(p, 2), "atom": a.text} for a, p in zip(_all, _scores)],
        selection=None,
    )
    return (DecisionClaimFilter,)


@app.cell
def _(mo):
    compare_gliner_filter = mo.ui.checkbox(label="Compare with GLiNER2.5-decide (downloads ~1.75 GB the first time)")
    mo.vstack(
        [
            mo.md("""
    **Other claim filters.** `ClaimFilter` is a role: any class with `score(atom)` works, and `FactAssessor(claim_filter=...)`
    takes it. The same filter on `GlinerRunner()` is the built-in alternative: 17/19 on `data/claim_cases.json` like Laya, but its misses
    drop real claims (never checked) where Laya's keep opinions (one wasted search), and it's ~10x slower.
    """),
            compare_gliner_filter,
        ]
    )
    return (compare_gliner_filter,)


@app.cell
async def _(DecisionClaimFilter, asyncio, atoms, compare_gliner_filter, mo, time):
    mo.stop(not compare_gliner_filter.value, mo.md("*Tick the box to score the same atoms with both filters.*"))
    from factassessor import GlinerRunner, LayaRunner

    _scores, _times = {}, {}
    for _name, _filter in (("Laya", DecisionClaimFilter(LayaRunner())), ("GLiNER", DecisionClaimFilter(GlinerRunner()))):
        await _filter.aload()
        _t = time.perf_counter()
        _scores[_name] = await asyncio.gather(*(_filter.score(a) for a in atoms))
        _times[_name] = time.perf_counter() - _t
    mo.vstack(
        [
            mo.md(f"Laya {_times['Laya'] * 1000:.0f} ms, GLiNER {_times['GLiNER'] * 1000:.0f} ms for {len(atoms)} atoms"),
            mo.ui.table(
                [{"atom": a.text, "Laya": round(l, 2), "GLiNER": round(g, 2)} for a, l, g in zip(atoms, _scores["Laya"], _scores["GLiNER"])],
                selection=None,
            ),
        ]
    )
    return


@app.cell
def _(mo):
    mo.md("""
    ### 3c. `DuckDuckGoSearcher`: query → search hits

    Web results as `{"url", "title", "snippet"}`, in rank order, with no API key. On its own it returns everything
    the search gives; chaining a `Filter` and `Take` shapes it (section 4). `SerperSearcher()` (Google via Serper)
    is the faster, steadier choice when you have a key, and `SearxngSearcher(url)` works with your own SearXNG.
    """)
    return


@app.cell
async def _(DuckDuckGoSearcher, atoms, mo):
    searcher = DuckDuckGoSearcher(num=10)
    raw_hits = await searcher.search(atoms[0].text)
    mo.ui.table(
        [{"rank": i + 1, "title": h["title"], "url": h["url"], "snippet": h["snippet"]} for i, h in enumerate(raw_hits)],
        selection=None,
    )
    return raw_hits, searcher


@app.cell
def _(mo):
    mo.md("""
    ### 3d. `Crawl4AICrawler`: url → page

    Fetches a page with a headless browser and returns clean plain text (no links, citations, or menus), as a `Fetch`:
    the page, the status, and whether the crawler's `accept` rule passed. `if f:` means a page you can use; as a step,
    only those come out.
    """)
    return


@app.cell
async def _(mo, raw_hits):
    from factassessor import Crawl4AICrawler

    crawler = Crawl4AICrawler(timeout=2.5)
    page = (_f := await crawler.crawl(raw_hits[0]["url"])) and _f.page
    mo.md(f"**{page['title']}** ({len(page['text']):,} chars)\n\n> {page['text'][:400]}…" if page else "*(crawl failed)*")
    return crawler, page


@app.cell
def _(mo):
    mo.md("""
    **Faster crawlers.** `Crawler` is a role too. `HTTPXCrawler` fetches with a plain HTTP request (no browser, no
    JavaScript): much faster, but it can't read pages built by JavaScript. `CascadedCrawler(fast, browser)` tries the fast
    one first and opens the browser only for pages it couldn't read. The same search hits through each:
    """)
    return


@app.cell
async def _(asyncio, crawler, mo, raw_hits, time):
    from factassessor import CascadedCrawler, HTTPXCrawler

    _urls = [h["url"] for h in raw_hits[:5]]
    _fast = HTTPXCrawler(timeout=2.5)
    _rows = []
    for _name, _crawler in (
        ("Crawl4AICrawler (browser)", crawler),
        ("HTTPXCrawler", _fast),
        ("CascadedCrawler(HTTPXCrawler, browser)", CascadedCrawler(_fast, crawler)),
    ):
        _t = time.perf_counter()
        _pages = await asyncio.gather(*(_crawler.crawl(u) for u in _urls))
        _rows.append({"crawler": _name, "pages read": f"{sum(map(bool, _pages))}/{len(_urls)}", "all at once": f"{time.perf_counter() - _t:.1f}s"})
    await _fast.stop()
    mo.ui.table(_rows, selection=None)
    return


@app.cell
def _(mo):
    mo.md("""
    ### 3e. `DecisionJudge` + `WeightedPolicy`: evidence → verdict

    The judge labels each passage *supports* / *refutes* / *not_enough_info* for a claim (pages are cut to their most
    relevant passage first). The policy weighs strong evidence into one verdict, and decides when there's enough
    evidence to stop looking (`settled`).
    """)
    return


@app.cell
async def _(atoms, mo, page, raw_hits):
    from factassessor import DecisionJudge, LayaRunner, WeightedPolicy

    judge = DecisionJudge(LayaRunner())  # shares the Laya model the filter uses
    policy = WeightedPolicy(strong=0.7, early_exit=0.9)
    _evidence = await judge.judge(atoms[0].text, raw_hits[:5] + ([page] if page else []))
    _verdict, _confidence = policy.verdict(_evidence)
    mo.vstack(
        [
            mo.md(f"**{atoms[0].text}** → **{_verdict}** ({_confidence:.2f}); settled: {policy.settled(_evidence)}"),
            mo.ui.table(
                [{"label": e.label, "prob": round(e.prob, 2), "source": e.source, "url": e.url, "passage": e.text} for e in _evidence],
                selection=None,
            ),
        ]
    )
    return judge, policy


@app.cell
def _(mo):
    compare_llm = mo.ui.checkbox(label="LLMRunner (gpt-6-luna; calls the OpenAI API)")
    compare_gliner_judge = mo.ui.checkbox(label="GlinerRunner (downloads ~1.75 GB the first time)")
    mo.vstack(
        [
            mo.md("""
    **Other runners.** The judge asks its model through a `DecisionRunner`. `LLMRunner` asks an LLM to pick an option
    per passage (the most accurate we measured: 15/15 vs Laya's 13/15, ~10x slower per pair); `GlinerRunner` runs
    GLiNER2.5-decide on CPU (12/15). Tick to judge the same evidence with them:
    """),
            mo.hstack([compare_llm, compare_gliner_judge], justify="start"),
        ]
    )
    return compare_gliner_judge, compare_llm


@app.cell
async def _(atoms, compare_gliner_judge, compare_llm, judge, mo, page, policy, raw_hits, time):
    from factassessor import DecisionJudge, GlinerRunner, LLMRunner

    _docs = raw_hits[:5] + ([page] if page else [])
    _judges = [("Laya", judge)]
    if compare_llm.value:
        _judges.append(("LLM", DecisionJudge(LLMRunner())))
    if compare_gliner_judge.value:
        _judges.append(("GLiNER", DecisionJudge(GlinerRunner())))
    _rows = []
    for _name, _judge in _judges:
        await _judge.aload()
        _t = time.perf_counter()
        _ev = await _judge.judge(atoms[0].text, _docs)
        _verdict, _conf = policy.verdict(_ev)
        _rows.append({
            "judge": _name, "verdict": _verdict, "confidence": round(_conf, 2), "time": f"{time.perf_counter() - _t:.2f}s",
            "labels": ", ".join(f"{e.label} {e.prob:.2f}" for e in _ev),
        })
    mo.ui.table(_rows, selection=None)
    return


@app.cell
def _(mo):
    mo.md("""
    ## 4. Chaining components

    Components are steps, so they chain with the building blocks. A `Filter` filters **whatever flows at that
    point**: atoms after the atomizer, search hits after the searcher.
    """)
    return


@app.cell
async def _(LLMAtomizer, DecisionClaimFilter, Predicate, Take, collect, mo, once, text):
    from factassessor import LayaRunner

    long_enough = Predicate(lambda atom: len(atom.text) > 15)
    atomizer_chain = LLMAtomizer() >> DecisionClaimFilter(LayaRunner(), threshold=0.4) >> long_enough >> Take(8)
    _chained = await collect(atomizer_chain(once(text)))
    mo.ui.table([{"atom": a.text, "claim_score": round(a.claim_score, 2)} for a in _chained], selection=None)
    return (atomizer_chain,)


@app.cell
async def _(Take, atoms, collect, mo, not_blocked, once, searcher):
    searcher_chain = searcher >> not_blocked() >> Take(5)  # drop social/video sites, keep the top 5
    hits = await collect(searcher_chain(once(atoms[0].text)))
    mo.ui.table([{"rank": i + 1, "title": h["title"], "url": h["url"]} for i, h in enumerate(hits)], selection=None)
    return hits, searcher_chain


@app.cell
async def _(collect, crawler, hits, mo, time):
    async def _urls():
        for h in hits:
            yield h["url"]

    _t = time.perf_counter()
    _pages = await collect(crawler(_urls()))  # every page crawled at once, yielded as each finishes
    print(f"{len(_pages)} of {len(hits)} pages crawled in {time.perf_counter() - _t:.1f}s")
    mo.ui.table([{"title": p["title"], "url": p["url"], "chars": len(p["text"])} for p in _pages], selection=None)
    return


@app.cell
def _(mo):
    mo.md("""
    ### `Verify`: one claim, start to finish

    `Verify` runs your searcher, crawler, judge, and policy for one claim: judge the snippets; if they don't settle
    it, crawl every hit at once, judge each page as it lands, and stop (cancelling the remaining crawls) as soon as
    the policy says settled. It's built from the same blocks:
    `crawler >> judge_page >> Scan(add, evidence) >> TakeUntil(policy.settled)`.

    As a step, `Verify` checks claims concurrently. By default every claim starts at once, which suits Laya and the LLM
    judge. A slow runner sets how many claims a judge on it takes at once (`GlinerRunner`: 3), and claims wait for a free slot;
    a claim's timeout starts when it gets one. Override it with `Verify(..., concurrency=5)` or
    `FactAssessor(max_concurrent_claims=5)`; `None` means no limit.
    """)
    return


@app.cell
async def _(atoms, crawler, judge, mo, policy, searcher_chain, time):
    from factassessor import Verify

    _verify = Verify(searcher_chain, crawler, judge, policy, timeout=15)
    _t = time.perf_counter()
    _one = await _verify.verify(atoms[0])
    mo.md(
        f"**{_one.atom.text}** → **{_one.verdict}** ({_one.confidence:.2f}) in {time.perf_counter() - _t:.1f}s, "
        f"from {len(_one.evidence)} passages ({sum(e.source == 'page' for e in _one.evidence)} from crawled pages)"
    )
    return


@app.cell
def _(mo):
    mo.md("""
    ## 5. Putting it together

    Hand your parts to `FactAssessor`: it wires them into `Verify` and runs every claim concurrently. Two ways to give it
    a claim filter:

    ```python
    # 1. separate arguments: FactAssessor chains atomizer >> claim_filter >> Take(max_claims) itself
    FactAssessor(atomizer=LLMAtomizer(), claim_filter=DecisionClaimFilter(threshold=0.4), searcher=..., judge=...)

    # 2. a chain that already filters (what we built in section 4): say so with claim_filter=None
    FactAssessor(atomizer=atomizer_chain, claim_filter=None, searcher=..., judge=...)
    ```

    `stream` shows each claim as it's found and flips it to its verdict the moment it settles.
    """)
    return


@app.cell
def _(
    FactAssessor,
    atomizer_chain,
    crawler,
    judge,
    policy,
    searcher_chain,
):
    # the atomizer chain already filters, so claim_filter=None (otherwise FactAssessor adds its default filter)
    custom = FactAssessor(atomizer=atomizer_chain, claim_filter=None, searcher=searcher_chain, crawler=crawler, judge=judge, policy=policy)
    return (custom,)


@app.cell
def _(mo, text):
    text_box = mo.ui.text_area(value=text, label="Text to fact-check", full_width=True, rows=3)
    run_button = mo.ui.run_button(label="Fact-check (streaming)")
    mo.vstack([text_box, run_button])
    return run_button, text_box


@app.cell
async def _(custom, mo, run_button, text_box):
    mo.stop(not run_button.value, mo.md("*Press **Fact-check** to stream.*"))
    _rows = {}
    result = None
    async for _event in custom.stream(text_box.value):
        if _event.type == "claim_found":
            _rows[_event.atom.id] = {"verdict": "⏳ checking…", "confidence": None, "claim": _event.atom.text}
        elif _event.type == "claim_verified":
            _r = _event.result
            _rows[_r.atom.id] = {"verdict": _r.verdict, "confidence": round(_r.confidence, 2), "claim": _r.atom.text}
        else:
            result = _event.result
        mo.output.replace(mo.ui.table([_rows[k] for k in sorted(_rows)], selection=None))
    return (result,)


@app.cell
def _(mo, result):
    from collections import Counter

    from factassessor import kg

    # The knowledge graph is a view of the result, built on demand (~0.1ms), not a pipeline step.
    _graph = kg.build(result)
    _score = "n/a" if result.fact_score is None else f"{result.fact_score:.0%}"
    _nodes = Counter(n["kind"] for n in _graph["nodes"])
    _edges = Counter(e["relation"] for e in _graph["edges"])
    mo.vstack(
        [
            mo.md(f"""
    ### The knowledge graph

    **Fact score {_score}** (`result.fact_score`, computed from the claims) in {result.latency_ms / 1000:.1f}s.

    `kg.build(result)` gives plain dicts: **{dict(_nodes)}** nodes and **{dict(_edges)}** edges.

    - `sentence` →contains→ `claim`: which sentence of your text each claim came from (`same_sentence` links them)
    - `passage` →supports / refutes (probability)→ `claim`: the evidence text behind each verdict; a passage used by
      several claims is one node
    - `source` →published→ `passage`: the site it came from

    `kg.to_mermaid(graph)` draws it:
    """),
            mo.mermaid(kg.to_mermaid(_graph)),
            mo.ui.table(
                [{"from": e["source"], "relation": e["relation"], "weight": e["weight"], "to": e["target"]} for e in _graph["edges"]],
                selection=None,
            ),
        ]
    )
    return


@app.cell
def _(mo):
    mo.md("""
    ## 6. Writing your own step

    Two ways:

    - **A function or a `Predicate`**: drop it straight into a chain (sync or async). Enough for most things.
    - **A `Step` subclass**: implement `__call__(items)` as an async generator. Use it when the step needs state
      across items, like this one that keeps at most one search hit per website.
    """)
    return


@app.cell
async def _(Take, atoms, collect, mo, not_blocked, once, searcher):
    from urllib.parse import urlparse

    from factassessor import Step

    class OnePerSite(Step):
        """Search hits -> at most one hit per website (more independent sources)."""

        async def __call__(self, hits):
            seen = set()
            async for hit in hits:
                site = urlparse(hit["url"]).netloc.removeprefix("www.")
                if site not in seen:
                    seen.add(site)
                    yield hit

    _diverse = searcher >> not_blocked() >> OnePerSite() >> Take(5)
    _hits = await collect(_diverse(once(atoms[0].text)))
    mo.ui.table([{"site": urlparse(h["url"]).netloc, "title": h["title"]} for h in _hits], selection=None)
    return


@app.cell
def _(mo):
    mo.md("""
    Pass it in like any other searcher: `FactAssessor(searcher=searcher >> not_blocked() >> OnePerSite() >> Take(5))`.

    **Cleanup**: components that hold resources (browser, HTTP pool) close with `await custom.aclose()`, or use
    `async with FactAssessor(...) as fa:`. In a notebook, stopping the kernel does it too.
    """)
    return


if __name__ == "__main__":
    app.run()
