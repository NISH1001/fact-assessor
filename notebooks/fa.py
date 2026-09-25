# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo>=0.25.0",
#     "fact-assessor[gliner,ddg] @ git+https://github.com/NISH1001/fact-assessor",
# ]
# ///
# Runs standalone: `uvx marimo edit --sandbox <this file or its raw GitHub URL>`. Inside the repo, use
# `uv run marimo edit --no-sandbox notebooks/fa.py` so it uses the local code.
import marimo

__generated_with = "0.25.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import os

    import marimo as mo

    return mo, os


@app.cell
def _(mo):
    mo.md("""
    # FactAssessor

    text → atoms (one fact each) → claim filter → search → crawl → judge → verdicts, fact score, knowledge graph.

    **Build a pipeline** below by picking each part, see the exact code for your choice, and fact-check some text.
    Further down: **recipes**, the different ways to set up a pipeline in code. For how the pieces work inside,
    see `fa_walk.py`.
    """)
    return


@app.cell
def _(mo, os):
    filter_choice = mo.ui.dropdown(
        options=["Laya (local, default)", "GLiNER2.5-decide (ONNX, CPU)", "None (check every atom)"],
        value="Laya (local, default)",
        label="Claim filter",
    )
    searcher_choice = mo.ui.dropdown(
        options=["Serper (Google)", "DuckDuckGo (no key)", "SearXNG (self-hosted)"], value="Serper (Google)", label="Searcher"
    )
    serper_key = mo.ui.text(value=os.environ.get("SERPER_API_KEY", ""), kind="password", label="Serper API key (from .env if set)")
    searxng_url = mo.ui.text(value="http://localhost:8888", label="SearXNG URL")
    sources_choice = mo.ui.dropdown(options=["Any site", "Official sites only (.gov .edu .int)"], value="Any site", label="Sources")
    crawler_choice = mo.ui.dropdown(
        options=["Browser (Crawl4AI)", "HTTPX (fast, no JavaScript)", "HTTPX, then browser if needed"],
        value="Browser (Crawl4AI)",
        label="Crawler",
    )
    judge_choice = mo.ui.dropdown(
        options=["Laya (local, default)", "GLiNER2.5-decide (ONNX, CPU)", "LLM (gpt-6-luna, most accurate)"],
        value="Laya (local, default)",
        label="Judge",
    )
    n_atoms = mo.ui.slider(1, 12, value=5, label="Max claims (n_atoms)")
    top_k = mo.ui.slider(1, 10, value=5, label="Search results per claim (top_k)")
    mo.vstack(
        [
            mo.md("## Build a pipeline"),
            filter_choice,
            mo.hstack([searcher_choice, sources_choice], justify="start"),
            serper_key,
            searxng_url,
            mo.hstack([crawler_choice, judge_choice], justify="start"),
            mo.hstack([n_atoms, top_k], justify="start"),
        ]
    )
    return (
        crawler_choice, filter_choice, judge_choice, n_atoms, searcher_choice, searxng_url, serper_key, sources_choice, top_k,
    )


@app.cell
def _(crawler_choice, filter_choice, judge_choice, mo, n_atoms, searcher_choice, sources_choice, top_k):
    # The Python for the current selection: exactly what the "Fact-check" button builds.
    _imports = {"FactAssessor", "LLMAtomizer", "Take", "not_blocked"}
    _pre = []

    _filter = {
        "Laya (local, default)": ("LayaClaimFilter", "LayaClaimFilter(threshold=0.4)"),
        "GLiNER2.5-decide (ONNX, CPU)": ("GlinerClaimFilter", 'GlinerClaimFilter(model="2.5-decide", threshold=0.4)'),
        "None (check every atom)": (None, "None"),
    }[filter_choice.value]
    if _filter[0]:
        _imports.add(_filter[0])

    _search = {
        "Serper (Google)": ("SerperSearcher", "SerperSearcher(num={num})"),
        "DuckDuckGo (no key)": ("DuckDuckGoSearcher", "DuckDuckGoSearcher(num={num})"),
        "SearXNG (self-hosted)": ("SearxngSearcher", 'SearxngSearcher("http://localhost:8888", num={num})'),
    }[searcher_choice.value]
    _imports.add(_search[0])
    _keep = "not_blocked()"
    if sources_choice.value.startswith("Official"):
        _imports.add("Pred")
        _pre += ['official = Pred(lambda hit: urlparse(hit["url"]).netloc.endswith((".gov", ".edu", ".int")))', ""]
        _keep = "(not_blocked() & official)"

    _crawl = {
        "Browser (Crawl4AI)": ({"Crawl4AICrawler"}, "Crawl4AICrawler(timeout=2.5)"),
        "HTTPX (fast, no JavaScript)": ({"HTTPXCrawler"}, "HTTPXCrawler(timeout=2.5)"),
        "HTTPX, then browser if needed": (
            {"HTTPXCrawler", "Crawl4AICrawler", "FallbackCrawler"},
            "FallbackCrawler(HTTPXCrawler(timeout=2.5), Crawl4AICrawler(timeout=2.5))",
        ),
    }[crawler_choice.value]
    _imports |= _crawl[0]

    _judge = {
        "Laya (local, default)": ("LayaJudge", "LayaJudge()"),
        "GLiNER2.5-decide (ONNX, CPU)": ("GlinerJudge", 'GlinerJudge(model="2.5-decide")'),
        "LLM (gpt-6-luna, most accurate)": ("LLMJudge", 'LLMJudge("openai:gpt-6-luna")'),
    }[judge_choice.value]
    _imports.add(_judge[0])

    _lines = _pre + [
        "fa = FactAssessor(",
        "    atomizer=LLMAtomizer(),",
        f"    claim_filter={_filter[1]},",
        f"    searcher={_search[1].format(num=2 * top_k.value)} >> {_keep} >> Take({top_k.value}),",
        f"    crawler={_crawl[1]},",
        f"    judge={_judge[1]},",
        f"    n_atoms={n_atoms.value},",
        ")",
        "",
        "result = await fa.assess(text)          # or: async for event in fa.stream(text): ...",
    ]
    code = (
        ("from urllib.parse import urlparse\n" if sources_choice.value.startswith("Official") else "")
        + f"from factassessor import {', '.join(sorted(_imports, key=str.lower))}\n\n"
        + "\n".join(_lines)
    )
    mo.md(
        f"**The code for this pipeline**\n\n```python\n{code}\n```\n\n"
        "Model-backed parts share their models automatically: the Laya filter and Laya judge use one copy of Laya."
    )
    return


@app.cell
def _():
    from urllib.parse import urlparse

    from factassessor import (
        Crawl4AICrawler,
        DuckDuckGoSearcher,
        FactAssessor,
        FallbackCrawler,
        GlinerClaimFilter,
        GlinerJudge,
        HTTPXCrawler,
        LayaClaimFilter,
        LayaJudge,
        LLMAtomizer,
        LLMJudge,
        Pred,
        SearxngSearcher,
        SerperSearcher,
        Take,
        not_blocked,
    )

    # Components built once and reused across every combination you try (models are shared inside anyway;
    # this also keeps one browser and one HTTP pool).
    browser = Crawl4AICrawler(timeout=2.5)
    fast_fetch = HTTPXCrawler(timeout=2.5)
    official = Pred(lambda hit: urlparse(hit["url"]).netloc.endswith((".gov", ".edu", ".int")))
    _cache = {}

    def component(kind, name):
        """The filter or judge for a dropdown choice, built once."""
        if (kind, name) not in _cache:
            _cache[(kind, name)] = {
                ("filter", "Laya (local, default)"): lambda: LayaClaimFilter(threshold=0.4),
                ("filter", "GLiNER2.5-decide (ONNX, CPU)"): lambda: GlinerClaimFilter(threshold=0.4),
                ("filter", "None (check every atom)"): lambda: None,
                ("judge", "Laya (local, default)"): lambda: LayaJudge(),
                ("judge", "GLiNER2.5-decide (ONNX, CPU)"): lambda: GlinerJudge(),
                ("judge", "LLM (gpt-6-luna, most accurate)"): lambda: LLMJudge("openai:gpt-6-luna"),
            }[(kind, name)]()
        return _cache[(kind, name)]

    def build(filter_name, searcher_name, key, searxng, sources, crawler_name, judge_name, n, k):
        search = {
            "Serper (Google)": lambda: SerperSearcher(api_key=key or None, num=2 * k),
            "DuckDuckGo (no key)": lambda: DuckDuckGoSearcher(num=2 * k),
            "SearXNG (self-hosted)": lambda: SearxngSearcher(searxng, num=2 * k),
        }[searcher_name]()
        keep = (not_blocked() & official) if sources.startswith("Official") else not_blocked()
        crawler = {
            "Browser (Crawl4AI)": browser,
            "HTTPX (fast, no JavaScript)": fast_fetch,
            "HTTPX, then browser if needed": FallbackCrawler(fast_fetch, browser),
        }[crawler_name]
        return FactAssessor(
            atomizer=LLMAtomizer(),
            claim_filter=component("filter", filter_name),
            searcher=search >> keep >> Take(k),
            crawler=crawler,
            judge=component("judge", judge_name),
            n_atoms=n,
        )

    return build, component


@app.cell
async def _(component, filter_choice, judge_choice, mo):
    import time as _time

    # Auto warm-up: load the picked filter's and judge's models now (re-runs when you change a dropdown), so the
    # first fact-check doesn't wait for them. Models are shared, so warming Laya once covers filter and judge.
    _t = _time.perf_counter()
    _parts = [component("filter", filter_choice.value), component("judge", judge_choice.value)]
    with mo.status.spinner(title="Loading models…"):
        for _part in _parts:
            if _part is not None:
                await (_part.start() if hasattr(_part, "start") else _part.aload())
    mo.md(f"✅ Filter **{filter_choice.value}** and judge **{judge_choice.value}** ready ({_time.perf_counter() - _t:.1f}s)")
    return


@app.cell
def _(mo):
    text_box = mo.ui.text_area(
        value=(
            "It was believed that Nepal's earthquake in 2017 of 7.8 magnitude scale caused massive damage. "
            "Total lives lost were 1 million people."
        ),
        label="Text to fact-check",
        full_width=True,
        rows=3,
    )
    run_button = mo.ui.run_button(label="Fact-check")
    mo.vstack([mo.md("## Fact-check"), text_box, run_button])
    return run_button, text_box


@app.cell
async def _(build, crawler_choice, filter_choice, judge_choice, mo, n_atoms, run_button, searcher_choice, searxng_url, serper_key, sources_choice, text_box, top_k):
    mo.stop(not run_button.value, mo.md("*Pick the parts above, then press **Fact-check**.*"))
    if searcher_choice.value.startswith("Serper") and not serper_key.value:
        mo.stop(True, mo.md("⚠️ Serper needs an API key: paste it above, add `SERPER_API_KEY` to `.env`, or pick DuckDuckGo."))
    fa = build(
        filter_choice.value, searcher_choice.value, serper_key.value, searxng_url.value, sources_choice.value,
        crawler_choice.value, judge_choice.value, n_atoms.value, top_k.value,
    )
    # stream(): each claim shows up as "checking…" as soon as it's found and flips to its verdict the moment it
    # settles; `await fa.assess(text)` returns the same final result in one go.
    _rows = {}
    result = None
    async for _event in fa.stream(text_box.value):
        if _event.type == "claim_found":
            _rows[_event.atom.id] = {"verdict": "⏳ checking…", "confidence": None, "claim": _event.atom.text}
        elif _event.type == "claim_verified":
            _r = _event.result
            _rows[_r.atom.id] = {
                "verdict": _r.verdict, "confidence": round(_r.confidence, 2), "claim": _r.atom.text, "error": _r.error or "",
            }
        else:
            result = _event.result
        mo.output.replace(mo.ui.table([_rows[k] for k in sorted(_rows)], selection=None))
    return (result,)


@app.cell
def _(mo, result):
    from factassessor import kg

    VERDICT_STYLE = kg.VERDICT_STYLE
    graph = kg.build(result)  # built on demand from the result (~0.1ms); not part of the pipeline
    _counts = {v: sum(a.verdict == v for a in result.atoms) for v in VERDICT_STYLE}
    _score = "n/a" if result.fact_score is None else f"{result.fact_score:.0%}"
    mo.vstack(
        [
            mo.hstack(
                [
                    mo.stat(value=_score, label="Fact score", caption="supported / decided atoms"),
                    *[mo.stat(value=str(c), label=f"{VERDICT_STYLE[v][0]} {v}") for v, c in _counts.items()],
                    mo.stat(value=f"{result.latency_ms / 1000:.1f}s", label="Latency"),
                ]
            ),
            mo.md(
                "### Knowledge graph\n"
                "sentences of your text → the claims in them (coloured by verdict) ← the evidence passages that "
                "support (thick) or refute (dotted) them, with the judge's probability; claims from one sentence are linked."
            ),
            mo.mermaid(kg.to_mermaid(graph)),
            mo.md("### Evidence"),
            mo.ui.table(
                [
                    {"claim": a.atom.text, "label": e.label, "prob": round(e.prob, 2), "from": e.source, "url": e.url, "passage": e.text}
                    for a in result.atoms
                    for e in sorted(a.evidence, key=lambda e: -e.prob)
                ],
                selection=None,
            ),
            mo.md(
                "**Skipped** (not factual claims): "
                + (", ".join(f"*{s.text}* ({s.claim_score:.2f})" for s in result.skipped) or "none")
            ),
        ]
    )
    return


@app.cell
def _(mo):
    mo.md("""
    ## Recipes: ways to set up a pipeline

    **1. Defaults.** Everything built for you (LLM atomizer, Laya claim filter, Serper, browser crawler, Laya judge):

    ```python
    from factassessor import FactAssessor

    async with FactAssessor() as fa:
        result = await fa.assess(text)        # or fa.assess_sync(text) in non-async code
    ```

    **2. Tune the defaults with arguments.** No composition needed:

    ```python
    FactAssessor(n_atoms=8, top_k=3, claim_threshold=0.5, crawl_timeout=2.0, search_hedge_after=1.0)
    ```

    **3. Swap components.** Pass any implementation of a role; everything else stays default. Model-backed
    components share their models automatically:

    ```python
    FactAssessor(judge=LLMJudge("openai:gpt-6-luna"))                     # most accurate judge
    FactAssessor(claim_filter=GlinerClaimFilter(), judge=GlinerJudge())   # GLiNER for both (one shared model)
    FactAssessor(crawler=FallbackCrawler(HTTPXCrawler(), Crawl4AICrawler()))   # fast fetch, browser if needed
    ```

    **4. Chain steps with `>>`.** A condition (`Pred`) filters, a plain function transforms, `Take(n)` caps:

    ```python
    is_forum = Pred(lambda hit: "reddit.com" in hit["url"] or "quora.com" in hit["url"])

    FactAssessor(
        claim_filter=LayaClaimFilter(threshold=0.5) >> Pred(lambda atom: len(atom.text) > 15),
        searcher=SerperSearcher(num=20) >> (not_blocked() & ~is_forum) >> Take(5),
    )
    ```

    **5. No search API key.** DuckDuckGo (`fact-assessor[ddg]`) or your own SearXNG:

    ```python
    FactAssessor(searcher=DuckDuckGoSearcher() >> not_blocked() >> Take(5))
    FactAssessor(searcher=SearxngSearcher("http://localhost:8888") >> not_blocked() >> Take(5))
    ```

    **6. Write your own component.** Subclass the role and implement its one method:

    ```python
    from factassessor import ClaimFilter, Crawler

    class LengthFilter(ClaimFilter):
        async def score(self, atom):
            return 1.0 if len(atom.text) > 15 else 0.0      # P(factual claim)

    class MyCrawler(Crawler):
        async def crawl(self, url):
            ...   # return {"url", "title", "text"}, or None if it failed

    FactAssessor(claim_filter=LengthFilter(), crawler=MyCrawler())
    ```

    | Role | Implement | Built in |
    |---|---|---|
    | `Atomizer` | `atomize(text)` | `LLMAtomizer` |
    | `ClaimFilter` | `score(atom) -> P(claim)` | `LayaClaimFilter`, `GlinerClaimFilter` |
    | `Searcher` | `search(query)` | `SerperSearcher`, `DuckDuckGoSearcher`, `SearxngSearcher` |
    | `Crawler` | `crawl(url)` | `Crawl4AICrawler`, `HTTPXCrawler`, `FallbackCrawler` |
    | `Judge` | `judge(claim, docs)` | `LayaJudge`, `GlinerJudge`, `LLMJudge` |
    | `Policy` | `settled(ev)`, `verdict(ev)` | `WeightedPolicy` |
    """)
    return


if __name__ == "__main__":
    app.run()
