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
    import marimo as mo

    return (mo,)


@app.cell
def _(mo):
    mo.md("""
    # FactAssessor

    text → atoms (one fact each) → filter → search → crawl → judge → verdicts, fact score, knowledge graph.

    Below are **ways to build the pipeline**, from the one-liner to your own chains. Pick one, look at its code,
    and fact-check some text with it. For how the pieces work, see `fa_walk.py`.
    """)
    return


@app.cell
def _():
    from urllib.parse import urlparse

    from factassessor import (
        Crawl4AICrawler,
        DuckDuckGoSearcher,
        FactAssessor,
        LayaCheckworthy,
        LayaRunner,
        LLMAtomizer,
        Pred,
        SerperSearcher,
        Take,
        not_blocked,
    )

    # Shared by every pipeline below, so switching pipelines doesn't load another Laya model or browser.
    laya = LayaRunner()
    crawler = Crawl4AICrawler(timeout=2.5)

    official = Pred(lambda hit: urlparse(hit["url"]).netloc.endswith((".gov", ".edu", ".int")))
    is_forum = Pred(lambda hit: any(site in hit["url"] for site in ("reddit.com", "quora.com", "stackexchange.com")))
    long_enough = Pred(lambda atom: len(atom.text) > 15)

    def gliner_judge():
        from factassessor.gliner import GlinerJudge  # needs fact-assessor[gliner]; ~1.75 GB on first use

        return GlinerJudge()

    PIPELINES = {
        "1. Default": (
            "FactAssessor()",
            lambda: FactAssessor(laya=laya, crawler=crawler),
        ),
        "2. Tuned with arguments": (
            "FactAssessor(n_atoms=8, top_k=3, checkworthy_threshold=0.5, crawl_timeout=2.0)",
            lambda: FactAssessor(n_atoms=8, top_k=3, checkworthy_threshold=0.5, laya=laya, crawler=crawler),
        ),
        "3. Chained steps": (
            '''long_enough = Pred(lambda atom: len(atom.text) > 15)
is_forum    = Pred(lambda hit: "reddit.com" in hit["url"] or ...)

FactAssessor(
    atomizer=LLMAtomizer() >> LayaCheckworthy(laya, threshold=0.5) >> long_enough >> Take(8),
    searcher=SerperSearcher(num=20) >> (not_blocked() & ~is_forum) >> Take(5),
)''',
            lambda: FactAssessor(
                laya=laya,
                crawler=crawler,
                atomizer=LLMAtomizer() >> LayaCheckworthy(laya, threshold=0.5) >> long_enough >> Take(8),
                searcher=SerperSearcher(num=20) >> (not_blocked() & ~is_forum) >> Take(5),
            ),
        ),
        "4. Official sources only": (
            '''official = Pred(lambda hit: urlparse(hit["url"]).netloc.endswith((".gov", ".edu", ".int")))

FactAssessor(searcher=SerperSearcher(num=20) >> (not_blocked() & official) >> Take(5))''',
            lambda: FactAssessor(
                laya=laya, crawler=crawler, searcher=SerperSearcher(num=20) >> (not_blocked() & official) >> Take(5)
            ),
        ),
        "5. No search API key (DuckDuckGo)": (
            "FactAssessor(searcher=DuckDuckGoSearcher() >> not_blocked() >> Take(5))   # fact-assessor[ddg]",
            lambda: FactAssessor(laya=laya, crawler=crawler, searcher=DuckDuckGoSearcher() >> not_blocked() >> Take(5)),
        ),
        "6. GLiNER2.5-decide judge": (
            "from factassessor.gliner import GlinerJudge\n\nFactAssessor(judge=GlinerJudge())   # fact-assessor[gliner], CPU, ONNX",
            lambda: FactAssessor(laya=laya, crawler=crawler, judge=gliner_judge()),
        ),
    }
    return PIPELINES, crawler, laya


@app.cell
def _(PIPELINES, mo):
    choice = mo.ui.dropdown(options=list(PIPELINES), value="1. Default", label="Pipeline")
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
    mo.vstack([choice, text_box, run_button])
    return choice, run_button, text_box


@app.cell
def _(PIPELINES, choice, mo):
    mo.md(f"**{choice.value}**\n\n```python\n{PIPELINES[choice.value][0]}\n```")
    return


@app.cell
def _():
    built = {}  # one assessor per pipeline, built the first time it's picked
    return (built,)


@app.cell
async def _(PIPELINES, built, choice, mo, run_button, text_box):
    mo.stop(not run_button.value, mo.md("*Press **Fact-check** to run.*"))
    if choice.value not in built:
        built[choice.value] = PIPELINES[choice.value][1]()
    fa = built[choice.value]
    # stream(): each claim shows up as "checking…" as soon as it's found and flips to its verdict the moment it
    # settles; `await fa.assess(text)` would return the same final result in one go.
    _rows = {}
    result = None
    async for _event in fa.stream(text_box.value):
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
            mo.md("### Atoms"),
            mo.ui.table(
                [
                    {
                        "verdict": f"{VERDICT_STYLE[a.verdict][0]} {a.verdict}",
                        "confidence": round(a.confidence, 2),
                        "atom": a.atom.text,
                        "evidence": len(a.evidence),
                        "from_pages": sum(e.source == "page" for e in a.evidence),
                        "error": a.error or "",
                    }
                    for a in result.atoms
                ]
                + [
                    {"verdict": "⏭️ skipped", "confidence": None, "atom": s.text, "evidence": 0, "from_pages": 0, "error": ""}
                    for s in result.skipped
                ],
                selection=None,
            ),
            mo.md("### Evidence"),
            mo.ui.table(
                [
                    {"atom": a.atom.text, "label": e.label, "prob": round(e.prob, 2), "source": e.source, "url": e.url, "passage": e.text}
                    for a in result.atoms
                    for e in sorted(a.evidence, key=lambda e: -e.prob)
                ],
                selection=None,
            ),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
