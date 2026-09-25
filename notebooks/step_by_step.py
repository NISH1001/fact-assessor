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
def _():
    from factassessor import Atomizer, FactAssessor, LayaCheckworthy, collect, once

    return Atomizer, FactAssessor, LayaCheckworthy, collect, once


@app.cell
async def _(FactAssessor, time):
    # One assessor for the whole notebook: shares the Laya model, HTTP pool, and browser across every step.
    fa = FactAssessor(n_atoms=8, top_k=5)  # atoms are single facts now, so a sentence can yield several
    _t = time.perf_counter()
    await fa.aload()  # warm spaCy, Laya, and the browser up front
    print(f"warmed up in {time.perf_counter() - _t:.1f}s")
    return (fa,)


@app.cell
def _(mo):
    mo.md("""
    # FactAssessor, step by step
    text → atoms (one fact each, self-contained) → filter → search → crawl → judge → knowledge graph + fact score.
    Jump to **Run everything** at the bottom for the one-shot version.
    """)
    return


@app.cell
def _():
    text = (
        "It was believed that Nepal's earthquake in 2017 of 7.8 magnitude scale caused massive damage. "
        "Total lives lost were 1 million people."
    )
    return (text,)


@app.cell
def _(mo):
    mo.md("""
    ## Step 1: Atomize
    One LLM call (gpt-5.6-luna, reasoning off) splits the text into atomic claims, one fact each, and makes each
    self-contained: pronouns and implicit references resolved, hedges like "It was believed that" unwrapped,
    values kept exactly as written. Each atom keeps the span of the words it came from, for UI highlighting.
    """)
    return


@app.cell
async def _(Atomizer, mo, text, time):
    _t = time.perf_counter()
    atoms = await Atomizer().aatomize(text)
    print(f"{len(atoms)} atoms in {(time.perf_counter() - _t) * 1000:.0f} ms")
    mo.ui.table([{"atom": a.text, "from": text[a.span[0] : a.span[1]]} for a in atoms])
    return (atoms,)


@app.cell
def _(mo):
    mo.md("""
    ## Step 2: Filter
    `LayaCheckworthy` (local Laya model) scores each atom; only factual claims (P ≥ 0.4) go on. In the pipeline
    it's chained right after the atomizer: `Atomizer() >> LayaCheckworthy() >> Take(n_atoms)`.
    """)
    return


@app.cell
async def _(LayaCheckworthy, asyncio, atoms, fa, mo, time):
    _t = time.perf_counter()
    _threshold = 0.4  # FactAssessor's default checkworthy_threshold
    # score every atom (threshold 0 keeps them all) so the table can show what was dropped and why
    _scored = await asyncio.gather(*(LayaCheckworthy(fa.laya, threshold=0.0).score(a) for a in atoms))
    kept = [a for a in _scored if a.checkworthiness >= _threshold]
    skipped = [a for a in _scored if a.checkworthiness < _threshold]
    print(f"filtered {len(atoms)} atoms in {(time.perf_counter() - _t) * 1000:.0f} ms: {len(kept)} kept")
    mo.ui.table(
        [{"keep": True, "p_factual": round(a.checkworthiness, 2), "atom": a.text} for a in kept]
        + [{"keep": False, "p_factual": round(a.checkworthiness, 2), "atom": a.text} for a in skipped]
    )
    return (kept,)


@app.cell
def _(mo):
    mo.md("""
    ## Step 3: Search
    `fa.searcher` is the chain `Serper() >> Filter(not_blocked()) >> Take(top_k)`: Google results with social and
    video sites dropped. Every atom searched in parallel. Key from `SERPER_API_KEY` in `.env`.
    """)
    return


@app.cell
async def _(asyncio, collect, fa, kept, once, time):
    _t = time.perf_counter()
    hits = await asyncio.gather(*(collect(fa.searcher(once(a.text))) for a in kept))
    print(f"{len(kept)} searches in {(time.perf_counter() - _t) * 1000:.0f} ms")
    return (hits,)


@app.cell
def _(hits):
    hits[0]
    return


@app.cell
def _(hits, kept, mo):
    mo.ui.table(
        [
            {"atom": a.text, "rank": i + 1, "title": h["title"], "snippet": h["snippet"], "url": h["url"]}
            for a, atom_hits in zip(kept, hits)
            for i, h in enumerate(atom_hits)
        ]
    )
    return


@app.cell
def _(mo):
    mo.md("""
    ## Step 4: Crawl
    crawl4ai fetches every hit as clean plain text (no links, citations, or menus), all in parallel through
    one shared browser, each under a hard timeout. Failures are `None`; the snippet still counts.
    """)
    return


@app.cell
async def _(asyncio, fa, hits, time):
    _t = time.perf_counter()
    # fa.crawler is a Crawl4ai step; .crawl(url) fetches one page (None on failure), so the table can line up with hits
    pages = await asyncio.gather(*(asyncio.gather(*(fa.crawler.crawl(h["url"]) for h in atom_hits)) for atom_hits in hits))
    _n = sum(len(p) for p in pages)
    _ok = sum(page is not None for p in pages for page in p)
    print(f"crawled {_n} pages in {(time.perf_counter() - _t) * 1000:.0f} ms: {_ok} ok, {_n - _ok} failed")
    return (pages,)


@app.cell
def _(hits, kept, mo, pages):
    mo.ui.table(
        [
            {
                "atom": a.text,
                "url": h["url"],
                "ok": page is not None,
                "chars": len(page["text"]) if page else 0,
                "preview": page["text"][:200] if page else "",
            }
            for a, atom_hits, atom_pages in zip(kept, hits, pages)
            for h, page in zip(atom_hits, atom_pages)
        ]
    )
    return


@app.cell
def _(mo):
    mo.md("""
    ## Step 5: Judge
    Snippets are judged as-is; each page is chunked with Laya's own tokenizer to exactly fit its budget,
    and BM25 keeps the 3 most relevant chunks. Every (atom, passage) pair is one Laya decision:
    supports / refutes / not_enough_info. Pairs from **all** atoms go through one shared micro-batcher,
    so they share forward passes. Strong evidence (≥ 0.7) is then weighed into one verdict per atom.
    """)
    return


@app.cell
async def _(asyncio, fa, hits, kept, pages, time):
    _t = time.perf_counter()
    evidence = await asyncio.gather(
        *(
            fa.judge.ajudge(a.text, atom_hits + [p for p in atom_pages if p])
            for a, atom_hits, atom_pages in zip(kept, hits, pages)
        )
    )
    print(f"judged {sum(len(e) for e in evidence)} (atom, passage) pairs in {(time.perf_counter() - _t) * 1000:.0f} ms")
    return (evidence,)


@app.cell
def _(evidence, fa, kept, mo):
    _verdicts = [fa.policy.verdict(e) for e in evidence]
    mo.vstack(
        [
            mo.ui.table(
                [
                    {
                        "atom": a.text,
                        "verdict": v,
                        "confidence": round(c, 2),
                        "strong_evidence": sum(x.label != "not_enough_info" and x.prob >= fa.policy.strong for x in e),
                    }
                    for a, e, (v, c) in zip(kept, evidence, _verdicts)
                ]
            ),
            mo.ui.table(
                [
                    {"atom": a.text, "label": x.label, "prob": round(x.prob, 2), "source": x.source, "url": x.url, "passage": x.text}
                    for a, e in zip(kept, evidence)
                    for x in sorted(e, key=lambda x: -x.prob)
                ]
            ),
        ]
    )
    return


@app.cell
def _(mo):
    mo.md("""
    ---
    # Run everything
    One call: `fa.stream(text)` (or `await fa.assess(text)` for just the final result). Claims are checked the
    moment they're found; snippets are judged first; pages are only crawled when the snippets aren't
    conclusive, each page is judged the moment its own crawl finishes, and crawling stops once a claim settles.
    """)
    return


@app.cell
def _(mo):
    text_box = mo.ui.text_area(
        value=(
            "Sanjog is computer scientist in Google. He works in AKD project."
        ),
        label="Text to fact-check",
        full_width=True,
        rows=4,
    )
    run_button = mo.ui.run_button(label="Fact-check")
    mo.vstack([text_box, run_button])
    return run_button, text_box


@app.cell
async def _(fa, mo, run_button, text_box):
    mo.stop(not run_button.value, mo.md("*Press **Fact-check** to run.*"))
    # stream(): each claim shows up as "checking…" as soon as it's found and flips to its verdict the moment
    # it settles; assess() would return the same final result in one go.
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
def _():
    VERDICT_STYLE = {
        "supported": ("✅", "fill:#d1fadf,stroke:#12b76a,color:#054f31"),
        "refuted": ("❌", "fill:#fee4e2,stroke:#f04438,color:#7a271a"),
        "contested": ("⚠️", "fill:#fef0c7,stroke:#f79009,color:#7a2e0e"),
        "unverified": ("❔", "fill:#f2f4f7,stroke:#98a2b3,color:#344054"),
    }

    def to_mermaid(graph):
        """Knowledge graph -> mermaid: atoms coloured by verdict, sources linked by supports / refutes edges."""
        ids = {n["id"]: f"n{i}" for i, n in enumerate(graph["nodes"])}
        lines = ["graph LR"]
        for n in graph["nodes"]:
            label = n["label"].replace('"', "'")
            if n["kind"] == "atom":
                lines.append(f'  {ids[n["id"]]}["{VERDICT_STYLE[n["verdict"]][0]} {label}"]:::{n["verdict"]}')
            else:
                lines.append(f'  {ids[n["id"]]}(["{label}"]):::source')
        for e in graph["edges"]:
            arrow = "-->" if e["relation"] == "supports" else "-.->"
            lines.append(f'  {ids[e["source"]]} {arrow}|"{e["relation"]} {e["weight"]:.2f}"| {ids[e["target"]]}')
        lines += [f"  classDef {v} {style}" for v, (_, style) in VERDICT_STYLE.items()]
        lines.append("  classDef source fill:#eef4ff,stroke:#6172f3,color:#1d2939")
        return "\n".join(lines)

    return VERDICT_STYLE, to_mermaid


@app.cell
def _(VERDICT_STYLE, mo, result, to_mermaid):
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
            mo.md("### Knowledge graph"),
            mo.mermaid(to_mermaid(result.graph)),
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
                ]
            ),
        ]
    )
    return



if __name__ == "__main__":
    app.run()
