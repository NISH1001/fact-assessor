import marimo

__generated_with = "0.25.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import time

    import marimo as mo

    from factassessor import FactAssessor

    return FactAssessor, mo, time


@app.cell
async def _(FactAssessor, time):
    # One assessor for the session: keeps Laya, the HTTP pool, and the browser warm between checks.
    fa = FactAssessor(n_atoms=8, top_k=5)  # atoms are single facts now, so a sentence can yield several
    _t = time.perf_counter()
    await fa.aload()
    print(f"warmed up in {time.perf_counter() - _t:.1f}s")
    return (fa,)


@app.cell
def _(mo):
    mo.md("""
    # FactAssessor
    text → atoms (one fact each, self-contained) → filter → search → crawl → judge → knowledge graph + fact score
    """)
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
        rows=4,
    )
    run_button = mo.ui.run_button(label="Fact-check")
    mo.vstack([text_box, run_button])
    return run_button, text_box


@app.cell
async def _(fa, mo, run_button, text_box):
    mo.stop(not run_button.value, mo.md("*Press **Fact-check** to run.*"))
    result = await fa.acheck(text_box.value)
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
            mo.md("### Evidence"),
            mo.ui.table(
                [
                    {"atom": a.atom.text, "label": e.label, "prob": round(e.prob, 2), "source": e.source, "url": e.url, "passage": e.text}
                    for a in result.atoms
                    for e in sorted(a.evidence, key=lambda e: -e.prob)
                ]
            ),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
