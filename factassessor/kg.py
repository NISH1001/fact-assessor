"""Knowledge graph of a CheckResult: which text backs which claim, and how claims relate. Built on demand.

    sentence ──contains──► claim ◄──supports / refutes (prob)── passage ◄──published── source
    claim ──same_sentence── claim

- sentence: a sentence of the input text (claims point back to it through their span)
- claim:    an atom, with its verdict and confidence
- passage:  the evidence text a judge decided on; one node even when several claims used it
- source:   the site it came from (en.wikipedia.org)

It's a view of the result, not a pipeline step: `kg.build(result)` takes ~0.1ms, so build it when you show it.
"""

from __future__ import annotations

from itertools import combinations
from typing import Any
from urllib.parse import urlparse

from factassessor.atomizer import _sentences
from factassessor.schema import CheckResult

VERDICT_STYLE = {
    "supported": ("✅", "fill:#d1fadf,stroke:#12b76a,color:#054f31"),
    "refuted": ("❌", "fill:#fee4e2,stroke:#f04438,color:#7a271a"),
    "contested": ("⚠️", "fill:#fef0c7,stroke:#f79009,color:#7a2e0e"),
    "unverified": ("❔", "fill:#f2f4f7,stroke:#98a2b3,color:#344054"),
}


def build(result: CheckResult, strong: float = 0.7, include_not_enough_info: bool = False) -> dict[str, Any]:
    """Nodes and edges as plain dicts (JSON-ready). Evidence below `strong` is left out, as is not_enough_info
    unless `include_not_enough_info`."""
    nodes: dict[str, dict[str, Any]] = {}
    edges: dict[tuple[str, str, str], float] = {}

    sentence_spans = _sentences(result.text) or [(0, len(result.text))]
    for i, (start, end) in enumerate(sentence_spans):
        nodes[f"sentence:{i}"] = {"id": f"sentence:{i}", "kind": "sentence", "label": result.text[start:end], "span": (start, end)}

    claims_in: dict[str, list[str]] = {}
    passage_ids: dict[tuple[str, str], str] = {}  # (url, text) -> node id: one node per passage, however many claims
    for r in result.atoms:
        claim_id = f"claim:{r.atom.id}"
        nodes[claim_id] = {
            "id": claim_id, "kind": "claim", "label": r.atom.text,
            "verdict": r.verdict, "confidence": r.confidence, "span": r.atom.span,
        }
        sentence_id = f"sentence:{_sentence_of(r.atom.span, sentence_spans)}"
        edges[(sentence_id, claim_id, "contains")] = 1.0
        claims_in.setdefault(sentence_id, []).append(claim_id)

        for e in r.evidence:
            if e.prob < strong or (e.label == "not_enough_info" and not include_not_enough_info):
                continue
            passage_id = passage_ids.setdefault((e.url, e.text), f"passage:{len(passage_ids)}")
            site = urlparse(e.url).netloc.removeprefix("www.") or e.url
            nodes.setdefault(passage_id, {"id": passage_id, "kind": "passage", "label": _excerpt(e.text), "text": e.text,
                                          "url": e.url, "site": site, "from": e.source})
            source_id = f"source:{site}"
            nodes.setdefault(source_id, {"id": source_id, "kind": "source", "label": site})
            edges[(source_id, passage_id, "published")] = 1.0
            edge = (passage_id, claim_id, e.label)
            edges[edge] = max(edges.get(edge, 0.0), round(e.prob, 3))

    for claim_ids in claims_in.values():
        for a, b in combinations(claim_ids, 2):
            edges[(a, b, "same_sentence")] = 1.0

    return {
        "nodes": list(nodes.values()),
        "edges": [{"source": s, "target": t, "relation": rel, "weight": w} for (s, t, rel), w in edges.items()],
    }


def to_mermaid(graph: dict[str, Any]) -> str:
    """Mermaid flowchart: sentences → claims (coloured by verdict) ← passages (labelled with their site).
    Source nodes are folded into the passage labels to keep the picture readable."""
    ids = {n["id"]: f"n{i}" for i, n in enumerate(graph["nodes"])}
    lines = ["graph LR"]
    for n in graph["nodes"]:
        label = _quote(n["label"])
        if n["kind"] == "claim":
            lines.append(f'  {ids[n["id"]]}["{VERDICT_STYLE[n["verdict"]][0]} {label}"]:::{n["verdict"]}')
        elif n["kind"] == "sentence":
            lines.append(f'  {ids[n["id"]]}(["{label}"]):::sentence')
        elif n["kind"] == "passage":
            lines.append(f'  {ids[n["id"]]}["{_quote(n["site"])}: {label}"]:::passage')
    for e in graph["edges"]:
        if e["relation"] == "published":
            continue
        a, b = ids[e["source"]], ids[e["target"]]
        if e["relation"] == "contains":
            lines.append(f"  {a} --- {b}")
        elif e["relation"] == "same_sentence":
            lines.append(f"  {a} -.- {b}")
        else:
            arrow = "==>" if e["relation"] == "supports" else "-.->"
            lines.append(f'  {a} {arrow}|"{e["relation"]} {e["weight"]:.2f}"| {b}')
    lines += [f"  classDef {v} {style}" for v, (_, style) in VERDICT_STYLE.items()]
    lines.append("  classDef sentence fill:#f9fafb,stroke:#d0d5dd,color:#344054")
    lines.append("  classDef passage fill:#eef4ff,stroke:#6172f3,color:#1d2939")
    return "\n".join(lines)


def _sentence_of(span: tuple[int, int], sentences: list[tuple[int, int]]) -> int:
    start = span[0]
    for i, (s, e) in enumerate(sentences):
        if s <= start < e:
            return i
    return max(range(len(sentences)), key=lambda i: sentences[i][0] <= start)  # nearest preceding sentence


def _excerpt(text: str, limit: int = 90) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1].rsplit(" ", 1)[0] + "…"


def _quote(text: str) -> str:
    return text.replace('"', "#quot;")
