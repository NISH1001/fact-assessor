"""Per-claim results -> fact score and knowledge graph."""

from __future__ import annotations

from typing import Any
from urllib.parse import urlparse

from factassessor.schema import AtomResult


def fact_score(results: list[AtomResult]) -> float | None:
    """supported / (supported + refuted + contested); None if nothing was decided."""
    decided = [r for r in results if r.verdict != "unverified"]
    return sum(r.verdict == "supported" for r in decided) / len(decided) if decided else None


def build_graph(results: list[AtomResult], strong: float = 0.7) -> dict[str, Any]:
    """Atom and source nodes; source -> atom edges for strong supports/refutes (max prob per pair).
    A site used by several claims is one node, so the graph shows which sources back which claims."""
    nodes: dict[str, dict[str, Any]] = {}
    edges: dict[tuple[str, str, str], float] = {}
    for r in results:
        atom_id = f"atom:{r.atom.id}"
        nodes[atom_id] = {"id": atom_id, "kind": "atom", "label": r.atom.text, "verdict": r.verdict}
        for e in r.evidence:
            if e.label == "not_enough_info" or e.prob < strong:
                continue
            source_id = f"source:{urlparse(e.url).netloc.removeprefix('www.') or e.url}"
            nodes.setdefault(source_id, {"id": source_id, "kind": "source", "label": source_id[7:], "url": e.url})
            key = (source_id, atom_id, e.label)
            edges[key] = max(edges.get(key, 0.0), e.prob)
    return {
        "nodes": list(nodes.values()),
        "edges": [{"source": s, "target": t, "relation": rel, "weight": round(w, 3)} for (s, t, rel), w in edges.items()],
    }
