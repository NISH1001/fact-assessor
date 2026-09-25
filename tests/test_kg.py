from factassessor import Atom, AtomResult, CheckResult, Evidence, kg

TEXT = "Nepal's earthquake in 2017 of 7.8 magnitude caused massive damage. Total lives lost were 1 million people."
#       0                                                          65   67                                   106


def ev(label, prob, url="https://en.wikipedia.org/wiki/April_2015_Nepal_earthquake", text="The 2015 Nepal earthquake..."):
    return Evidence(url=url, title="t", text=text, source="page", label=label, prob=prob)


SHARED = "A magnitude 7.8 earthquake struck Nepal on 25 April 2015, killing nearly 9,000 people."
RESULT = CheckResult(
    text=TEXT,
    atoms=[
        AtomResult(atom=Atom(id=0, text="Nepal's earthquake occurred in 2017.", span=(0, 26)), verdict="refuted",
                   confidence=0.9, evidence=[ev("refutes", 0.92, text=SHARED), ev("not_enough_info", 0.8, text="Unrelated.")]),
        AtomResult(atom=Atom(id=1, text="The Nepal earthquake had a magnitude of 7.8.", span=(27, 43)), verdict="supported",
                   confidence=0.95, evidence=[ev("supports", 0.95, text=SHARED), ev("supports", 0.5, url="https://weak.org/x")]),
        AtomResult(atom=Atom(id=2, text="The Nepal earthquake killed 1 million people.", span=(67, 106)), verdict="refuted",
                   confidence=0.9, evidence=[ev("refutes", 0.88, url="https://www.ngdc.noaa.gov/x", text="Deaths: 8,964.")]),
    ],
    fact_score=1 / 3,
    latency_ms=5000,
)


def by_kind(graph, kind):
    return [n for n in graph["nodes"] if n["kind"] == kind]


def edges(graph, relation):
    return [(e["source"], e["target"]) for e in graph["edges"] if e["relation"] == relation]


def test_sentences_contain_their_claims():
    g = kg.build(RESULT)
    assert [s["label"] for s in by_kind(g, "sentence")] == [
        "Nepal's earthquake in 2017 of 7.8 magnitude caused massive damage.",
        "Total lives lost were 1 million people.",
    ]
    assert sorted(edges(g, "contains")) == [("sentence:0", "claim:0"), ("sentence:0", "claim:1"), ("sentence:1", "claim:2")]


def test_claims_from_the_same_sentence_are_linked():
    assert edges(kg.build(RESULT), "same_sentence") == [("claim:0", "claim:1")]


def test_passages_link_to_claims_with_label_and_probability():
    g = kg.build(RESULT)
    stance = {(e["source"], e["target"], e["relation"]): e["weight"] for e in g["edges"] if e["relation"] in ("supports", "refutes")}
    shared = next(p["id"] for p in by_kind(g, "passage") if p["text"] == SHARED)
    assert stance[(shared, "claim:0", "refutes")] == 0.92
    assert stance[(shared, "claim:1", "supports")] == 0.95  # one passage node, two claims: shared evidence is visible
    assert len([p for p in by_kind(g, "passage") if p["text"] == SHARED]) == 1


def test_weak_and_not_enough_info_evidence_is_left_out_by_default():
    g = kg.build(RESULT)
    assert not any("weak.org" in p["url"] for p in by_kind(g, "passage"))
    assert edges(g, "not_enough_info") == []
    assert edges(kg.build(RESULT, include_not_enough_info=True), "not_enough_info")


def test_sources_publish_passages():
    g = kg.build(RESULT)
    assert {s["label"] for s in by_kind(g, "source")} == {"en.wikipedia.org", "ngdc.noaa.gov"}
    assert all(src.startswith("source:") and dst.startswith("passage:") for src, dst in edges(g, "published"))


def test_claim_nodes_carry_verdict_confidence_and_span():
    claim = next(c for c in by_kind(kg.build(RESULT), "claim") if c["id"] == "claim:2")
    assert (claim["verdict"], claim["confidence"], claim["span"]) == ("refuted", 0.9, (67, 106))


def test_mermaid_renders_every_claim_and_edge_label():
    text = kg.to_mermaid(kg.build(RESULT))
    assert text.startswith("graph LR")
    assert "Nepal's earthquake occurred in 2017." in text and "refutes 0.92" in text and "supports 0.95" in text


def test_mermaid_escapes_quotes_in_labels():
    quoted = RESULT.model_copy(deep=True)
    quoted.atoms[0].atom.text = 'The report said "2017".'
    text = kg.to_mermaid(kg.build(quoted))
    assert 'said "2017"' not in text and "said #quot;2017#quot;" in text
