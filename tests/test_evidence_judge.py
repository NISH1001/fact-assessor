from types import SimpleNamespace

from factassessor.evidence_judge import LayaJudge
from tests.test_passages import WordTokenizer

CLAIM = "Marie Curie won the Nobel Prize in Physics in 1903."


class FakeLaya:
    """Says 'supports' for evidence mentioning 1903, 'not_enough_info' otherwise."""

    def __init__(self, budget=64):
        self.budget = budget
        self.batches = []

    async def agent(self, model):
        # room = max_len - head_max_len - 16; the judge subtracts the claim's tokens itself
        claim_tokens = len(CLAIM.split())
        return SimpleNamespace(tok=WordTokenizer(), cfg={"max_len": self.budget + claim_tokens + 16, "head_max_len": 0})

    async def predict_batch(self, requests):
        self.batches.append(requests)
        out = []
        for r in requests:
            s = 0.9 if "1903" in r["state"]["evidence"] else 0.1
            out.append({"answers": {"stance": {"probabilities": {"supports": s, "refutes": 0.05, "not_enough_info": 0.95 - s}}}})
        return out


async def test_snippets_are_judged_as_is_in_one_batch():
    laya = FakeLaya()
    ev = await LayaJudge(runner=laya).judge(CLAIM, [
        {"url": "u1", "title": "Nobel", "snippet": "Curie shared the 1903 Nobel Prize in Physics."},
        {"url": "u2", "title": "Paris", "snippet": "Paris is in France."},
        {"url": "u3", "title": "Empty", "snippet": ""},
    ])
    assert [(e.url, e.source, e.label, e.prob) for e in ev] == [
        ("u1", "snippet", "supports", 0.9),
        ("u2", "snippet", "not_enough_info", 0.85),
    ]
    assert len(laya.batches) == 1
    assert list(laya.batches[0][0]["state"]) == ["evidence", "claim"]  # evidence first: 13/15 vs 9/15


async def test_pages_are_cut_to_top_passages_within_the_token_budget():
    filler = " ".join(f"filler{i}" for i in range(2000))
    page = {"url": "wiki", "title": "Marie Curie", "text": f"{filler} Curie won the Nobel Prize in Physics in 1903. {filler}"}
    laya = FakeLaya(budget=64)
    ev = await LayaJudge(runner=laya, passages_per_page=2).judge(CLAIM, [page])
    assert len(ev) == 2 and all(e.source == "page" for e in ev)
    assert ev[0].label == "supports" and "1903" in ev[0].text  # BM25 put the relevant chunk first
    assert all(len(r["state"]["evidence"].split()) <= 64 for r in laya.batches[0])


async def test_no_docs_gives_no_evidence():
    assert await LayaJudge(runner=FakeLaya()).judge(CLAIM, []) == []
