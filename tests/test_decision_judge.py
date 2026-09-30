from factassessor import DecisionJudge, DecisionResponse, DecisionRunner
from factassessor.judges.decision import QUESTION

CLAIM = "Marie Curie won the Nobel Prize in Physics in 1903."


class FakeRunner:
    """Says 'supports' for evidence mentioning 1903, 'not_enough_info' otherwise; records every batch."""

    batch_size = 32

    def __init__(self):
        self.batches = []

    async def predict(self, requests):
        self.batches.append(requests)
        out = []
        for r in requests:
            s = 0.9 if "1903" in r.state["evidence"] else 0.1
            out.append(DecisionResponse(answers={"stance": {"probabilities": {"supports": s, "refutes": 0.05, "not_enough_info": 0.95 - s}}}))
        return out


async def test_snippets_are_judged_as_is_in_one_batch_with_the_stance_question():
    runner = FakeRunner()
    assert isinstance(runner, DecisionRunner)
    ev = await DecisionJudge(runner).judge(CLAIM, [
        {"url": "u1", "title": "Nobel", "snippet": "Curie shared the 1903 Nobel Prize in Physics."},
        {"url": "u2", "title": "Paris", "snippet": "Paris is in France."},
        {"url": "u3", "title": "Empty", "snippet": ""},
    ])
    assert [(e.url, e.source, e.label, e.prob) for e in ev] == [
        ("u1", "snippet", "supports", 0.9),
        ("u2", "snippet", "not_enough_info", 0.85),
    ]
    assert len(runner.batches) == 1
    first = runner.batches[0][0]
    assert list(first.state) == ["evidence", "claim"]  # evidence first: 13/15 vs 9/15
    assert first.questions == QUESTION and first.model is None  # the runner's own default model


async def test_pages_are_cut_to_top_passages_by_words():
    filler = " ".join(f"filler{i}" for i in range(2000))
    page = {"url": "wiki", "title": "Marie Curie", "text": f"{filler} Curie won the Nobel Prize in Physics in 1903. {filler}"}
    runner = FakeRunner()
    ev = await DecisionJudge(runner, passages_per_page=2, passage_words=64).judge(CLAIM, [page])
    assert len(ev) == 2 and all(e.source == "page" for e in ev)
    assert ev[0].label == "supports" and "1903" in ev[0].text  # BM25 put the relevant chunk first
    assert all(len(r.state["evidence"].split()) <= 64 for r in runner.batches[0])


async def test_a_custom_chunker_replaces_the_word_windows():
    page = {"url": "wiki", "title": "", "text": "first part about nothing. Curie won in 1903. last part."}
    ev = await DecisionJudge(FakeRunner(), chunk=lambda text: text.split(". ")).judge(CLAIM, [page])
    assert ev[0].text == "Curie won in 1903" and ev[0].label == "supports"


async def test_no_docs_gives_no_evidence():
    assert await DecisionJudge(FakeRunner()).judge(CLAIM, []) == []


async def test_missing_titles_become_empty_strings():
    # crawl4ai and search APIs can return the key with None (a page with no <title>), not just omit it
    ev = await DecisionJudge(FakeRunner()).judge(CLAIM, [
        {"url": "u1", "title": None, "snippet": "Curie shared the 1903 Nobel Prize in Physics."},
        {"url": "u2", "title": None, "text": "Marie Curie won the Nobel Prize in Physics in 1903."},
    ])
    assert [(e.url, e.title) for e in ev] == [("u1", ""), ("u2", "")]


async def test_long_snippets_are_cut_to_fit_so_the_claim_is_not_truncated():
    # evidence goes first, then the claim: an 800-word snippet pushed the claim past Laya's 512 tokens, and a true
    # and a false claim came back identical (not_enough_info 0.454 both). Long snippets are cut like pages.
    filler = " ".join(f"filler{i}" for i in range(2000))
    ev = await DecisionJudge(FakeRunner(), passage_words=64).judge(CLAIM, [
        {"url": "long", "title": "", "snippet": f"{filler} Curie shared the 1903 Nobel Prize in Physics. {filler}"},
        {"url": "short", "title": "", "snippet": "Curie shared the 1903 Nobel Prize in Physics."},
    ])
    assert [(e.url, e.source, e.label) for e in ev] == [("long", "snippet", "supports"), ("short", "snippet", "supports")]
    assert len(ev[0].text.split()) <= 64 and "1903" in ev[0].text
    assert ev[1].text == "Curie shared the 1903 Nobel Prize in Physics."  # short snippets stay whole


async def test_claim_and_evidence_are_normalized_before_judging():
    # a live run: "saturation point at 247 Mg ha⁻¹" vs a source's "247 Mg ha−1" was judged a refutation (0.84)
    runner = FakeRunner()
    await DecisionJudge(runner).judge("WorldView-3 saturates at 247 Mg ha⁻¹.", [
        {"url": "u1", "title": "t", "snippet": "WV3 had a saturation point of 247 Mg ha−1."},
    ])
    state = runner.batches[0][0].state
    assert state["claim"] == "WorldView-3 saturates at 247 Mg ha-1."
    assert state["evidence"] == "WV3 had a saturation point of 247 Mg ha-1."


async def test_the_judge_loads_and_closes_its_runner_and_takes_its_concurrency():
    class Resource(FakeRunner):
        concurrency = 3
        loaded = closed = False

        async def aload(self):
            self.loaded = True

        async def aclose(self):
            self.closed = True

    judge = DecisionJudge(runner := Resource())
    await judge.aload()
    await judge.aclose()
    assert runner.loaded and runner.closed and judge.concurrency == 3
    assert DecisionJudge(FakeRunner()).concurrency is None
