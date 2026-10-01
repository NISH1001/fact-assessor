import asyncio
import threading

from factassessor import (
    Atom,
    CheckResult,
    ClaimFound,
    ClaimVerified,
    Done,
    Evidence,
    FactAssessor,
    Filter,
    FlatMap,
    Map,
    Step,
    kg,
)

TEXT = "NASA was founded in 1958. I love pizza. The Moon is made of cheese."


class FakeAtomizer(Step):
    """Emits atoms one at a time with a delay, like a streaming LLM."""

    def __call__(self, texts):
        async def atoms(text):
            for i, claim in enumerate(["NASA was founded in 1958.", "I love pizza.", "The Moon is made of cheese."]):
                await asyncio.sleep(0.01)
                yield Atom(id=i, text=claim, span=(0, 1))

        return FlatMap(atoms)(texts)


class FakeSearcher(Step):
    def __init__(self):
        self.loops = set()

    def __call__(self, queries):
        async def hits(query):
            self.loops.add(id(asyncio.get_running_loop()))
            for i in range(2):
                yield {"url": f"https://s{i}.org", "title": "t", "snippet": query}

        return FlatMap(hits)(queries)


class FakeJudge:
    async def judge(self, claim, docs):
        label = "refutes" if "cheese" in claim else "supports"
        return [Evidence(url=d["url"], title="t", text="x", source="snippet", label=label, prob=0.95) for d in docs]


class NoCrawl(Step):
    def __call__(self, urls):
        return Map(lambda url: None)(urls)


def offline_assessor():
    searcher = FakeSearcher()
    fa = FactAssessor(
        atomizer=FakeAtomizer(),
        claim_filter=Filter(lambda a: "pizza" not in a.text),  # any step works as a claim filter
        searcher=searcher,
        crawler=NoCrawl(),
        judge=FakeJudge(),
    )
    fa.fake_searcher = searcher
    return fa


async def test_assess_runs_the_whole_pipeline():
    result = await offline_assessor().assess(TEXT)
    assert isinstance(result, CheckResult)
    assert [(a.atom.text, a.verdict) for a in result.atoms] == [
        ("NASA was founded in 1958.", "supported"),
        ("The Moon is made of cheese.", "refuted"),
    ]
    assert [a.text for a in result.skipped] == ["I love pizza."]
    assert result.fact_score == 0.5
    assert {n["kind"] for n in kg.build(result)["nodes"]} == {"sentence", "claim", "passage", "source"}


async def test_stream_yields_found_then_verified_per_claim_then_done():
    events = [e async for e in offline_assessor().stream(TEXT)]
    kinds = [e.type for e in events]
    assert kinds.count("claim_found") == 2 and kinds.count("claim_verified") == 2 and kinds[-1] == "done"
    for atom_id in (0, 2):
        found = next(i for i, e in enumerate(events) if isinstance(e, ClaimFound) and e.atom.id == atom_id)
        verified = next(i for i, e in enumerate(events) if isinstance(e, ClaimVerified) and e.result.atom.id == atom_id)
        assert found < verified
    assert isinstance(events[-1], Done) and events[-1].result.fact_score == 0.5


async def test_first_claim_is_verified_before_the_atomizer_finishes():
    events = [e async for e in offline_assessor().stream(TEXT)]
    first_verified = next(i for i, e in enumerate(events) if isinstance(e, ClaimVerified))
    last_found = max(i for i, e in enumerate(events) if isinstance(e, ClaimFound))
    assert first_verified < last_found  # streaming: claim 0 settled while claim 2 was still being written


async def test_acheck_is_an_alias_for_assess():
    fa = offline_assessor()
    assert (await fa.acheck(TEXT)).atoms == (await fa.assess(TEXT)).atoms


def test_assess_sync_from_plain_code_reuses_one_background_loop():
    fa = offline_assessor()
    first = fa.assess_sync(TEXT)
    second = fa.assess_sync(TEXT)
    assert first.fact_score == second.fact_score == 0.5
    assert len(fa.fake_searcher.loops) == 1  # same loop both times, so warm resources are reused
    fa.close()
    assert not any(t.name == "factassessor-loop" and t.is_alive() for t in threading.enumerate())


def test_assess_sync_works_while_another_event_loop_is_running():
    fa = offline_assessor()

    async def caller():
        return await asyncio.to_thread(fa.assess_sync, TEXT)

    assert asyncio.run(caller()).fact_score == 0.5
    fa.close()


def test_close_without_ever_running_is_a_no_op():
    FactAssessor().close()


def test_sync_context_manager_closes():
    with offline_assessor() as fa:
        assert fa.assess_sync(TEXT).fact_score == 0.5
    assert fa._loop is None


def test_default_pipeline_is_built_from_the_familiar_arguments():
    fa = FactAssessor(n_atoms=8, top_k=3, crawl_timeout=1.5, blocked_domains=("example.com",), claim_threshold=0.6)
    atomizer, claim_filter, take_atoms = fa.atoms.steps
    assert type(atomizer).__name__ == "LLMAtomizer" and type(claim_filter).__name__ == "DecisionClaimFilter"
    assert claim_filter.threshold == 0.6
    serper, block, take_hits = fa.searcher.step.steps  # under the Cache wrapper
    assert take_atoms.n == 8 and serper.num == 6 and take_hits.n == 3 and fa.crawler.timeout == 1.5
    assert block.pred({"url": "https://facebook.com/x"}) and not block.pred({"url": "https://example.com/x"})
    assert serper.exclude == ("example.com",) and serper.query("q") == "q -site:example.com"  # the same list, in the query


def test_claim_filter_none_means_no_filter():
    fa = FactAssessor(claim_filter=None, n_atoms=3)
    assert [type(s).__name__ for s in fa.atoms.steps] == ["LLMAtomizer", "Take"]


def test_default_filter_and_judge_share_one_laya_runner():
    from factassessor import DecisionJudge, LayaRunner

    fa = FactAssessor(laya_model="multilingual")
    assert fa.claim_filter.runner is fa.judge.runner and isinstance(fa.judge.runner, LayaRunner)
    assert fa.judge.runner.model == "multilingual"
    fa = FactAssessor(judge=DecisionJudge(runner=object()))  # a judge on another runner: the filter still gets Laya
    assert isinstance(fa.claim_filter.runner, LayaRunner) and not isinstance(fa.judge.runner, LayaRunner)


def test_overfetch_keeps_more_hits_than_pages():
    # overfetch=1.0: search keeps 100% more hits than pages, the crawl stage keeps the first top_k readable pages
    from factassessor import FactAssessor

    fa = FactAssessor(top_k=4, overfetch=1.0)
    _, _, take_hits = fa.searcher.step.steps
    assert take_hits.n == 8 and fa.verify.pages_per_claim == 4
    fa = FactAssessor(top_k=4)  # default: today's behaviour, every kept hit is read
    _, _, take_hits = fa.searcher.step.steps
    assert take_hits.n == 4 and fa.verify.pages_per_claim is None
    assert FactAssessor(top_k=4, overfetch=1.0, source_query=True).verify.pages_per_claim == 8  # own + source query's


async def test_assess_many_returns_results_in_input_order_with_at_most_concurrency_texts_in_flight():
    import pytest

    fa = offline_assessor()
    running = peak = 0
    original = fa.assess

    async def tracked(text):
        nonlocal running, peak
        running += 1
        peak = max(peak, running)
        try:
            await asyncio.sleep(0.02)
            return await original(text)
        finally:
            running -= 1

    fa.assess = tracked
    texts = [TEXT, "NASA was founded in 1958.", TEXT, TEXT, "NASA was founded in 1958."]
    results = await fa.assess_many(texts, concurrency=2)
    assert [r.text for r in results] == texts and peak == 2
    assert await fa.assess_many([]) == []
    with pytest.raises(ValueError):
        await fa.assess_many(texts, concurrency=0)


def test_assess_many_sync_for_plain_scripts():
    fa = offline_assessor()
    results = fa.assess_many_sync([TEXT, TEXT], concurrency=2)
    assert [r.fact_score for r in results] == [0.5, 0.5]
    fa.close()
