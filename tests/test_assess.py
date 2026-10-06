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
        resolver=None,  # a bare step crawler: nothing to read resolved copies with
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
    verdicts = lambda r: [(a.atom.text, a.verdict) for a in r.atoms]  # noqa: E731  (times differ run to run)
    assert verdicts(await fa.acheck(TEXT)) == verdicts(await fa.assess(TEXT))


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
    fa = FactAssessor(max_claims=8, top_k=3, overfetch=0.0, crawl_timeout=1.5, blocked_domains=("example.com",), claim_threshold=0.6)
    atomizer, claim_filter, take_atoms = fa.atoms.steps
    assert type(atomizer).__name__ == "LLMAtomizer" and type(claim_filter).__name__ == "DecisionClaimFilter"
    assert claim_filter.threshold == 0.6
    serper, block, take_hits = fa.searcher.step.steps  # under the Cache wrapper
    assert take_atoms.n == 8 and not take_atoms.silent and serper.num == 10 and take_hits.n == 3  # 10 results = 1 Serper credit
    assert [c.timeout for c in fa.crawler.crawlers] == [1.5, 1.5, 1.5]
    assert block.pred({"url": "https://facebook.com/x"}) and not block.pred({"url": "https://example.com/x"})
    assert serper.exclude == ("example.com",) and serper.query("q") == "q -site:example.com"  # the same list, in the query


def test_defaults_are_the_measured_best_setup():
    # the configuration behind the eval numbers in issue #1, local judge: Laya (Jev is one argument away)
    from factassessor import CascadedCrawler, CompositeResolver, Crawl4AICrawler, HTTPXCrawler, SystemOneRunner

    fa = FactAssessor()
    assert fa.atomizer.agent.model == "openai:gpt-6-luna" and fa.atomizer.source_queries == 2  # two searches for the text's source
    serper, _, take_hits = fa.searcher.step.steps
    assert take_hits.n == 10 and serper.num == 10  # overfetch 1.0: 10 hits per query, still 1 credit
    assert fa.verify.pages_per_claim == 10  # the first 10 readable pages: 15 tripled the claims hitting the deadline
    assert fa.policy.strong == 0.7 and fa.policy.strong_refute == 0.9  # a refutation needs more confidence than a support
    assert isinstance(fa.verify.resolver, CompositeResolver)  # papers read in full (arXiv, OpenAlex)
    from factassessor.crawlers import ImpitCrawler

    # honest HTTP, then HTTP that looks like Firefox, then a real browser: on 600 hits (with the resolver) 395 read vs
    # 350 for HTTP then browser, with 17% fewer browser renders
    assert isinstance(fa.crawler, CascadedCrawler) and [type(c) for c in fa.crawler.crawlers] == [HTTPXCrawler, ImpitCrawler, Crawl4AICrawler]
    assert fa.crawler.crawlers[0].max_pdf_bytes == 50_000_000  # a cut PDF can't be read at all: theses run 25-35 MB
    assert fa.judge.passages_per_page == 3 and isinstance(fa.judge.runner, SystemOneRunner)  # Jev: the measured judge
    assert fa.verify.timeout == 30
    assert FactAssessor(resolver=None).verify.resolver is None  # opt out


def test_claim_filter_none_means_no_filter():
    fa = FactAssessor(claim_filter=None, max_claims=3)
    assert [type(s).__name__ for s in fa.atoms.steps] == ["LLMAtomizer", "Take"] and fa.atoms.steps[-1].n == 3 and not fa.atoms.steps[-1].silent


def test_default_filter_and_judge_share_one_runner():
    from factassessor import DecisionJudge, LayaRunner, SystemOneRunner

    fa = FactAssessor(runner=LayaRunner("multilingual"))
    assert fa.claim_filter.runner is fa.judge.runner and fa.judge.runner.model == "multilingual"
    fa = FactAssessor(judge=DecisionJudge(runner=object()))  # a judge on another runner: the filter still gets the default
    assert isinstance(fa.claim_filter.runner, SystemOneRunner) and not isinstance(fa.judge.runner, SystemOneRunner)


def test_overfetch_keeps_more_hits_than_pages():
    # overfetch=1.0 (the default): search keeps 100% more hits than pages, the crawl stage the first readable pages
    from factassessor import FactAssessor

    fa = FactAssessor(top_k=4, overfetch=1.0, source_queries=0)
    _, _, take_hits = fa.searcher.step.steps
    assert take_hits.n == 8 and fa.verify.pages_per_claim == 4
    fa = FactAssessor(top_k=4, overfetch=0.0)  # every kept hit is read
    _, _, take_hits = fa.searcher.step.steps
    assert take_hits.n == 4 and fa.verify.pages_per_claim is None
    assert FactAssessor(top_k=4, overfetch=1.0, source_queries=3).verify.pages_per_claim == 8  # 2 x top_k, however many source queries
    assert FactAssessor(top_k=4, pages_per_claim=12).verify.pages_per_claim == 12


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


def test_the_default_cascade_sends_every_failure_to_the_browser_except_404s_and_pdfs():
    # a browser can't bring back a page that is gone (404, 410) or read a PDF plain HTTP couldn't (0 of 95 rescued)
    from factassessor.crawlers import Fetch

    when = FactAssessor().crawler.when
    tried = lambda **kw: when(Fetch(url="u", **kw))  # noqa: E731
    assert not tried(status=404) and not tried(status=410)
    assert not tried(status=200, content_type="application/pdf")
    assert tried(status=403, content_type="text/html")         # bot blocks: often readable in a browser
    assert tried(status=200, content_type="text/html", words=5)  # JavaScript shells
    assert tried(error="TimeoutError") and tried(status=503)


# --- every claim is checked; a cap says what it left out; given claims; one claim queue per assessor -----------

class ManyClaims(Step):
    """An atomizer of 8 claims, emitted one at a time."""

    def __init__(self):
        self.calls = 0

    def __call__(self, texts):
        async def atoms(text):
            self.calls += 1
            for i in range(8):
                await asyncio.sleep(0)
                yield Atom(id=i, text=f"Claim number {i} is true.", span=(0, 1))

        return FlatMap(atoms)(texts)


def many_claims_assessor(**kw):
    return FactAssessor(atomizer=ManyClaims(), claim_filter=None, searcher=FakeSearcher(), crawler=NoCrawl(),
                        resolver=None, judge=FakeJudge(), **kw)


async def test_every_claim_is_checked_by_default():
    # the old default checked the first 5 claims and dropped the rest without saying so
    result = await many_claims_assessor().assess("text")
    assert len(result.atoms) == 8 and result.unchecked == []
    assert FactAssessor().atoms.steps[-1].n is None  # Take(None): every claim passes


async def test_a_cap_checks_the_first_n_and_reports_the_rest_as_unchecked():
    result = await many_claims_assessor(max_claims=3).assess("text")
    assert [a.atom.id for a in result.atoms] == [0, 1, 2]
    assert [a.id for a in result.unchecked] == [3, 4, 5, 6, 7] and result.skipped == []


async def test_given_claims_skip_the_atomizer_and_the_filter():
    fa = many_claims_assessor()
    fa.claim_filter = Filter(lambda a: False)  # would drop everything: given claims are checked as given
    text = "NASA was founded in 1958. The Moon is made of cheese."
    result = await fa.assess(text, claims=["NASA was founded in 1958.", "The Moon is made of cheese."])
    assert fa.atomizer.calls == 0
    assert [(a.atom.text, a.verdict) for a in result.atoms] == [("NASA was founded in 1958.", "supported"), ("The Moon is made of cheese.", "refuted")]
    assert result.atoms[1].atom.span == (26, len(text))  # located in the text


async def test_given_claims_still_get_the_texts_source_queries():
    class WithSources(ManyClaims):
        async def source_queries_for(self, text):
            return ["the source paper"]

    fa = FactAssessor(atomizer=WithSources(), claim_filter=None, searcher=FakeSearcher(), crawler=NoCrawl(), resolver=None, judge=FakeJudge())
    result = await fa.assess("text", claims=["A claim."])
    assert result.atoms[0].atom.source_queries == ["the source paper"]


async def test_claims_in_flight_are_capped_across_all_texts_of_an_assessor():
    # per-text limits multiplied with assess_many: 10 answers at once put ~170 claims in flight and 63% hit the deadline
    running = peak = 0

    class SlowJudge(FakeJudge):
        async def judge(self, claim, docs):
            nonlocal running, peak
            running += 1
            peak = max(peak, running)
            await asyncio.sleep(0.02)
            running -= 1
            return await super().judge(claim, docs)

    fa = FactAssessor(atomizer=ManyClaims(), claim_filter=None, searcher=FakeSearcher(), crawler=NoCrawl(), resolver=None,
                      judge=SlowJudge(), max_concurrent_claims=4)
    results = await fa.assess_many(["a", "b", "c"], concurrency=3)
    assert [len(r.atoms) for r in results] == [8, 8, 8] and peak == 4
    assert FactAssessor().verify.concurrency == 50  # the default: about the claim load measured fine on a laptop


def test_the_package_reports_its_version():
    import factassessor

    assert factassessor.__version__ == "0.1.0"
