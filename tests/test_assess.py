import asyncio
import threading

from factassessor import Atom, CheckResult, ClaimFound, ClaimVerified, Done, Evidence, FactAssessor, Filter, FlatMap, Map, Step

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
    async def ajudge(self, claim, docs):
        label = "refutes" if "cheese" in claim else "supports"
        return [Evidence(url=d["url"], title="t", text="x", source="snippet", label=label, prob=0.95) for d in docs]


class NoCrawl(Step):
    def __call__(self, urls):
        return Map(lambda url: None)(urls)


def offline_assessor():
    searcher = FakeSearcher()
    fa = FactAssessor(
        atomizer=FakeAtomizer() >> Filter(lambda a: "pizza" not in a.text),
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
    assert {n["kind"] for n in result.graph["nodes"]} == {"atom", "source"}


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
    fa = FactAssessor(n_atoms=8, top_k=3, crawl_timeout=1.5, blocked_domains=("example.com",))
    take_atoms = fa.atomizer.steps[-1]
    serper, block, take_hits = fa.searcher.steps
    assert take_atoms.n == 8 and serper.num == 6 and take_hits.n == 3 and fa.crawler.timeout == 1.5
    assert block.pred({"url": "https://facebook.com/x"}) and not block.pred({"url": "https://example.com/x"})
