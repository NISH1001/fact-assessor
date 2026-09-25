import asyncio
import threading

from factassessor import Atom, CheckResult, Evidence, FactAssessor

TEXT = "NASA was founded in 1958. I love pizza."


class FakeAtomizer:
    async def aatomize(self, text):
        return [Atom(id=0, text="NASA was founded in 1958.", span=(0, 24)), Atom(id=1, text="I love pizza.", span=(26, 39))]


class FakeFilter:
    async def afilter(self, atoms):
        return atoms[:1], atoms[1:]


class FakeJudge:
    async def ajudge(self, claim, docs):
        return [Evidence(url=d["url"], title="t", text="x", source="snippet", label="supports", prob=0.95) for d in docs]


def offline_assessor():
    fa = FactAssessor(atomizer=FakeAtomizer(), atom_filter=FakeFilter(), judge=FakeJudge())
    fa.loops = set()

    async def search(query):
        fa.loops.add(id(asyncio.get_running_loop()))
        await asyncio.sleep(0)
        return [{"url": f"https://s{i}.org", "title": "t", "snippet": "s"} for i in range(2)]

    fa._search = search
    return fa


async def test_assess_runs_the_whole_pipeline():
    result = await offline_assessor().assess(TEXT)
    assert isinstance(result, CheckResult)
    assert [(a.atom.text, a.verdict) for a in result.atoms] == [("NASA was founded in 1958.", "supported")]
    assert [a.text for a in result.skipped] == ["I love pizza."]
    assert result.fact_score == 1.0


async def test_acheck_is_an_alias_for_assess():
    fa = offline_assessor()
    assert (await fa.acheck(TEXT)).atoms == (await fa.assess(TEXT)).atoms


def test_assess_sync_from_plain_code_reuses_one_background_loop():
    fa = offline_assessor()
    first = fa.assess_sync(TEXT)
    second = fa.assess_sync(TEXT)
    assert first.fact_score == second.fact_score == 1.0
    assert len(fa.loops) == 1  # same loop both times, so warm resources (browser, HTTP pool) are reused
    fa.close()
    assert not any(t.name == "factassessor-loop" and t.is_alive() for t in threading.enumerate())


def test_assess_sync_works_while_another_event_loop_is_running():
    # e.g. Jupyter: a loop is already running in this thread, so asyncio.run() would fail
    fa = offline_assessor()

    async def caller():
        return await asyncio.to_thread(fa.assess_sync, TEXT)

    assert asyncio.run(caller()).fact_score == 1.0
    fa.close()


def test_close_without_ever_running_is_a_no_op():
    FactAssessor().close()


def test_sync_context_manager_closes():
    with offline_assessor() as fa:
        assert fa.assess_sync(TEXT).fact_score == 1.0
    assert fa._loop is None
