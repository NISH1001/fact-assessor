"""The eval harness (scripts/eval_atoms.py) on fakes: a live run records, for every claim, how long each component
took, so a slow run explains itself from its results file instead of needing a traced rerun."""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import eval_atoms as harness  # noqa: E402
from factassessor import Crawler, FactAssessor, Fetch, FlatMap, Step  # noqa: E402
from factassessor.schema import Evidence  # noqa: E402


class NoAtomizer(Step):
    async def __call__(self, texts):
        async for _ in texts:
            pass
        return
        yield


class Searcher(Step):
    def __call__(self, queries):
        async def hits(query):
            for i in range(2):
                yield {"url": f"https://s{i}.org/{i}", "title": "t", "snippet": "s"}

        return FlatMap(hits)(queries)


class Reader(Crawler):
    async def crawl(self, url):
        await asyncio.sleep(0.02)
        return Fetch(url=url, page={"url": url, "title": "t", "text": "page " * 400})


class Judge:
    async def judge(self, claim, docs):
        await asyncio.sleep(0.02)
        page = bool(docs) and "text" in docs[0]
        return [Evidence(url=d["url"], title="t", text="x", source="page" if page else "snippet",
                         label="supports" if page else "not_enough_info", prob=0.9 if page else 0.5) for d in docs]


async def test_a_live_run_records_each_claims_time_per_component():
    fa = FactAssessor(atomizer=NoAtomizer(), claim_filter=None, searcher=Searcher(), crawler=Reader(), resolver=None,
                      judge=Judge(), source_queries=0)
    timings = harness.instrument(fa)
    result = await fa.assess("Some text.", claims=["the first claim", "the second claim"])
    assert [r.verdict for r in result.atoms] == ["supported", "supported"]
    for r in result.atoms:
        times = harness.component_times(timings[id(r.atom)])
        assert times["search"] >= 0 and times["judge_snippets"] >= 0.02
        assert times["crawl"] >= 0.02 and times["judge_pages"] >= 0.02
