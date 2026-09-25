import asyncio

from factassessor import Atom, AtomResult, Evidence, Map, Step, Verify, WeightedPolicy, fact_score

ATOM = Atom(id=0, text="Marie Curie won the Nobel Prize in Physics in 1903.", span=(0, 50))
POLICY = WeightedPolicy()


def ev(label, prob, url="https://en.wikipedia.org/wiki/Marie_Curie", source="snippet"):
    return Evidence(url=url, title="t", text="x", source=source, label=label, prob=prob)


def test_verdicts():
    assert POLICY.verdict([]) == ("unverified", 0.0)
    assert POLICY.verdict([ev("not_enough_info", 0.99), ev("supports", 0.6)]) == ("unverified", 0.0)  # nothing strong
    assert POLICY.verdict([ev("supports", 0.8), ev("supports", 0.95)]) == ("supported", 0.95)
    assert POLICY.verdict([ev("refutes", 0.9)]) == ("refuted", 0.9)
    # one stray refutation (a related-but-different fact) doesn't flip three supports
    assert POLICY.verdict([ev("supports", 0.9), ev("supports", 0.9), ev("supports", 0.8), ev("refutes", 0.98)])[0] == "supported"
    verdict, conf = POLICY.verdict([ev("supports", 0.9), ev("refutes", 0.8)])
    assert verdict == "contested" and round(conf, 3) == round(0.9 / 1.7, 3)


def test_settled_needs_two_sure_passages_and_no_strong_disagreement():
    assert POLICY.settled([ev("supports", 0.95), ev("supports", 0.92)])
    assert not POLICY.settled([ev("supports", 0.99)])  # one isn't enough
    assert not POLICY.settled([ev("supports", 0.95), ev("supports", 0.92), ev("refutes", 0.75)])
    assert POLICY.settled([ev("refutes", 0.95), ev("refutes", 0.91), ev("not_enough_info", 0.99)])


def test_score_ignores_unverified():
    results = [AtomResult(atom=ATOM, verdict=v) for v in ["supported", "supported", "refuted", "unverified"]]
    assert fact_score(results) == 2 / 3
    assert fact_score([AtomResult(atom=ATOM, verdict="unverified")]) is None


# --- Verify with fake components ---------------------------------------------------------------------


class FakeJudge:
    def __init__(self, snippet_ev, page_ev):
        self.snippet_ev, self.page_ev = snippet_ev, page_ev

    async def judge(self, claim, docs):
        return list(self.page_ev) if docs and "text" in docs[0] else list(self.snippet_ev)


class FakeSearcher(Step):
    def __init__(self, n=3, fail=False):
        self.n, self.fail = n, fail

    async def __call__(self, queries):
        async for _ in queries:
            if self.fail:
                raise RuntimeError("serper down")
            for i in range(self.n):
                yield {"url": f"https://s{i}.org", "title": "t", "snippet": "s"}


class FakeCrawler(Step):
    def __init__(self, dead=()):
        self.crawled, self.dead = [], set(dead)

    def __call__(self, urls):
        async def crawl(url):
            self.crawled.append(url)
            if url in self.dead:
                await asyncio.sleep(5)  # a dead site that would hit the crawl timeout
            return {"url": url, "title": "t", "text": "page"}

        return Map(crawl)(urls)


async def test_verify_skips_crawling_when_snippets_are_conclusive():
    crawler = FakeCrawler()
    verify = Verify(FakeSearcher(), crawler, FakeJudge([ev("supports", 0.95), ev("supports", 0.93)], []))
    result = await verify.verify(ATOM)
    assert result.verdict == "supported" and crawler.crawled == []


async def test_verify_crawls_and_judges_pages_when_snippets_are_not_enough():
    crawler = FakeCrawler()
    verify = Verify(FakeSearcher(), crawler, FakeJudge([ev("not_enough_info", 0.9)], [ev("supports", 0.85, source="page")]))
    result = await verify.verify(ATOM)
    assert len(crawler.crawled) == 3
    assert result.verdict == "supported" and sum(e.source == "page" for e in result.evidence) == 3


async def test_crawling_stops_once_pages_settle_the_atom():
    crawler = FakeCrawler(dead={"https://s2.org", "https://s3.org"})
    verify = Verify(FakeSearcher(n=4), crawler, FakeJudge([ev("not_enough_info", 0.9)], [ev("supports", 0.95, source="page")]))
    start = asyncio.get_running_loop().time()
    result = await verify.verify(ATOM)
    assert asyncio.get_running_loop().time() - start < 1  # didn't wait for the dead sites
    assert result.verdict == "supported"
    assert sum(e.source == "page" for e in result.evidence) == 2  # two sure pages were enough


async def test_verify_failure_is_unverified_not_a_crash():
    result = await Verify(FakeSearcher(fail=True), FakeCrawler(), FakeJudge([], [])).verify(ATOM)
    assert result.verdict == "unverified" and "serper down" in result.error


async def test_verify_times_out_to_unverified():
    crawler = FakeCrawler(dead={"https://s0.org", "https://s1.org", "https://s2.org"})
    verify = Verify(FakeSearcher(), crawler, FakeJudge([], []), timeout=0.1)
    result = await verify.verify(ATOM)
    assert result.verdict == "unverified" and result.error == "timeout"
