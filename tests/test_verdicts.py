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


async def test_a_timeout_keeps_the_evidence_gathered_so_far():
    # the snippets were judged (one strong support) before the crawls hung: that evidence still decides the claim
    crawler = FakeCrawler(dead={"https://s0.org", "https://s1.org", "https://s2.org"})
    verify = Verify(FakeSearcher(), crawler, FakeJudge([ev("supports", 0.8)], []), timeout=0.1)
    result = await verify.verify(ATOM)
    assert result.error == "timeout" and result.verdict == "supported" and len(result.evidence) == 1


async def test_a_timeout_keeps_pages_judged_before_it():
    class OnePageThenHang(Step):
        def __call__(self, urls):
            async def crawl(url):
                if url != "https://s0.org":
                    await asyncio.sleep(5)
                return {"url": url, "title": "t", "text": "page"}

            return Map(crawl)(urls)

    judge = FakeJudge([ev("not_enough_info", 0.9)], [ev("refutes", 0.9, source="page")])
    result = await Verify(FakeSearcher(), OnePageThenHang(), judge, timeout=0.2).verify(ATOM)
    assert result.error == "timeout" and result.verdict == "refuted"
    assert [e.source for e in result.evidence] == ["snippet", "page"]


def test_fact_score_is_computed_from_the_atoms_and_serialized():
    from factassessor import CheckResult

    result = CheckResult(text="t", atoms=[AtomResult(atom=ATOM, verdict=v) for v in ["supported", "refuted"]], latency_ms=1.0)
    assert result.fact_score == 0.5
    result.atoms.append(AtomResult(atom=ATOM, verdict="supported"))
    assert result.fact_score == 2 / 3  # follows the atoms; can't go stale
    assert '"fact_score":0.6666666666666666' in result.model_dump_json()


# --- how many claims verify at once ---------------------------------------------------------------------


class SlowJudge:
    """Takes `delay` per call; records the most claims it was ever judging at once."""

    def __init__(self, delay=0.05, concurrency=None):
        self.delay, self.running, self.peak = delay, 0, 0
        if concurrency is not None:
            self.concurrency = concurrency

    async def judge(self, claim, docs):
        self.running += 1
        self.peak = max(self.peak, self.running)
        await asyncio.sleep(self.delay)
        self.running -= 1
        return [ev("supports", 0.95), ev("supports", 0.95)]


async def _atoms(n):
    for i in range(n):
        yield Atom(id=i, text=f"claim {i}", span=(0, 7))


async def test_concurrency_caps_the_claims_in_flight():
    from factassessor import collect

    judge = SlowJudge()
    results = await collect(Verify(FakeSearcher(), FakeCrawler(), judge, concurrency=2)(_atoms(5)))
    assert len(results) == 5 and judge.peak == 2


async def test_no_concurrency_limit_runs_every_claim_at_once():
    from factassessor import collect

    judge = SlowJudge()
    await collect(Verify(FakeSearcher(), FakeCrawler(), judge, concurrency=None)(_atoms(5)))
    assert judge.peak == 5


async def test_waiting_for_a_slot_does_not_count_toward_the_timeout():
    # a slow judge (GLiNER on CPU) can't take 20 claims at once; queued claims must not time out while they wait
    from factassessor import collect

    verify = Verify(FakeSearcher(), FakeCrawler(), SlowJudge(delay=0.1), timeout=0.15, concurrency=1)
    results = await collect(verify(_atoms(3)))  # 0.3s in total, 0.1s each
    assert [r.verdict for r in results] == ["supported"] * 3


def test_verify_takes_the_judges_concurrency_unless_given():
    assert Verify(FakeSearcher(), FakeCrawler(), SlowJudge(concurrency=3)).concurrency == 3
    assert Verify(FakeSearcher(), FakeCrawler(), SlowJudge(concurrency=3), concurrency=7).concurrency == 7
    assert Verify(FakeSearcher(), FakeCrawler(), SlowJudge(concurrency=3), concurrency=None).concurrency is None
    assert Verify(FakeSearcher(), FakeCrawler(), FakeJudge([], [])).concurrency is None  # no hint: no limit


def test_fact_assessor_passes_max_concurrent_claims_to_verify():
    from factassessor import FactAssessor

    def fa(**kw):
        return FactAssessor(atomizer=Step(), claim_filter=None, searcher=FakeSearcher(), crawler=FakeCrawler(), **kw)

    assert fa(judge=SlowJudge(concurrency=3)).verify.concurrency == 3
    assert fa(judge=SlowJudge(concurrency=3), max_concurrent_claims=4).verify.concurrency == 4
    assert fa(judge=SlowJudge(concurrency=3), max_concurrent_claims=None).verify.concurrency is None
    assert fa(judge=FakeJudge([], [])).verify.concurrency is None


async def test_snippets_still_decide_when_every_crawl_returns_an_error_page():
    # searcher -> snippets are judged first; crawling only adds pages on top. Error pages (404, 503) are dropped by
    # the crawler, so the claim is decided from its snippets instead of failing or judging junk.
    from tests.test_crawl import crawler_with, result

    async def error_page(url):
        r = result(markdown="PubChem is temporarily unavailable (HTTP 503)")
        r.status_code = 503 if url.endswith("0.org") else 404
        return r

    crawler = crawler_with(error_page)
    judged_pages = []

    class Judge:
        async def judge(self, claim, docs):
            if docs and "text" in docs[0]:
                judged_pages.extend(docs)
                return [ev("refutes", 0.95, source="page")]  # would flip the verdict if an error page got through
            return [ev("supports", 0.8, url=d["url"]) for d in docs]  # strong, but not sure enough to stop early

    result_ = await Verify(FakeSearcher(n=3), crawler, Judge()).verify(ATOM)
    assert judged_pages == []  # no error page was judged
    assert result_.verdict == "supported" and result_.error is None
    assert [e.source for e in result_.evidence] == ["snippet"] * 3


# --- resolve -> read: resolvers before the crawler ----------------------------------------------------------------

from factassessor import Crawler  # noqa: E402

PAPER_TEXT = " ".join(["word"] * 400)  # a real paper's worth of words
STUB_TEXT = " ".join(["bot"] * 180)  # a bot-check page: long enough to look real, too short for a paper


class Copies:
    """A resolver: each hit has one free copy at `<hit>/copy`; records what it was asked."""

    def __init__(self, delay=0.0):
        self.asked, self.delay = [], delay

    async def resolve(self, url):
        self.asked.append(url)
        await asyncio.sleep(self.delay)
        return [url + "/copy"]


class PageCrawler(Crawler):
    """crawl(url) -> page from `pages` ({url: text}); None for others; urls in `hang` never answer."""

    def __init__(self, pages, hang=()):
        self.pages, self.hang, self.crawled = pages, set(hang), []

    async def crawl(self, url):
        self.crawled.append(url)
        if url in self.hang:
            await asyncio.sleep(5)
        text = self.pages.get(url)
        return {"url": url, "title": "t", "text": text} if text else None


class RecordingJudge(FakeJudge):
    """Pages support the claim; snippets don't settle it. Records every page it judged."""

    def __init__(self):
        super().__init__([ev("not_enough_info", 0.9)], [ev("supports", 0.85, source="page")])
        self.pages = []

    async def judge(self, claim, docs):
        self.pages += [d for d in docs if "text" in d]
        return await super().judge(claim, docs)


async def test_the_crawler_reads_the_resolved_copy_first_under_the_hits_url():
    crawler, judge = PageCrawler({"https://s0.org/copy": PAPER_TEXT, "https://s0.org": "the page"}), RecordingJudge()
    result = await Verify(FakeSearcher(n=1), crawler, judge, resolver=Copies()).verify(ATOM)
    assert crawler.crawled == ["https://s0.org/copy"]  # the copy worked: the page itself is never fetched
    assert judge.pages == [{"url": "https://s0.org", "title": "t", "text": PAPER_TEXT}]  # cited as the hit
    assert result.verdict == "supported"


async def test_a_stub_copy_is_skipped_for_the_hits_own_page():
    crawler = PageCrawler({"https://s0.org/copy": STUB_TEXT, "https://s0.org": "the page itself, short but real"})
    result = await Verify(FakeSearcher(n=1), crawler, RecordingJudge(), resolver=Copies()).verify(ATOM)
    assert crawler.crawled == ["https://s0.org/copy", "https://s0.org"] and result.verdict == "supported"


async def test_every_hit_is_resolved_at_once():
    resolver = Copies(delay=0.2)
    crawler = PageCrawler({f"https://s{i}.org/copy": PAPER_TEXT for i in range(3)})
    start = asyncio.get_running_loop().time()
    await Verify(FakeSearcher(n=3), crawler, FakeJudge([ev("not_enough_info", 0.9)], []), resolver=resolver).verify(ATOM)
    assert sorted(resolver.asked) == ["https://s0.org", "https://s1.org", "https://s2.org"]
    assert asyncio.get_running_loop().time() - start < 0.35  # concurrently, not 0.6s


async def test_a_hanging_source_gives_up_at_its_read_deadline():
    crawler = PageCrawler({"https://s1.org/copy": PAPER_TEXT}, hang={"https://s0.org/copy"})
    verify = Verify(FakeSearcher(n=2), crawler, RecordingJudge(), resolver=Copies(), read_timeout=0.2, timeout=2)
    start = asyncio.get_running_loop().time()
    result = await verify.verify(ATOM)
    assert asyncio.get_running_loop().time() - start < 1 and result.error is None  # not the 2s claim timeout
    assert result.verdict == "supported" and "https://s0.org" not in crawler.crawled  # its deadline was spent


async def test_no_resolver_is_the_crawler_on_each_hit_as_before():
    crawler = FakeCrawler()
    await Verify(FakeSearcher(), crawler, FakeJudge([ev("not_enough_info", 0.9)], [])).verify(ATOM)
    assert sorted(crawler.crawled) == ["https://s0.org", "https://s1.org", "https://s2.org"]


def test_a_resolver_needs_a_crawler_that_crawls_one_url():
    import pytest

    with pytest.raises(TypeError):
        Verify(FakeSearcher(), FakeCrawler(), FakeJudge([], []), resolver=Copies())  # a bare step: no crawl(url)


async def test_fact_assessor_takes_a_resolver_and_manages_its_lifecycle():
    from factassessor import FactAssessor, OpenAlexResolver

    oa = OpenAlexResolver()
    fa = FactAssessor(atomizer=Step(), claim_filter=None, searcher=FakeSearcher(), crawler=PageCrawler({}),
                      judge=FakeJudge([], []), resolver=oa)
    assert fa.verify.resolver is oa
    await fa.aload()
    assert oa._http is not None
    await fa.aclose()
    assert oa._http is None
