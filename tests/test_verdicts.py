from factassessor import Atom, AtomResult, Evidence, FactAssessor

ATOM = Atom(id=0, text="Marie Curie won the Nobel Prize in Physics in 1903.", span=(0, 50))


def ev(label, prob, url="https://en.wikipedia.org/wiki/Marie_Curie", source="snippet"):
    return Evidence(url=url, title="t", text="x", source=source, label=label, prob=prob)


FR = FactAssessor()


def test_aggregate_verdicts():
    assert FR._aggregate([]) == ("unverified", 0.0)
    assert FR._aggregate([ev("not_enough_info", 0.99), ev("supports", 0.6)]) == ("unverified", 0.0)  # nothing strong
    assert FR._aggregate([ev("supports", 0.8), ev("supports", 0.95)]) == ("supported", 0.95)
    assert FR._aggregate([ev("refutes", 0.9)]) == ("refuted", 0.9)
    # one stray refutation (a related-but-different fact) doesn't flip three supports
    assert FR._aggregate([ev("supports", 0.9), ev("supports", 0.9), ev("supports", 0.8), ev("refutes", 0.98)])[0] == "supported"
    verdict, conf = FR._aggregate([ev("supports", 0.9), ev("refutes", 0.8)])
    assert verdict == "contested" and round(conf, 3) == round(0.9 / 1.7, 3)


def test_early_exit_needs_two_sure_passages_and_no_strong_disagreement():
    assert FR._is_confident([ev("supports", 0.95), ev("supports", 0.92)])
    assert not FR._is_confident([ev("supports", 0.99)])  # one isn't enough
    assert not FR._is_confident([ev("supports", 0.95), ev("supports", 0.92), ev("refutes", 0.75)])
    assert FR._is_confident([ev("refutes", 0.95), ev("refutes", 0.91), ev("not_enough_info", 0.99)])


def test_score_ignores_unverified():
    results = [AtomResult(atom=ATOM, verdict=v) for v in ["supported", "supported", "refuted", "unverified"]]
    assert FR._score(results) == 2 / 3
    assert FR._score([AtomResult(atom=ATOM, verdict="unverified")]) is None


def test_graph_links_sources_to_atoms_with_strong_edges_only():
    result = AtomResult(atom=ATOM, verdict="supported", evidence=[
        ev("supports", 0.8), ev("supports", 0.95),  # same source twice -> one edge, max weight
        ev("refutes", 0.75, url="https://www.example.com/a"),
        ev("not_enough_info", 0.99, url="https://noise.org/x"),  # not an edge
        ev("supports", 0.5, url="https://weak.org/y"),  # too weak
    ])
    graph = FR._build_graph([result])
    assert {n["id"] for n in graph["nodes"]} == {"atom:0", "source:en.wikipedia.org", "source:example.com"}
    assert sorted((e["source"], e["relation"], e["weight"]) for e in graph["edges"]) == [
        ("source:en.wikipedia.org", "supports", 0.95),
        ("source:example.com", "refutes", 0.75),
    ]


class FakeJudge:
    def __init__(self, snippet_ev, page_ev):
        self.snippet_ev, self.page_ev = snippet_ev, page_ev

    async def ajudge(self, claim, docs):
        return list(self.page_ev) if docs and "text" in docs[0] else list(self.snippet_ev)


def reasoner(snippet_ev, page_ev):
    fr = FactAssessor(judge=FakeJudge(snippet_ev, page_ev))
    fr.crawled = []

    async def search(q):
        return [{"url": f"https://s{i}.org", "title": "t", "snippet": "s"} for i in range(3)]

    async def crawl(url):
        fr.crawled.append(url)
        return {"url": url, "title": "t", "text": "page"}

    fr._search, fr._crawl = search, crawl
    return fr


async def test_verify_skips_crawling_when_snippets_are_conclusive():
    fr = reasoner([ev("supports", 0.95), ev("supports", 0.93)], [])
    result = await fr._verify(ATOM)
    assert result.verdict == "supported" and fr.crawled == []


async def test_verify_crawls_and_judges_pages_when_snippets_are_not_enough():
    fr = reasoner([ev("not_enough_info", 0.9)], [ev("supports", 0.85, source="page")])
    result = await fr._verify(ATOM)
    assert len(fr.crawled) == 3
    assert result.verdict == "supported" and sum(e.source == "page" for e in result.evidence) == 3


async def test_verify_failure_is_unverified_not_a_crash():
    fr = reasoner([], [])

    async def broken(q):
        raise RuntimeError("serper down")

    fr._search = broken
    result = await fr._verify(ATOM)
    assert result.verdict == "unverified" and "serper down" in result.error


async def test_crawling_stops_once_pages_settle_the_atom():
    import asyncio

    fr = FactAssessor(judge=FakeJudge([ev("not_enough_info", 0.9)], [ev("supports", 0.95, source="page")]))
    fr.crawled = []

    async def search(q):
        return [{"url": f"https://s{i}.org", "title": "t", "snippet": "s"} for i in range(4)]

    async def crawl(url):
        fr.crawled.append(url)
        if url in ("https://s2.org", "https://s3.org"):
            await asyncio.sleep(5)  # dead sites that would hit the crawl timeout
        return {"url": url, "title": "t", "text": "page"}

    fr._search, fr._crawl = search, crawl
    start = asyncio.get_running_loop().time()
    result = await fr._verify(ATOM)
    assert asyncio.get_running_loop().time() - start < 1  # didn't wait for the dead sites
    assert result.verdict == "supported"
    assert sum(e.source == "page" for e in result.evidence) == 2  # two sure pages were enough

