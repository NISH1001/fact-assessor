import httpx
import pytest

from factassessor import AKDDoiCrawler, AKDScholarSearcher, CascadedCrawler, Crawler, Fetch, Take, collect, once
from factassessor.keys import MissingAPIKeyError, auth_error

BASE = "http://scholar.test:8000"


def doc(doi, title, *passages, link=None):
    return {
        "title": title, "doi": doi, "link": link if link is not None else (f"https://doi.org/{doi}" if doi else None),
        "content": passages[0], "matches": [{"content": p, "score": 1.0} for p in passages],
        "match_count": len(passages), "score": 30.0,
    }


SEARCH_RESPONSE = {
    "searchParameters": {}, "took_ms": 50,
    "docs": [
        doc("10.1/a", "Aspirin and colorectal cancer", "…aspirin reduced the risk…", "…long-term aspirin use…"),
        doc("10.1/b", "Aspirin cohort", "…a cohort in Sweden…"),
        doc("10.1/a", "Aspirin and colorectal cancer", "…aspirin reduced the risk…", "…daily dose of aspirin…"),
        doc(None, "No DOI, open-access PDF", "…a passage…", link="https://example.org/paper.pdf"),
        doc(None, "No DOI, no link", "…nowhere to cite…", link=""),
    ],
}


def searcher(handler, **kwargs):
    s = AKDScholarSearcher(url=BASE, api_key="test-key", **kwargs)
    s._http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return s


async def test_search_sends_the_query_with_the_bearer_key_and_passage_settings():
    seen = {}

    def handler(request):
        seen["url"], seen["auth"] = request.url, request.headers["authorization"]
        return httpx.Response(200, json=SEARCH_RESPONSE)

    await searcher(handler, num=5, context_words=40, max_matches=3).search("aspirin colorectal cancer")
    assert seen["auth"] == "Bearer test-key"
    assert seen["url"].path == "/search"
    params = dict(seen["url"].params)
    assert params["q"] == "aspirin colorectal cancer"
    assert params["context_words"] == "40" and params["max_matches"] == "3"
    assert int(params["top_k"]) > 5  # room for the repeats that are dropped


async def test_hits_are_papers_once_each_with_their_passages_as_the_snippet():
    hits = await searcher(lambda r: httpx.Response(200, json=SEARCH_RESPONSE)).search("aspirin")
    assert [h["url"] for h in hits] == ["https://doi.org/10.1/a", "https://doi.org/10.1/b", "https://example.org/paper.pdf"]
    first = hits[0]
    assert first["title"] == "Aspirin and colorectal cancer" and set(first) == {"url", "title", "snippet"}
    # a repeated paper's passages are merged, each once, best first; the ellipses at the cut ends are dropped
    assert first["snippet"] == "aspirin reduced the risk … long-term aspirin use … daily dose of aspirin"


async def test_passages_per_paper_are_capped_at_max_matches():
    hits = await searcher(lambda r: httpx.Response(200, json=SEARCH_RESPONSE), max_matches=2).search("aspirin")
    assert hits[0]["snippet"] == "aspirin reduced the risk … long-term aspirin use"


async def test_at_most_num_hits():
    hits = await searcher(lambda r: httpx.Response(200, json=SEARCH_RESPONSE), num=1).search("aspirin")
    assert len(hits) == 1


async def test_it_is_a_searcher_step():
    s = searcher(lambda r: httpx.Response(200, json=SEARCH_RESPONSE))
    hits = await collect((s >> Take(2))(once("aspirin")))
    assert [h["url"] for h in hits] == ["https://doi.org/10.1/a", "https://doi.org/10.1/b"]


async def test_http_error_raises():
    s = searcher(lambda r: httpx.Response(500, json={"detail": "boom"}))
    with pytest.raises(httpx.HTTPStatusError):
        await s.search("aspirin")


async def test_a_rejected_key_raises_the_http_error_that_auth_error_names(monkeypatch):
    monkeypatch.setenv("AKD_SCHOLAR_SEARCH_URL", BASE)
    s = searcher(lambda r: httpx.Response(401, json={"detail": "Invalid API key"}))
    with pytest.raises(httpx.HTTPStatusError) as err:
        await s.search("aspirin")
    assert auth_error(err.value).env == "AKD_SCHOLAR_SEARCH_API_KEY"  # as SerperSearcher: Verify stops the check


def test_url_and_key_come_from_the_environment(monkeypatch):
    monkeypatch.setenv("AKD_SCHOLAR_SEARCH_URL", BASE + "/")
    monkeypatch.setenv("AKD_SCHOLAR_SEARCH_API_KEY", "env-key")
    s = AKDScholarSearcher()
    assert s.url == BASE and s.api_key == "env-key"


def test_a_missing_key_says_which_variable(monkeypatch):
    monkeypatch.delenv("AKD_SCHOLAR_SEARCH_API_KEY", raising=False)
    with pytest.raises(MissingAPIKeyError, match="AKD_SCHOLAR_SEARCH_API_KEY"):
        AKDScholarSearcher(url=BASE)


DOI_TEXT = "\nMETHODS\n\nTwo cohorts of women and men."
DOI_RESPONSE = {
    "title": "Long-term aspirin use and colorectal cancer risk", "text": DOI_TEXT,
    "doi": "10.1038/sj.bjc.6603442", "authors": ["S. Larsson"], "extra": {}, "link": "https://doi.org/10.1038/sj.bjc.6603442",
}


def doi_crawler(handler):
    c = AKDDoiCrawler(url=BASE, api_key="test-key")
    c._http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return c


async def test_doi_crawler_reads_the_full_paper_for_a_doi_url():
    seen = {}

    def handler(request):
        seen["path"], seen["auth"] = request.url.path, request.headers["authorization"]
        return httpx.Response(200, json=DOI_RESPONSE)

    f = await doi_crawler(handler).crawl("https://doi.org/10.1038/sj.bjc.6603442")
    assert seen == {"path": "/doi/10.1038/sj.bjc.6603442", "auth": "Bearer test-key"}
    assert f and f.status == 200 and f.crawler == "AKDDoiCrawler"
    assert f.page == {"url": "https://doi.org/10.1038/sj.bjc.6603442", "title": DOI_RESPONSE["title"], "text": DOI_TEXT.strip()}
    assert f.words == len(DOI_TEXT.split())


async def test_doi_crawler_reads_a_publisher_url_by_its_doi():
    seen = {}

    def handler(request):
        seen["path"] = request.url.path
        return httpx.Response(200, json=DOI_RESPONSE)

    f = await doi_crawler(handler).crawl("https://www.nature.com/articles/10.1038/sj.bjc.6603442")
    assert f and seen["path"] == "/doi/10.1038/sj.bjc.6603442"


async def test_doi_crawler_unknown_doi_is_a_fetch_without_a_page():
    f = await doi_crawler(lambda r: httpx.Response(404, json={"detail": "No paper"})).crawl("https://doi.org/10.9999/missing")
    assert not f and f.page is None and f.status == 404


async def test_doi_crawler_skips_urls_without_a_doi():
    def handler(request):
        raise AssertionError("no request for a url without a DOI")

    f = await doi_crawler(handler).crawl("https://example.org/paper.pdf")
    assert not f and f.status is None and f.error == "no DOI"


async def test_doi_crawler_never_raises_on_a_dead_server():
    def handler(request):
        raise httpx.ConnectError("refused")

    f = await doi_crawler(handler).crawl("https://doi.org/10.1000/abc")
    assert not f and f.status is None and f.error == "ConnectError"


async def test_doi_crawler_first_in_a_cascade_hands_non_doi_urls_on():
    class Plain(Crawler):
        async def crawl(self, url):
            return self.mark(Fetch(url=url, page={"url": url, "title": "", "text": "plain page"}, status=200))

    cascade = CascadedCrawler(doi_crawler(lambda r: httpx.Response(200, json=DOI_RESPONSE)), Plain())
    assert (await cascade.crawl("https://doi.org/10.1038/sj.bjc.6603442")).crawler == "AKDDoiCrawler"
    assert (await cascade.crawl("https://example.org/x")).page["text"] == "plain page"


async def test_doi_crawler_raises_on_a_rejected_key_instead_of_a_silent_miss(monkeypatch):
    monkeypatch.setenv("AKD_SCHOLAR_SEARCH_URL", BASE)
    c = doi_crawler(lambda r: httpx.Response(403, json={"detail": "Forbidden"}))
    with pytest.raises(httpx.HTTPStatusError) as err:
        await c.crawl("https://doi.org/10.1038/sj.bjc.6603442")
    assert auth_error(err.value).env == "AKD_SCHOLAR_SEARCH_API_KEY"


async def test_akd_clients_use_a_small_reused_keep_alive_pool(monkeypatch):
    """256 /doi calls at once through httpx's defaults (100 connections, 20 kept alive) took 77s on a Mac, the event
    loop stuck 42s in the OS's socket calls; 32 reused connections: 1.3s, no errors."""
    made = []
    real = httpx.AsyncClient
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kw: made.append(kw) or real(**kw))
    for component in (AKDScholarSearcher(url=BASE, api_key="k"), AKDDoiCrawler(url=BASE, api_key="k", max_connections=8)):
        await component.start()
        await component.stop()
    assert [(kw["limits"].max_connections, kw["limits"].max_keepalive_connections) for kw in made] == [(32, 32), (8, 8)]
    assert all(kw["timeout"].pool == 30.0 for kw in made)  # a queued request waits for a connection, it doesn't fail
