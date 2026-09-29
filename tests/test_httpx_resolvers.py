"""HTTPXCrawler with resolvers: a paper's full text from its free copies (arXiv, OpenAlex), then the page itself."""

import json

import httpx

from factassessor import ArxivResolver, FallbackCrawler, HTTPXCrawler, OpenAlexResolver
from tests.pdfs import minimal_pdf

WORK = {
    "title": "Biomass resilience of Neotropical secondary forests",
    "best_oa_location": {"pdf_url": "https://europepmc.org/articles/pmc123/pdf", "landing_page_url": "https://europepmc.org/abstract/123"},
}
PAPER = minimal_pdf(*["Secondary forests gained 122 Mg per ha in 20 years."] * 40)  # 400 words: a real paper
ABSTRACT_PAGE = "<html><body>" + "<p>An abstract page with enough words to read on its own.</p>" * 15 + "</body></html>"


def crawler(routes, **kwargs):
    """HTTPXCrawler(resolvers=[arXiv, OpenAlex]) whose HTTP goes to `routes`: {url prefix: (status, type, body)}."""
    seen = []

    def handler(request):
        seen.append(str(request.url))
        for prefix, (status, ctype, body) in routes.items():
            if str(request.url).startswith(prefix):
                return httpx.Response(status, headers={"content-type": ctype}, content=body)
        return httpx.Response(404)

    transport = httpx.MockTransport(handler)
    openalex = OpenAlexResolver()
    openalex._http = httpx.AsyncClient(transport=transport)
    c = HTTPXCrawler(resolvers=[ArxivResolver(), openalex], **kwargs)
    c._http = httpx.AsyncClient(transport=transport, follow_redirects=True)
    return c, seen


async def test_blocked_publisher_page_is_read_from_its_open_access_pdf():
    url = "https://zslpublications.onlinelibrary.wiley.com/doi/full/10.1002/rse2.203"
    c, seen = crawler({
        "https://api.openalex.org/works/doi:10.1002/rse2.203": (200, "application/json", json.dumps(WORK).encode()),
        "https://europepmc.org/articles/pmc123/pdf": (200, "application/pdf", PAPER),
        url: (403, "text/html", b"Just a moment..."),
    })
    page = await c.crawl(url)
    await c.stop()
    assert page["url"] == url and "122 Mg per ha" in page["text"]  # cited under the URL search found
    assert seen[:2] == ["https://api.openalex.org/works/doi:10.1002/rse2.203", "https://europepmc.org/articles/pmc123/pdf"]
    assert url not in seen  # the free copy worked: the blocked page is never requested


async def test_arxiv_html_first_then_pdf_when_there_is_no_html():
    html = "<html><body>" + "<p>The HTML version of the paper, with its sections.</p>" * 50 + "</body></html>"
    c, seen = crawler({"https://arxiv.org/html/2401.00001": (200, "text/html", html.encode())})
    page = await c.crawl("https://arxiv.org/abs/2401.00001")
    assert "HTML version of the paper" in page["text"] and seen == ["https://arxiv.org/html/2401.00001"]
    await c.stop()

    c, seen = crawler({"https://arxiv.org/pdf/1706.03762": (200, "application/pdf", PAPER)})  # no HTML: 404
    page = await c.crawl("https://arxiv.org/abs/1706.03762")
    await c.stop()
    assert "122 Mg per ha" in page["text"]
    assert seen == ["https://arxiv.org/html/1706.03762", "https://arxiv.org/pdf/1706.03762"]


async def test_stub_copies_are_skipped_for_the_next_one():
    # live: Wiley's own PDF and HAL serve ~180-word bot-check pages with status 200; a repository copy has the text
    bot = ("<html><body>" + "<p>Making sure you're not a bot! Loading...</p>" * 30 + "</body></html>").encode()  # 180 words
    work = {
        "best_oa_location": {"pdf_url": "https://hal.example/document"},
        "locations": [{"is_oa": True, "pdf_url": "https://repository.example/paper.pdf"}],
    }
    c, seen = crawler({
        "https://api.openalex.org/": (200, "application/json", json.dumps(work).encode()),
        "https://hal.example/document": (200, "text/html", bot),
        "https://repository.example/paper.pdf": (200, "application/pdf", PAPER),
    })
    page = await c.crawl("https://doi.org/10.3847/1538-4357/ac1a76")
    await c.stop()
    assert "122 Mg per ha" in page["text"] and "https://repository.example/paper.pdf" in seen


async def test_the_page_itself_is_the_last_candidate_with_a_lower_bar():
    # no free copy of the paper: the landing page (an abstract, 150 words) is still evidence
    closed = {"best_oa_location": None, "locations": []}
    url = "https://doi.org/10.1016/j.rse.2013.04.005"
    c, seen = crawler({
        "https://api.openalex.org/": (200, "application/json", json.dumps(closed).encode()),
        url: (200, "text/html", ABSTRACT_PAGE.encode()),
    })
    page = await c.crawl(url)
    await c.stop()
    assert "abstract page" in page["text"] and seen == ["https://api.openalex.org/works/doi:10.1016/j.rse.2013.04.005", url]


async def test_ordinary_pages_cost_no_resolver_request():
    c, seen = crawler({"https://en.wikipedia.org/wiki/NASA": (200, "text/html", ABSTRACT_PAGE.encode())})
    assert await c.crawl("https://en.wikipedia.org/wiki/NASA") is not None
    await c.stop()
    assert seen == ["https://en.wikipedia.org/wiki/NASA"]


async def test_direct_pdf_links_are_read_as_the_page_itself():
    url = "https://repository.example/directbitstream/9813d075/biomass.pdf"
    c, seen = crawler({url: (200, "application/octet-stream", PAPER)})
    page = await c.crawl(url)
    await c.stop()
    assert "122 Mg per ha" in page["text"] and seen == [url]


async def test_a_search_hit_that_is_already_a_candidate_is_fetched_once():
    c, seen = crawler({"https://arxiv.org/pdf/1706.03762": (200, "application/pdf", PAPER)})
    assert await c.crawl("https://arxiv.org/pdf/1706.03762") is not None
    await c.stop()
    assert seen.count("https://arxiv.org/pdf/1706.03762") == 1


async def test_nothing_readable_is_none_so_the_next_crawler_tries():
    class Browser:
        async def crawl(self, url):
            return {"url": url, "title": "", "text": "rendered by the browser"}

    url = "https://doi.org/10.1002/rse2.203"
    c, _ = crawler({
        "https://api.openalex.org/": (200, "application/json", json.dumps(WORK).encode()),
        "https://europepmc.org/": (403, "text/html", b"blocked"),
        url: (403, "text/html", b"Just a moment..."),
    })
    assert await c.crawl(url) is None
    assert (await FallbackCrawler(c, Browser()).crawl(url))["text"] == "rendered by the browser"
    await c.stop()


async def test_a_failing_resolver_never_breaks_the_crawl():
    class Broken:
        async def resolve(self, url):
            raise RuntimeError("boom")

    c = HTTPXCrawler(resolvers=[Broken()])
    c._http = httpx.AsyncClient(transport=httpx.MockTransport(
        lambda r: httpx.Response(200, headers={"content-type": "text/html"}, content=ABSTRACT_PAGE.encode())))
    assert (await c.crawl("https://example.org"))["url"] == "https://example.org"
    await c.stop()


async def test_stop_closes_the_resolvers_too():
    openalex = OpenAlexResolver()
    await openalex.start()
    c = HTTPXCrawler(resolvers=[openalex])
    await c.stop()
    assert openalex._http is None
