import json

import httpx

from factassessor import FallbackCrawler
from factassessor.crawlers import OpenAccessCrawler
from tests.pdfs import minimal_pdf

WORK = {
    "title": "Biomass resilience of Neotropical secondary forests",
    "best_oa_location": {"pdf_url": "https://europepmc.org/articles/pmc123/pdf", "landing_page_url": "https://europepmc.org/abstract/123"},
}


def crawler(routes, **kwargs):
    """An OpenAccessCrawler whose HTTP goes to `routes`: {url prefix: (status, content-type, body)}; logs requests."""
    seen = []

    def handler(request):
        seen.append(str(request.url))
        for prefix, (status, ctype, body) in routes.items():
            if str(request.url).startswith(prefix):
                return httpx.Response(status, headers={"content-type": ctype}, content=body)
        return httpx.Response(404)

    kwargs.setdefault("min_words", 0)  # the sample texts are short; the stub test sets the real minimum
    c = OpenAccessCrawler(**kwargs)
    c._http = httpx.AsyncClient(transport=httpx.MockTransport(handler), follow_redirects=True)
    return c, seen


async def test_blocked_publisher_page_is_read_from_its_open_access_pdf():
    url = "https://zslpublications.onlinelibrary.wiley.com/doi/full/10.1002/rse2.203"
    c, seen = crawler({
        "https://api.openalex.org/works/doi:10.1002/rse2.203": (200, "application/json", json.dumps(WORK).encode()),
        "https://europepmc.org/articles/pmc123/pdf": (200, "application/pdf", minimal_pdf("Secondary forests gained 122 Mg per ha in 20 years.")),
    })
    page = await c.crawl(url)
    await c.stop()
    assert page == {"url": url, "title": WORK["title"], "text": "Secondary forests gained 122 Mg per ha in 20 years."}
    assert seen[0].startswith("https://api.openalex.org/works/doi:10.1002/rse2.203")


async def test_html_open_access_copy_when_there_is_no_pdf():
    work = {"title": "T", "best_oa_location": {"pdf_url": None, "landing_page_url": "https://arxiv.org/abs/2201.00001"}}
    c, _ = crawler({
        "https://api.openalex.org/": (200, "application/json", json.dumps(work).encode()),
        "https://arxiv.org/abs/2201.00001": (200, "text/html", b"<html><body><p>The full text says 1,468 plots.</p></body></html>"),
    })
    page = await c.crawl("https://doi.org/10.48550/arxiv.2201.00001")
    await c.stop()
    assert page["text"] == "The full text says 1,468 plots."


async def test_no_doi_no_open_access_copy_or_a_failed_fetch_is_none():
    closed = {"title": "T", "best_oa_location": None, "open_access": {"oa_url": None}}
    c, seen = crawler({"https://api.openalex.org/": (200, "application/json", json.dumps(closed).encode())})
    assert await c.crawl("https://en.wikipedia.org/wiki/NASA") is None and seen == []  # no DOI: no request at all
    assert await c.crawl("https://doi.org/10.1016/j.rse.2013.04.005") is None  # paywalled, no free copy
    assert seen == ["https://api.openalex.org/works/doi:10.1016/j.rse.2013.04.005"]  # it did look it up
    await c.stop()
    c, _ = crawler({
        "https://api.openalex.org/": (200, "application/json", json.dumps(WORK).encode()),
        "https://europepmc.org/": (403, "text/html", b"blocked"),
    })
    assert await c.crawl("https://doi.org/10.1002/rse2.203") is None
    await c.stop()


async def test_fallback_tries_open_access_only_when_the_browser_fails():
    class Blocked:
        async def crawl(self, url):
            return None

    url = "https://iopscience.iop.org/article/10.3847/PSJ/ac75c4"
    c, _ = crawler({
        "https://api.openalex.org/": (200, "application/json", json.dumps(WORK).encode()),
        "https://europepmc.org/": (200, "application/pdf", minimal_pdf("Recurring slope lineae appear in warm seasons.")),
    })
    page = await FallbackCrawler(Blocked(), c).crawl(url)
    await c.stop()
    assert page["url"] == url and "warm seasons" in page["text"]


async def test_other_open_access_copies_are_tried_and_stub_pages_rejected():
    # live: OpenAlex's "best" copy for Wiley/IOP papers is often the publisher's own PDF, which serves a 384-char
    # bot-check page; a repository or author copy listed under `locations` has the real text
    work = {
        "title": "T",
        "best_oa_location": {"pdf_url": "https://publisher.example/pdf/1"},
        "locations": [
            {"is_oa": True, "pdf_url": "https://publisher.example/pdf/1"},
            {"is_oa": False, "pdf_url": "https://paywalled.example/pdf"},
            {"is_oa": True, "pdf_url": "https://repository.example/paper.pdf", "landing_page_url": "https://repository.example/1"},
        ],
    }
    body = minimal_pdf(*["Recurring slope lineae lengthen during the warm season."] * 30)
    c, seen = crawler({
        "https://api.openalex.org/": (200, "application/json", json.dumps(work).encode()),
        "https://publisher.example/pdf/1": (200, "application/pdf", minimal_pdf("Please verify you are a human.")),  # a stub
        "https://repository.example/paper.pdf": (200, "application/pdf", body),
    }, min_words=100)
    page = await c.crawl("https://doi.org/10.3847/1538-4357/ac1a76")
    await c.stop()
    assert page is not None and "warm season" in page["text"]
    assert "https://paywalled.example/pdf" not in seen  # only open-access locations


# --- direct PDFs and arXiv: papers found by scholarly search (61% of Google Scholar hits were PDF links) ------------

async def test_direct_pdf_links_are_read():
    url = "https://repository.example/papers/biomass.pdf"
    c, seen = crawler({url: (200, "application/pdf", minimal_pdf("Secondary forests gained 122 Mg per ha in 20 years."))})
    page = await c.crawl(url)
    await c.stop()
    assert page["text"] == "Secondary forests gained 122 Mg per ha in 20 years." and seen == [url]


async def test_arxiv_links_read_the_full_paper_not_the_abstract_page():
    c, seen = crawler({"https://arxiv.org/pdf/2104.10311": (200, "application/pdf", minimal_pdf("Full text of the paper."))})
    for url in ("https://arxiv.org/abs/2104.10311", "https://arxiv.org/abs/2104.10311v2", "https://arxiv.org/html/2104.10311v1"):
        page = await c.crawl(url)
        assert page == {"url": url, "title": "", "text": "Full text of the paper."}
    await c.stop()
    assert set(seen) == {"https://arxiv.org/pdf/2104.10311"}


async def test_a_pdf_link_that_serves_a_bot_page_falls_back_to_open_access():
    url = "https://iopscience.iop.org/article/10.3847/1538-4357/ac1a76/pdf"
    work = {"title": "T", "best_oa_location": {"pdf_url": "https://arxiv.org/pdf/2106.00001"}}
    c, seen = crawler({
        url: (200, "text/html", b"<html><body>Radware bot check</body></html>"),
        "https://api.openalex.org/": (200, "application/json", json.dumps(work).encode()),
        "https://arxiv.org/pdf/2106.00001": (200, "application/pdf", minimal_pdf("The real paper text, long enough to count.")),
    }, min_words=5)
    page = await c.crawl(url)
    await c.stop()
    assert page["text"].startswith("The real paper") and page["url"] == url
