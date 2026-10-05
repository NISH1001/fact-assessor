import json

import httpx
import pytest

from factassessor.resolvers import ArxivResolver, OpenAlexResolver, Resolver, arxiv_id, doi_in


# --- the role ----------------------------------------------------------------------------------------------------

def test_resolver_is_a_runtime_checkable_protocol():
    class Custom:  # no base class needed: anything with `async resolve(url) -> list[str]`
        async def resolve(self, url: str) -> list[str]:
            return []

    assert isinstance(Custom(), Resolver)
    assert isinstance(ArxivResolver(), Resolver) and isinstance(OpenAlexResolver(), Resolver)
    assert not isinstance(object(), Resolver)


# --- arXiv -------------------------------------------------------------------------------------------------------

def test_arxiv_id_from_every_link_form():
    assert arxiv_id("https://arxiv.org/abs/2104.10311") == "2104.10311"
    assert arxiv_id("https://arxiv.org/abs/2104.10311v2") == "2104.10311v2"  # the version search found is kept
    assert arxiv_id("https://arxiv.org/pdf/2104.10311v1.pdf") == "2104.10311v1"
    assert arxiv_id("https://arxiv.org/html/2104.10311v1") == "2104.10311v1"
    assert arxiv_id("https://export.arxiv.org/abs/2104.10311") == "2104.10311"
    assert arxiv_id("https://doi.org/10.48550/arXiv.2104.10311") == "2104.10311"  # arXiv DOIs: straight to arXiv
    assert arxiv_id("https://arxiv.org/abs/hep-th/9901001") == "hep-th/9901001"  # old-style ids
    assert arxiv_id("https://en.wikipedia.org/wiki/ArXiv") is None
    assert arxiv_id("https://example.org/2104.10311") is None  # an id-like number off arXiv is not an arXiv link


async def test_arxiv_proposes_html_then_pdf_without_any_request():
    urls = await ArxivResolver().resolve("https://arxiv.org/abs/2104.10311v2")
    assert urls == ["https://arxiv.org/html/2104.10311v2", "https://arxiv.org/pdf/2104.10311v2"]


async def test_arxiv_order_is_a_parameter():
    urls = await ArxivResolver(prefer="pdf").resolve("https://arxiv.org/abs/2104.10311")
    assert urls == ["https://arxiv.org/pdf/2104.10311", "https://arxiv.org/html/2104.10311"]
    with pytest.raises(ValueError):
        ArxivResolver(prefer="docx")


async def test_arxiv_ignores_other_urls():
    assert await ArxivResolver().resolve("https://en.wikipedia.org/wiki/NASA") == []


# --- OpenAlex ----------------------------------------------------------------------------------------------------

def test_doi_is_found_in_publisher_urls():
    assert doi_in("https://zslpublications.onlinelibrary.wiley.com/doi/full/10.1002/rse2.203") == "10.1002/rse2.203"
    assert doi_in("https://agupubs.onlinelibrary.wiley.com/doi/abs/10.1029/2021GL095922") == "10.1029/2021gl095922"
    assert doi_in("https://iopscience.iop.org/article/10.3847/PSJ/ac75c4/pdf") == "10.3847/psj/ac75c4"
    assert doi_in("https://doi.org/10.1038/nature16512") == "10.1038/nature16512"
    assert doi_in("https://www.nature.com/articles/s41598-025-15585-6") is None  # no DOI in the URL
    assert doi_in("https://en.wikipedia.org/wiki/Marie_Curie") is None


def openalex(routes):
    """An OpenAlexResolver whose HTTP goes to `routes`: {url prefix: (status, json body)}; logs requests."""
    seen = []

    def handler(request):
        seen.append(str(request.url))
        for prefix, (status, body) in routes.items():
            if str(request.url).startswith(prefix):
                return httpx.Response(status, content=json.dumps(body).encode())
        return httpx.Response(404)

    r = OpenAlexResolver()
    r._http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return r, seen


async def test_openalex_lists_open_access_copies_pdfs_first():
    work = {
        "best_oa_location": {"pdf_url": "https://publisher.example/pdf/1", "landing_page_url": "https://publisher.example/1"},
        "locations": [
            {"is_oa": True, "pdf_url": "https://publisher.example/pdf/1"},  # the same copy again: listed once
            {"is_oa": False, "pdf_url": "https://paywalled.example/pdf"},  # not open access: never proposed
            {"is_oa": True, "pdf_url": "https://repository.example/paper.pdf", "landing_page_url": "https://repository.example/1"},
        ],
        "open_access": {"oa_url": "https://repository.example/1"},
    }
    r, seen = openalex({"https://api.openalex.org/works/doi:10.3847/1538-4357/ac1a76": (200, work)})
    urls = await r.resolve("https://iopscience.iop.org/article/10.3847/1538-4357/ac1a76")
    await r.stop()
    assert urls == [
        "https://publisher.example/pdf/1", "https://repository.example/paper.pdf",
        "https://publisher.example/1", "https://repository.example/1",
    ]
    assert seen == ["https://api.openalex.org/works/doi:10.3847/1538-4357/ac1a76"]  # one metadata lookup


async def test_openalex_no_doi_means_no_request():
    r, seen = openalex({})
    assert await r.resolve("https://en.wikipedia.org/wiki/NASA") == [] and seen == []
    await r.stop()


async def test_openalex_leaves_arxiv_dois_to_the_arxiv_resolver():
    r, seen = openalex({})
    assert await r.resolve("https://doi.org/10.48550/arXiv.2104.10311") == [] and seen == []
    await r.stop()


async def test_openalex_failures_are_an_empty_list_not_an_error():
    closed = {"best_oa_location": None, "locations": [], "open_access": {"oa_url": None}}
    r, _ = openalex({"https://api.openalex.org/works/doi:10.1016/j.rse.2013.04.005": (200, closed)})
    assert await r.resolve("https://doi.org/10.1016/j.rse.2013.04.005") == []  # paywalled: no free copy
    assert await r.resolve("https://doi.org/10.9999/unknown") == []  # 404 from OpenAlex
    await r.stop()

    def broken(request):
        raise httpx.ConnectError("down")

    r = OpenAlexResolver()
    r._http = httpx.AsyncClient(transport=httpx.MockTransport(broken))
    assert await r.resolve("https://doi.org/10.1038/nature16512") == []
    await r.stop()


async def test_openalex_drops_doi_org_links_they_only_redirect_to_the_publisher():
    work = {
        "best_oa_location": {"pdf_url": "https://onlinelibrary.wiley.com/doi/pdfdirect/10.1002/rse2.203",
                             "landing_page_url": "https://doi.org/10.1002/rse2.203"},
        "locations": [{"is_oa": True, "pdf_url": None, "landing_page_url": "https://hal.inrae.fr/hal-03193170v1/document"}],
    }
    r, _ = openalex({"https://api.openalex.org/": (200, work)})
    urls = await r.resolve("https://zslpublications.onlinelibrary.wiley.com/doi/full/10.1002/rse2.203")
    await r.stop()
    assert urls == ["https://onlinelibrary.wiley.com/doi/pdfdirect/10.1002/rse2.203", "https://hal.inrae.fr/hal-03193170v1/document"]


# --- CompositeResolver and the order locations are tried in -----------------------------------------------------

async def test_composite_asks_every_resolver_at_once_and_keeps_their_order():
    import asyncio

    class Slow:
        def __init__(self, urls, delay):
            self.urls, self.delay = urls, delay

        async def resolve(self, url):
            await asyncio.sleep(self.delay)
            return self.urls

    from factassessor.resolvers import CompositeResolver

    r = CompositeResolver(Slow(["https://a.org/1", "https://b.org/2"], 0.2), Slow(["https://b.org/2", "https://c.org/3"], 0.2))
    start = asyncio.get_running_loop().time()
    assert await r.resolve("https://x.org") == ["https://a.org/1", "https://b.org/2", "https://c.org/3"]  # no duplicates
    assert asyncio.get_running_loop().time() - start < 0.35  # concurrently, not 0.4s one after the other
    assert isinstance(r, Resolver)


async def test_composite_survives_a_failing_resolver():
    from factassessor.resolvers import CompositeResolver

    class Broken:
        async def resolve(self, url):
            raise RuntimeError("boom")

    assert await CompositeResolver(Broken(), ArxivResolver()).resolve("https://arxiv.org/abs/1706.03762") == [
        "https://arxiv.org/html/1706.03762", "https://arxiv.org/pdf/1706.03762"]


async def test_composite_closes_the_resolvers_that_hold_a_client():
    from factassessor.resolvers import CompositeResolver

    oa = OpenAlexResolver()
    await oa.start()
    await CompositeResolver(ArxivResolver(), oa).aclose()
    assert oa._http is None


def test_locations_direct_pdf_hit_first_ordinary_page_last():
    from factassessor.resolvers import locations

    copies = ["https://repository.example/paper.pdf", "https://hal.example/document"]
    # the hit is itself a PDF: exactly the document search matched, so it comes first
    assert locations("https://site.example/papers/biomass.pdf", copies) == ["https://site.example/papers/biomass.pdf", *copies]
    # an ordinary page (publisher landing page, Wikipedia): after the copies
    assert locations("https://publisher.example/doi/10.1/x", copies) == [*copies, "https://publisher.example/doi/10.1/x"]
    assert locations("https://en.wikipedia.org/wiki/NASA", []) == ["https://en.wikipedia.org/wiki/NASA"]
    assert locations("https://arxiv.org/pdf/1706.03762", ["https://arxiv.org/html/1706.03762", "https://arxiv.org/pdf/1706.03762"]) == [
        "https://arxiv.org/pdf/1706.03762", "https://arxiv.org/html/1706.03762"]  # listed once


async def test_a_pmc_article_resolves_to_its_europe_pmc_full_text_with_no_request():
    # PMC served reCAPTCHA pages to bursts of requests; Europe PMC mirrors every PMC article, and its full-text
    # endpoint read over plain HTTP on 3 of 4 sampled papers (10-14k words each)
    from factassessor.resolvers import PMCResolver

    r = PMCResolver()
    assert isinstance(r, Resolver)
    url = "https://www.ebi.ac.uk/europepmc/webservices/rest/PMC12375767/fullTextXML"
    assert await r.resolve("https://pmc.ncbi.nlm.nih.gov/articles/PMC12375767/") == [url]
    assert await r.resolve("https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12375767") == [url]
    assert await r.resolve("https://example.org/paper") == []


def test_the_default_resolver_includes_pmc():
    from factassessor import FactAssessor
    from factassessor.resolvers import PMCResolver

    assert any(isinstance(r, PMCResolver) for r in FactAssessor().verify.resolver.resolvers)
