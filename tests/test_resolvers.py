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
