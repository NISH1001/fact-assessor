import asyncio
from types import SimpleNamespace

from factassessor import Crawl4AICrawler, collect


def result(success=True, markdown="# Marie Curie\nBorn in Warsaw in 1867.", title="Marie Curie - Wikipedia"):
    return SimpleNamespace(
        success=success,
        markdown=SimpleNamespace(raw_markdown=markdown) if markdown is not None else None,
        metadata={"title": title},
        error_message=None if success else "net::ERR_NAME_NOT_RESOLVED",
    )


class FakeCrawler:
    def __init__(self, behaviour):
        self.behaviour = behaviour

    async def arun(self, url, config=None):
        return await self.behaviour(url)


def crawler_with(behaviour, **kwargs):
    c = Crawl4AICrawler(**kwargs)
    c._browser = FakeCrawler(behaviour)
    return c


async def test_crawl_returns_url_title_text():
    async def ok(url):
        return result()

    page = await crawler_with(ok).crawl("https://en.wikipedia.org/wiki/Marie_Curie")
    assert page == {
        "url": "https://en.wikipedia.org/wiki/Marie_Curie",
        "title": "Marie Curie - Wikipedia",
        "text": "Marie Curie\nBorn in Warsaw in 1867.",  # cleaned: markdown heading stripped
    }


async def test_crawl_failure_returns_none():
    async def failed(url):
        return result(success=False, markdown=None)

    assert await crawler_with(failed).crawl("https://nope.invalid") is None


async def test_crawl_empty_page_returns_none():
    async def empty(url):
        return result(markdown="   ")

    assert await crawler_with(empty).crawl("https://example.com") is None


async def test_crawl_exception_returns_none():
    async def boom(url):
        raise RuntimeError("browser crashed")

    assert await crawler_with(boom).crawl("https://example.com") is None


async def test_crawl_timeout_returns_none():
    async def slow(url):
        await asyncio.sleep(5)
        return result()

    assert await crawler_with(slow, timeout=0.05).crawl("https://slow.example.com") is None


async def test_crawler_step_drops_failures_and_yields_pages_as_they_finish():
    import asyncio

    async def behaviour(url):
        if "dead" in url:
            return result(success=False, markdown=None)
        await asyncio.sleep(0.05 if "slow" in url else 0)
        return result(title=url)

    async def urls():
        for u in ["https://slow.org", "https://dead.org", "https://fast.org"]:
            yield u

    pages = await collect(crawler_with(behaviour)(urls()))
    assert [p["url"] for p in pages] == ["https://fast.org", "https://slow.org"]


async def test_error_status_pages_are_dropped_so_the_snippet_stands():
    # crawl4ai reports success for pages that loaded with an error status: a recorded PubChem page was
    # "temporarily unavailable (HTTP 503)" and got judged as evidence. Verify has already judged every hit's
    # snippet, so dropping the page leaves the snippet as that hit's evidence.
    async def unavailable(url):
        r = result(markdown="PubChem is temporarily unavailable (HTTP 503)")
        r.status_code = 503
        return r

    async def forbidden(url):
        r = result(markdown="Access denied")
        r.status_code = 403
        return r

    async def ok(url):
        r = result()
        r.status_code = 200
        return r

    assert await crawler_with(unavailable).crawl("https://pubchem.ncbi.nlm.nih.gov/x") is None
    assert await crawler_with(forbidden).crawl("https://example.com/x") is None
    assert (await crawler_with(ok).crawl("https://en.wikipedia.org/wiki/Marie_Curie"))["title"] == "Marie Curie - Wikipedia"
    assert await crawler_with(lambda url: asyncio.sleep(0, result())).crawl("https://a.org") is not None  # no status: kept
