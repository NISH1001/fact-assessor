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

    page = (await crawler_with(ok).crawl("https://en.wikipedia.org/wiki/Marie_Curie")).page
    assert page == {
        "url": "https://en.wikipedia.org/wiki/Marie_Curie",
        "title": "Marie Curie - Wikipedia",
        "text": "Marie Curie\nBorn in Warsaw in 1867.",  # cleaned: markdown heading stripped
    }


async def test_crawl_failure_returns_none():
    async def failed(url):
        return result(success=False, markdown=None)

    assert not await crawler_with(failed).crawl("https://nope.invalid")


async def test_crawl_empty_page_returns_none():
    async def empty(url):
        return result(markdown="   ")

    assert not await crawler_with(empty).crawl("https://example.com")


async def test_crawl_exception_returns_none():
    async def boom(url):
        raise RuntimeError("browser crashed")

    assert not await crawler_with(boom).crawl("https://example.com")


async def test_crawl_timeout_returns_none():
    async def slow(url):
        await asyncio.sleep(5)
        return result()

    assert not await crawler_with(slow, timeout=0.05).crawl("https://slow.example.com")


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

    assert not await crawler_with(unavailable).crawl("https://pubchem.ncbi.nlm.nih.gov/x")
    assert not await crawler_with(forbidden).crawl("https://example.com/x")
    assert (await crawler_with(ok).crawl("https://en.wikipedia.org/wiki/Marie_Curie")).page["title"] == "Marie Curie - Wikipedia"
    assert await crawler_with(lambda url: asyncio.sleep(0, result())).crawl("https://a.org")  # no status: kept


# --- crawl: the Fetch, every fact about the render --------------------------------------------------------

async def test_crawl_records_the_rendered_page_and_its_status():
    from factassessor.crawlers import Fetch, HasPage

    async def ok(url):
        r = result(); r.status_code = 200; r.html = "<html><h1>Marie Curie</h1></html>"
        return r

    f = await crawler_with(ok).crawl("https://en.wikipedia.org/wiki/Marie_Curie")
    assert isinstance(f, Fetch) and f and HasPage()(f) and (f.status, f.content_type, f.crawler) == (200, "text/html", "Crawl4AICrawler")
    assert f.words == len(f.page["text"].split()) and "<h1>" in f.body_head


async def test_an_error_status_render_is_kept_and_marked_unusable():
    async def blocked(url):
        r = result(markdown="Access denied"); r.status_code = 403; r.html = "<html>Just a moment...</html>"
        return r

    f = await crawler_with(blocked).crawl("https://publisher.org/paper")
    assert f.status == 403 and "Just a moment" in f.body_head
    assert f.page["text"] == "Access denied" and not f.usable and not f


async def test_crawl_records_why_there_was_no_render():
    async def boom(url):
        raise RuntimeError("net::ERR_NAME_NOT_RESOLVED")

    f = await crawler_with(boom).crawl("https://dead.org")
    assert (f.status, f.page, f.error) == (None, None, "RuntimeError")


async def test_a_successful_render_without_a_status_counts_as_200():
    async def ok(url):
        return result()  # crawl4ai sometimes reports no status_code for a page it loaded fine

    assert (await crawler_with(ok).crawl("https://x.org")).page["title"] == "Marie Curie - Wikipedia"
