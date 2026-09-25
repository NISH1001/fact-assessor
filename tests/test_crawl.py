import asyncio
from types import SimpleNamespace

from factassessor import FactAssessor


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


def assessor_with(behaviour, **kwargs):
    fr = FactAssessor(**kwargs)
    fr._crawler = FakeCrawler(behaviour)
    return fr


async def test_crawl_returns_url_title_text():
    async def ok(url):
        return result()

    page = await assessor_with(ok)._crawl("https://en.wikipedia.org/wiki/Marie_Curie")
    assert page == {
        "url": "https://en.wikipedia.org/wiki/Marie_Curie",
        "title": "Marie Curie - Wikipedia",
        "text": "Marie Curie\nBorn in Warsaw in 1867.",  # cleaned: markdown heading stripped
    }


async def test_crawl_failure_returns_none():
    async def failed(url):
        return result(success=False, markdown=None)

    assert await assessor_with(failed)._crawl("https://nope.invalid") is None


async def test_crawl_empty_page_returns_none():
    async def empty(url):
        return result(markdown="   ")

    assert await assessor_with(empty)._crawl("https://example.com") is None


async def test_crawl_exception_returns_none():
    async def boom(url):
        raise RuntimeError("browser crashed")

    assert await assessor_with(boom)._crawl("https://example.com") is None


async def test_crawl_timeout_returns_none():
    async def slow(url):
        await asyncio.sleep(5)
        return result()

    assert await assessor_with(slow, crawl_timeout=0.05)._crawl("https://slow.example.com") is None
