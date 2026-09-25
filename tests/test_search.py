import json

import httpx
import pytest

from factassessor import FactAssessor

SERPER_RESPONSE = {
    "organic": [
        {"title": "Marie Curie - Wikipedia", "link": "https://en.wikipedia.org/wiki/Marie_Curie",
         "snippet": "Born in Warsaw in 1867...", "position": 1},
        {"title": "Nobel Prize", "link": "https://www.nobelprize.org/curie", "snippet": "Awarded 1903.", "position": 2},
        {"title": "No snippet here", "link": "https://example.com/x", "position": 3},
    ]
}


def assessor_with(handler, **kwargs):
    fr = FactAssessor(serper_api_key="test-key", **kwargs)
    fr._http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return fr


async def test_search_returns_url_title_snippet_and_sends_query():
    seen = {}

    def handler(request):
        seen["key"] = request.headers["X-API-KEY"]
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json=SERPER_RESPONSE)

    fr = assessor_with(handler, top_k=5)
    hits = await fr._search("Marie Curie was born in Warsaw in 1867.")
    await fr.aclose()

    assert seen == {"key": "test-key", "body": {"q": "Marie Curie was born in Warsaw in 1867.", "num": 10}}
    assert hits == [
        {"url": "https://en.wikipedia.org/wiki/Marie_Curie", "title": "Marie Curie - Wikipedia",
         "snippet": "Born in Warsaw in 1867..."},
        {"url": "https://www.nobelprize.org/curie", "title": "Nobel Prize", "snippet": "Awarded 1903."},
        {"url": "https://example.com/x", "title": "No snippet here", "snippet": ""},
    ]


async def test_search_truncates_to_top_k():
    fr = assessor_with(lambda r: httpx.Response(200, json=SERPER_RESPONSE), top_k=2)
    assert len(await fr._search("q")) == 2
    await fr.aclose()


async def test_search_http_error_raises():
    fr = assessor_with(lambda r: httpx.Response(403, json={"message": "bad key"}))
    with pytest.raises(httpx.HTTPStatusError):
        await fr._search("q")
    await fr.aclose()


async def test_search_without_api_key_fails_clearly(monkeypatch):
    monkeypatch.delenv("SERPER_API_KEY", raising=False)
    fr = FactAssessor()
    with pytest.raises(RuntimeError, match="SERPER_API_KEY"):
        await fr._search("q")


async def test_search_drops_social_media_but_keeps_twitter_and_linkedin_and_refills_to_top_k():
    organic = [
        {"title": "fb", "link": "https://www.facebook.com/oncodaily/videos/1", "snippet": "a"},
        {"title": "ig", "link": "https://instagram.com/p/xyz", "snippet": "b"},
        {"title": "fb mobile", "link": "https://m.facebook.com/story", "snippet": "c"},
        {"title": "x", "link": "https://twitter.com/nasa/status/1", "snippet": "d"},
        {"title": "li", "link": "https://www.linkedin.com/in/someone", "snippet": "e"},
        {"title": "wiki", "link": "https://en.wikipedia.org/wiki/NASA", "snippet": "f"},
        {"title": "tt", "link": "https://www.tiktok.com/@nasa/video/1", "snippet": "g"},
        {"title": "notfacebook", "link": "https://notfacebook.com/page", "snippet": "h"},
    ]
    seen = {}

    def handler(request):
        seen["num"] = json.loads(request.content)["num"]
        return httpx.Response(200, json={"organic": organic})

    fr = assessor_with(handler, top_k=3)
    hits = await fr._search("q")
    await fr.aclose()
    assert seen["num"] == 6  # over-fetch so blocked results don't leave us short
    assert [h["title"] for h in hits] == ["x", "li", "wiki"]


async def test_slow_search_is_hedged_with_a_duplicate_and_the_first_reply_wins():
    import asyncio

    calls = []

    async def handler(request):
        calls.append(len(calls))
        if len(calls) == 1:
            await asyncio.sleep(1.0)  # the tail-latency outlier
        return httpx.Response(200, json=SERPER_RESPONSE)

    fr = assessor_with(handler, search_hedge_after=0.05)
    start = asyncio.get_running_loop().time()
    hits = await fr._search("q")
    elapsed = asyncio.get_running_loop().time() - start
    await fr.aclose()
    assert len(calls) == 2 and elapsed < 0.5 and len(hits) == 3


async def test_fast_search_is_not_hedged():
    calls = []

    def handler(request):
        calls.append(1)
        return httpx.Response(200, json=SERPER_RESPONSE)

    fr = assessor_with(handler, search_hedge_after=1.0)
    await fr._search("q")
    await fr.aclose()
    assert len(calls) == 1


async def test_video_pages_are_blocked_by_default():
    fr = FactAssessor()
    assert fr._is_blocked("https://www.youtube.com/watch?v=x") and fr._is_blocked("https://youtu.be/x")
