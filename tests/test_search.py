import asyncio
import json

import httpx
import pytest

from factassessor import Filter, Serper, Take, collect, is_blocked, not_blocked, once

SERPER_RESPONSE = {
    "organic": [
        {"title": "Marie Curie - Wikipedia", "link": "https://en.wikipedia.org/wiki/Marie_Curie",
         "snippet": "Born in Warsaw in 1867...", "position": 1},
        {"title": "Nobel Prize", "link": "https://www.nobelprize.org/curie", "snippet": "Awarded 1903.", "position": 2},
        {"title": "No snippet here", "link": "https://example.com/x", "position": 3},
    ]
}


def serper(handler, **kwargs):
    s = Serper(api_key="test-key", **kwargs)
    s._http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return s


async def test_search_sends_the_query_and_returns_hits_in_rank_order():
    seen = {}

    def handler(request):
        seen["key"] = request.headers["X-API-KEY"]
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json=SERPER_RESPONSE)

    s = serper(handler, num=10)
    hits = await s.search("Marie Curie was born in Warsaw in 1867.")
    await s.stop()
    assert seen == {"key": "test-key", "body": {"q": "Marie Curie was born in Warsaw in 1867.", "num": 10}}
    assert hits == [
        {"url": "https://en.wikipedia.org/wiki/Marie_Curie", "title": "Marie Curie - Wikipedia",
         "snippet": "Born in Warsaw in 1867..."},
        {"url": "https://www.nobelprize.org/curie", "title": "Nobel Prize", "snippet": "Awarded 1903."},
        {"url": "https://example.com/x", "title": "No snippet here", "snippet": ""},
    ]


async def test_serper_is_a_step_that_chains_with_filter_and_take():
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
    s = serper(lambda r: httpx.Response(200, json={"organic": organic}))
    searcher = s >> Filter(not_blocked()) >> Take(3)
    hits = await collect(searcher(once("q")))
    await s.stop()
    assert [h["title"] for h in hits] == ["x", "li", "wiki"]  # social dropped, twitter/linkedin kept, rank kept


def test_blocked_domains_cover_subdomains_but_not_lookalikes():
    assert is_blocked("https://m.facebook.com/x") and is_blocked("https://youtu.be/x")
    assert not is_blocked("https://notfacebook.com/x") and not is_blocked("https://twitter.com/x")
    assert not_blocked(("example.com",))({"url": "https://facebook.com/x"})  # custom list


async def test_http_error_raises():
    s = serper(lambda r: httpx.Response(403, json={"message": "bad key"}))
    with pytest.raises(httpx.HTTPStatusError):
        await s.search("q")
    await s.stop()


async def test_missing_api_key_fails_clearly(monkeypatch):
    monkeypatch.delenv("SERPER_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="SERPER_API_KEY"):
        await Serper().search("q")


async def test_slow_search_is_hedged_with_a_duplicate_and_the_first_reply_wins():
    calls = []

    async def handler(request):
        calls.append(len(calls))
        if len(calls) == 1:
            await asyncio.sleep(1.0)  # the tail-latency outlier
        return httpx.Response(200, json=SERPER_RESPONSE)

    s = serper(handler, hedge_after=0.05)
    start = asyncio.get_running_loop().time()
    hits = await s.search("q")
    elapsed = asyncio.get_running_loop().time() - start
    await s.stop()
    assert len(calls) == 2 and elapsed < 0.5 and len(hits) == 3


async def test_fast_search_is_not_hedged():
    calls = []

    def handler(request):
        calls.append(1)
        return httpx.Response(200, json=SERPER_RESPONSE)

    s = serper(handler, hedge_after=1.0)
    await s.search("q")
    await s.stop()
    assert len(calls) == 1
