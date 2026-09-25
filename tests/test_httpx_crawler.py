import asyncio

import httpx

from factassessor import Crawler, FallbackCrawler, HTTPXCrawler, collect

PAGE = """<html><head><title>Nepal earthquake - Wikipedia</title><script>var x = 1;</script>
<style>body { color: red }</style></head>
<body><nav>Home | About</nav>
<h1>April 2015 Nepal earthquake</h1>
<p>The earthquake struck on <b>25 April 2015</b> with a magnitude of 7.8.</p>
<p>Nearly 9,000 people died.</p>
<footer>Privacy policy</footer></body></html>"""


def crawler(handler, **kwargs):
    c = HTTPXCrawler(**kwargs)
    c._http = httpx.AsyncClient(transport=httpx.MockTransport(handler), follow_redirects=True)
    return c


def html(body=PAGE, status=200, ctype="text/html; charset=utf-8"):
    return lambda request: httpx.Response(status, content=body.encode(), headers={"content-type": ctype})


async def test_extracts_title_and_readable_text_without_markup_or_chrome():
    page = await crawler(html()).crawl("https://en.wikipedia.org/wiki/April_2015_Nepal_earthquake")
    assert page["title"] == "Nepal earthquake - Wikipedia"
    assert "magnitude of 7.8" in page["text"] and "Nearly 9,000 people died." in page["text"]
    for junk in ("var x", "color: red", "Home | About", "Privacy policy", "<p>"):
        assert junk not in page["text"]


async def test_non_html_error_and_empty_responses_are_none():
    assert await crawler(html(ctype="application/pdf")).crawl("https://x.org/a.pdf") is None
    assert await crawler(html(status=404)).crawl("https://x.org/missing") is None
    assert await crawler(html(body="<html><body><script>app()</script></body></html>")).crawl("https://spa.org") is None


async def test_slow_pages_hit_the_total_deadline():
    async def slow(request):
        await asyncio.sleep(5)
        return httpx.Response(200, content=PAGE.encode(), headers={"content-type": "text/html"})

    start = asyncio.get_running_loop().time()
    assert await crawler(slow, timeout=0.1).crawl("https://slow.org") is None
    assert asyncio.get_running_loop().time() - start < 1


async def test_network_errors_are_none_not_exceptions():
    def boom(request):
        raise httpx.ConnectError("dns failed")

    assert await crawler(boom).crawl("https://nope.invalid") is None


async def test_huge_pages_are_cut_off():
    big = "<html><body>" + "<p>word</p>" * 200_000 + "</body></html>"
    page = await crawler(html(body=big), max_bytes=10_000).crawl("https://big.org")
    assert page is not None and len(page["text"]) < 20_000


async def test_is_a_crawler_step():
    c = crawler(html())
    assert isinstance(c, Crawler)

    async def urls():
        yield "https://a.org"
        yield "https://b.org"

    assert len(await collect(c(urls()))) == 2


async def test_fallback_uses_the_next_crawler_only_when_the_first_fails():
    class Fake(Crawler):
        def __init__(self, pages):
            self.pages, self.calls = pages, []

        async def crawl(self, url):
            self.calls.append(url)
            return self.pages.get(url)

    fast = Fake({"https://static.org": {"url": "https://static.org", "title": "", "text": "fast"}})
    browser = Fake({"https://static.org": {"url": "https://static.org", "title": "", "text": "slow"},
                    "https://spa.org": {"url": "https://spa.org", "title": "", "text": "rendered"}})
    both = FallbackCrawler(fast, browser)
    assert (await both.crawl("https://static.org"))["text"] == "fast"
    assert (await both.crawl("https://spa.org"))["text"] == "rendered"
    assert await both.crawl("https://dead.org") is None
    assert browser.calls == ["https://spa.org", "https://dead.org"]  # only for what the fast path couldn't read
