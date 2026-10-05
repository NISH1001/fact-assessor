import asyncio

import httpx

from factassessor import Crawler, CascadedCrawler, HTTPXCrawler, collect
from tests.pdfs import minimal_pdf

PAGE = """<html><head><title>Nepal earthquake - Wikipedia</title><script>var x = 1;</script>
<style>body { color: red }</style></head>
<body><nav>Home | About</nav>
<h1>April 2015 Nepal earthquake</h1>
<p>The earthquake struck on <b>25 April 2015</b> with a magnitude of 7.8.</p>
<p>Nearly 9,000 people died.</p>
<footer>Privacy policy</footer></body></html>"""


def crawler(handler, **kwargs):
    kwargs.setdefault("min_words", 0)  # the sample pages are short; test_short_pages_are_not_documents sets it
    c = HTTPXCrawler(**kwargs)
    c._http = httpx.AsyncClient(transport=httpx.MockTransport(handler), follow_redirects=True)
    return c


def html(body=PAGE, status=200, ctype="text/html; charset=utf-8"):
    return lambda request: httpx.Response(status, content=body.encode(), headers={"content-type": ctype})


async def test_extracts_title_and_readable_text_without_markup_or_chrome():
    page = (await crawler(html()).crawl("https://en.wikipedia.org/wiki/April_2015_Nepal_earthquake")).page
    assert page["title"] == "Nepal earthquake - Wikipedia"
    assert "magnitude of 7.8" in page["text"] and "Nearly 9,000 people died." in page["text"]
    for junk in ("var x", "color: red", "Home | About", "Privacy policy", "<p>"):
        assert junk not in page["text"]


def raw(body: bytes, ctype: str):
    return lambda request: httpx.Response(200, content=body, headers={"content-type": ctype})


async def test_pdfs_are_read_too_by_content_type_or_their_bytes():
    pdf = minimal_pdf("Secondary forests gained 122 Mg per ha in 20 years.")
    for ctype in ("application/pdf", "application/octet-stream", ""):  # repositories often mislabel PDFs
        page = (await crawler(raw(pdf, ctype)).crawl("https://repositorio.example/directbitstream/9813d075")).page
        assert page == {"url": "https://repositorio.example/directbitstream/9813d075", "title": "",
                        "text": "Secondary forests gained 122 Mg per ha in 20 years."}


async def test_pdfs_get_a_larger_size_limit_than_pages():
    pdf = minimal_pdf("A long paper.")
    assert await crawler(raw(pdf, "application/pdf"), max_bytes=100).crawl("https://x.org/a.pdf")
    assert not await crawler(raw(pdf, "application/pdf"), max_pdf_bytes=100).crawl("https://x.org/a.pdf")  # cut: unreadable


async def test_non_html_error_and_empty_responses_are_none():
    assert not await crawler(raw(b"\x89PNG....", "image/png")).crawl("https://x.org/a.png")
    assert not await crawler(raw(b'{"a": 1}', "application/json")).crawl("https://x.org/api")
    assert not await crawler(html(status=404)).crawl("https://x.org/missing")
    assert not await crawler(html(body="<html><body><script>app()</script></body></html>")).crawl("https://spa.org")


async def test_slow_pages_hit_the_total_deadline():
    async def slow(request):
        await asyncio.sleep(5)
        return httpx.Response(200, content=PAGE.encode(), headers={"content-type": "text/html"})

    start = asyncio.get_running_loop().time()
    assert not await crawler(slow, timeout=0.1).crawl("https://slow.org")
    assert asyncio.get_running_loop().time() - start < 1


async def test_network_errors_are_none_not_exceptions():
    def boom(request):
        raise httpx.ConnectError("dns failed")

    assert not await crawler(boom).crawl("https://nope.invalid")


async def test_huge_pages_are_cut_off():
    big = "<html><body>" + "<p>word</p>" * 200_000 + "</body></html>"
    page = (await crawler(html(body=big), max_bytes=10_000).crawl("https://big.org")).page
    assert page is not None and len(page["text"]) < 20_000


async def test_is_a_crawler_step():
    c = crawler(html())
    assert isinstance(c, Crawler)

    async def urls():
        yield "https://a.org"
        yield "https://b.org"

    assert len(await collect(c(urls()))) == 2


async def test_fallback_uses_the_next_crawler_only_when_the_first_fails():
    from factassessor.crawlers import Fetch

    class Fake(Crawler):
        def __init__(self, pages):
            self.pages, self.calls = pages, []

        async def crawl(self, url):
            self.calls.append(url)
            text = self.pages.get(url)
            return Fetch(url=url, status=200 if text else 404, page={"url": url, "title": "", "text": text} if text else None)

    fast = Fake({"https://static.org": "fast"})
    browser = Fake({"https://static.org": "slow", "https://spa.org": "rendered"})
    both = CascadedCrawler(fast, browser)
    assert (await both.crawl("https://static.org")).page["text"] == "fast"
    assert (await both.crawl("https://spa.org")).page["text"] == "rendered"
    assert not await both.crawl("https://dead.org")
    assert browser.calls == ["https://spa.org", "https://dead.org"]  # only for what the fast path couldn't read


async def test_short_pages_are_not_documents():
    # login walls, "Loading..." shells, browser checks: on 2,748 crawled pages, those under 100 words were junk
    shell = "<html><body><p>Checking your browser before accessing pubmed.ncbi.nlm.nih.gov...</p></body></html>"
    assert not await crawler(html(body=shell), min_words=100).crawl("https://pubmed.ncbi.nlm.nih.gov/1/")
    assert HTTPXCrawler().min_words == 100


async def test_pdfs_get_more_time_than_pages_once_the_response_says_pdf():
    async def slow_body(body):
        await asyncio.sleep(0.3)  # the headers arrive fast; the body (a several-MB paper) takes a while
        yield body

    def slow(ctype, body):
        return lambda request: httpx.Response(200, headers={"content-type": ctype}, content=slow_body(body))

    pdf = minimal_pdf("A paper that takes a while to download.")
    assert await crawler(slow("application/pdf", pdf), timeout=0.1, pdf_timeout=2).crawl("https://x.org/a.pdf")
    assert not await crawler(slow("text/html", PAGE.encode()), timeout=0.1, pdf_timeout=2).crawl("https://x.org/")


# --- crawl: the Fetch, every fact about the response, and whether accept passed ----------------------------------

async def test_crawl_records_a_page_with_its_status_type_word_count_and_crawler():
    from factassessor.crawlers import Fetch

    f = await crawler(html()).crawl("https://en.wikipedia.org/wiki/Nepal")
    assert isinstance(f, Fetch) and f and f.usable and f.crawler == "HTTPXCrawler"
    assert (f.status, f.content_type, f.page["title"]) == (200, "text/html", "Nepal earthquake - Wikipedia")
    assert f.words == len(f.page["text"].split()) and "<h1>" in f.body_head


async def test_a_page_accept_turns_down_is_kept_and_marked_unusable():
    # a JavaScript shell: 200, html, almost no text. The page stays as fetched; the 100-word rule marks it unusable
    shell = "<html><body><div id='root'></div><p>Loading...</p><script>app()</script></body></html>"
    f = await crawler(html(body=shell), min_words=100).crawl("https://spa.org")
    assert (f.status, f.words, f.page["text"], f.usable) == (200, 1, "Loading...", False)
    assert not f  # a page, but not a usable one


async def test_crawl_records_error_statuses_with_the_start_of_the_body():
    challenge = "<html><title>Just a moment...</title><body>Checking your browser. cf-chl-bypass</body></html>"
    f = await crawler(html(body=challenge, status=403)).crawl("https://publisher.org/paper")
    assert (f.status, f.page, f.content_type) == (403, None, "text/html") and "Just a moment" in f.body_head
    f = await crawler(html(status=404)).crawl("https://x.org/missing")
    assert (f.status, f.page, bool(f)) == (404, None, False)


async def test_crawl_records_why_there_was_no_response():
    def boom(request):
        raise httpx.ConnectError("refused")

    f = await crawler(boom).crawl("https://dead.org")
    assert (f.status, f.page, f.error, f.crawler) == (None, None, "ConnectError", "HTTPXCrawler")


async def test_a_url_is_fetched_once_and_then_cached():
    calls = []

    def counted(request):
        calls.append(request.url)
        return httpx.Response(200, content=PAGE.encode(), headers={"content-type": "text/html"})

    c = crawler(counted)
    await c.crawl("https://a.org"); await c.crawl("https://a.org")
    assert len(calls) == 1


# --- CascadedCrawler(when=...): which failures go on to the next crawler --------------------------------------

class Counting(Crawler):
    """The browser: counts what it is asked to render, returns a page for everything."""

    def __init__(self):
        super().__init__()
        self.asked = []

    async def crawl(self, url):
        from factassessor.crawlers import Fetch

        self.asked.append(url)
        return self.mark(Fetch(url=url, status=200, words=200, page={"url": url, "title": "", "text": "rendered " * 200}))


def responses(table):
    return lambda request: table[str(request.url)]()


SITES = {
    "https://good.org/": lambda: httpx.Response(200, content=("<p>" + "word " * 300 + "</p>").encode(), headers={"content-type": "text/html"}),
    "https://spa.org/": lambda: httpx.Response(200, content=b"<div id='root'></div><p>Loading...</p>", headers={"content-type": "text/html"}),
    "https://pub.org/": lambda: httpx.Response(403, content=b"<title>Just a moment...</title> cf-chl", headers={"content-type": "text/html"}),
    "https://gone.org/": lambda: httpx.Response(404, content=b"not found", headers={"content-type": "text/html"}),
    "https://paid.org/": lambda: httpx.Response(402, content=b"payment required", headers={"content-type": "text/html"}),
}


async def test_when_none_sends_every_failure_on_as_before():
    from factassessor.crawlers import CascadedCrawler

    browser = Counting()
    f = CascadedCrawler(crawler(responses(SITES), min_words=100), browser)
    for url in SITES:
        await f.crawl(url)
    assert sorted(browser.asked) == sorted(u for u in SITES if u != "https://good.org/")


async def test_when_sends_only_the_failures_it_names_and_the_successes_never():
    from factassessor.crawlers import CascadedCrawler, NeedsBrowser

    browser = Counting()
    f = CascadedCrawler(crawler(responses(SITES), min_words=100), browser, when=NeedsBrowser())
    pages = {url: await f.crawl(url) for url in SITES}
    assert sorted(browser.asked) == ["https://pub.org/", "https://spa.org/"]  # the shell and the bot check only
    assert pages["https://good.org/"].page["text"].startswith("word") and pages["https://spa.org/"].page["text"].startswith("rendered")
    assert not pages["https://gone.org/"] and not pages["https://paid.org/"]  # no tab spent on them


async def test_when_takes_an_async_predicate_too():
    from factassessor import Predicate
    from factassessor.crawlers import CascadedCrawler

    async def only_spa(fetch):
        return "spa" in fetch.url

    browser = Counting()
    f = CascadedCrawler(crawler(responses(SITES), min_words=100), browser, when=Predicate(only_spa))
    for url in SITES:
        await f.crawl(url)
    assert browser.asked == ["https://spa.org/"]


async def test_a_failure_with_no_response_goes_on_only_if_when_says_so():
    from factassessor.crawlers import CascadedCrawler, Fetch, NeedsBrowser

    class Dead(Crawler):  # no response at all: no status, no body
        async def crawl(self, url):
            return self.mark(Fetch(url=url, error="ConnectError"))

    browser = Counting()
    f = await CascadedCrawler(Dead(), browser, when=NeedsBrowser()).crawl("https://x.org")
    assert browser.asked == [] and f.crawler == "Dead" and f.error == "ConnectError"  # the browser can't reach it either
    await CascadedCrawler(Dead(), browser).crawl("https://x.org")
    assert browser.asked == ["https://x.org"]  # when=None: every failure goes on


def test_each_crawler_has_an_accept_rule_defaulting_to_todays_behaviour():
    from factassessor.crawlers import Crawl4AICrawler, HasPage, MinWords

    long_page = Fetch_(page=True, status=200, words=150)
    short_page = Fetch_(page=True, status=200, words=40)
    assert HTTPXCrawler().accept(long_page) and not HTTPXCrawler().accept(short_page)    # 2xx + 100 words, as before
    assert not HTTPXCrawler(min_words=200).accept(long_page)                              # min_words still works
    assert HTTPXCrawler(accept=HasPage()).accept(short_page)                              # or any rule
    assert Crawl4AICrawler().accept(short_page) and not Crawl4AICrawler().accept(Fetch_(page=False, status=200))
    assert Crawl4AICrawler(accept=HasPage() & MinWords(100)).accept(long_page)


async def test_mark_records_the_crawler_and_applies_accept_and_no_rule_means_usable():
    from factassessor.crawlers import Fetch, MinWords

    class Three(Crawler):
        async def crawl(self, url):
            return self.mark(Fetch(url=url, status=200, words=3, page={"url": url, "title": "", "text": "three words here"}))

    f = await Three().crawl("u")
    assert f and f.usable and f.crawler == "Three"           # no accept rule: whatever was fetched is usable
    f = await Three(accept=MinWords(10)).crawl("u")
    assert not f and not f.usable and f.page["text"] == "three words here"   # turned down, page kept


def Fetch_(page, status, words=0):
    from factassessor.crawlers import Fetch

    return Fetch(url="u", status=status, words=words, page={"url": "u", "title": "", "text": "w " * words} if page else None)


async def test_when_is_checked_after_every_crawler_of_a_longer_cascade():
    from factassessor.crawlers import CascadedCrawler, Fetch, StatusIn

    class Reporting(Crawler):
        """A page for the urls in `pages`, else the status given."""

        def __init__(self, pages, statuses):
            super().__init__()
            self.pages, self.statuses, self.asked = pages, statuses, []

        async def crawl(self, url):
            self.asked.append(url)
            if url in self.pages:
                return self.mark(Fetch(url=url, status=200, page={"url": url, "title": "", "text": "text " * 200}, words=200))
            return self.mark(Fetch(url=url, status=self.statuses.get(url, 500)))

    first = Reporting({"a"}, {"b": 403, "c": 404, "d": 403})
    second = Reporting({"b"}, {"d": 404})
    third = Counting()
    cascade = CascadedCrawler(first, second, third, when=~StatusIn(404))  # pass on anything but a 404
    fetches = {u: await cascade.crawl(u) for u in "abcd"}
    assert first.asked == list("abcd")
    assert second.asked == ["b", "d"]          # a: first had it; c: first's 404 stops it
    assert third.asked == []                   # b: second had it; d: second's 404 stops it
    assert fetches["a"] and fetches["b"] and not fetches["c"] and not fetches["d"]
    assert (fetches["c"].status, fetches["d"].status) == (404, 404)   # why each stopped
    # cascades nest: the inner one returns the Fetch it stopped at, so the outer when can decide
    outer_last = Counting()
    nested = CascadedCrawler(CascadedCrawler(first, second, when=~StatusIn(404)), outer_last, when=~StatusIn(404))
    await nested.crawl("d")   # first 403 -> second 404: the inner cascade stops and returns the 404
    assert outer_last.asked == []


async def test_the_cascade_returns_the_fetch_of_the_crawler_it_stopped_at():
    from factassessor.crawlers import CascadedCrawler, NeedsBrowser

    f = await CascadedCrawler(crawler(responses(SITES), min_words=100), Counting(), when=NeedsBrowser()).crawl("https://spa.org/")
    assert f.crawler == "Counting" and f.page["text"].startswith("rendered")
    f = await CascadedCrawler(crawler(responses(SITES), min_words=100), Counting(), when=NeedsBrowser()).crawl("https://gone.org/")
    assert f.crawler == "HTTPXCrawler" and f.status == 404 and not f


# --- the network is timed and capped; parsing is not --------------------------------------------------------

async def test_parsing_neither_holds_a_connection_slot_nor_counts_against_the_deadline(monkeypatch):
    # under load, pages parsed inside the slot and the deadline made downloads queue behind the CPU (16.7s median
    # wait at 600 URLs) and pages that had downloaded fine time out mid-parse
    import time

    import factassessor.crawlers.plain_http as ph

    real = ph.extract
    held = []

    def slow_extract(body, kind, encoding=None):
        held.append(c._slots._value)  # free slots while parsing
        time.sleep(0.3)  # a big page: longer than the deadline below
        return real(body, kind, encoding)

    monkeypatch.setattr(ph, "extract", slow_extract)
    c = crawler(html(), timeout=0.2, max_concurrent=1)
    f = await c.crawl("https://big.org")
    assert f and f.page["title"] == "Nepal earthquake - Wikipedia"  # read, not a TimeoutError
    assert held == [1]  # the one slot was free again while parsing


# --- ImpitCrawler: the same crawler, over a client that looks like a real browser at the TLS level ---------------

class FakeImpit:
    """impit's client surface as HTTPXCrawler uses it: stream() -> status, headers, aiter_bytes()."""

    def __init__(self, status=200, body=PAGE.encode(), ctype="text/html; charset=utf-8"):
        self.status, self.body, self.ctype = status, body, ctype

    def stream(self, method, url):
        test = self

        class Response:
            status_code, headers = test.status, {"content-type": test.ctype}

            async def aiter_bytes(self):
                yield test.body

            async def __aenter__(self):
                return self

            async def __aexit__(self, *exc):
                return False

        return Response()

    async def aclose(self):
        pass


async def test_impit_crawler_reads_pages_like_the_httpx_one_and_names_itself():
    from factassessor.crawlers import ImpitCrawler

    c = ImpitCrawler(min_words=0)
    c._http = FakeImpit()
    f = await c.crawl("https://journal.org/paper")
    assert f and f.crawler == "ImpitCrawler" and f.page["title"] == "Nepal earthquake - Wikipedia"
    c._http = FakeImpit(status=403, body=b"<html>Access Denied</html>")
    f = await c.crawl("https://journal.org/other")
    assert (f.status, bool(f)) == (403, False)


async def test_impit_crawler_impersonates_firefox_by_default():
    # on 313 pages plain httpx couldn't read: Firefox's fingerprint read 62, Chrome's 23 (Chrome's drew JS checks)
    from factassessor.crawlers import ImpitCrawler

    c = ImpitCrawler()
    assert c.browser == "firefox" and ImpitCrawler(browser="chrome").browser == "chrome"
    await c.start()
    import impit

    assert isinstance(c._http, impit.AsyncClient)
    await c.stop()


def test_the_charset_comes_from_the_content_type_header():
    from factassessor.crawlers.plain_http import charset

    assert charset("text/html; charset=ISO-8859-1") == "iso-8859-1"
    assert charset('text/html;charset="utf-8"') == "utf-8"
    assert charset("text/html") is None and charset("") is None


async def test_at_most_max_per_host_requests_to_one_site_at_once():
    # 50 connections at once could all go to one site, and sites throttle bursts: PMC answered with reCAPTCHA pages
    # mid-run, and pages the old 20-connection cascade had read went missing. Scrapy and Crawlee also cap per domain.
    running, peak = {}, {}

    async def handler(request):
        host = request.url.host
        running[host] = running.get(host, 0) + 1
        peak[host] = max(peak.get(host, 0), running[host])
        await asyncio.sleep(0.02)
        running[host] -= 1
        return httpx.Response(200, content=PAGE.encode(), headers={"content-type": "text/html"})

    c = crawler(handler, max_concurrent=50, max_per_host=3)
    urls = [f"https://pmc.org/{i}" for i in range(12)] + [f"https://other{i}.org/" for i in range(6)]
    await asyncio.gather(*(c.crawl(u) for u in urls))
    assert peak["pmc.org"] == 3 and all(peak[f"other{i}.org"] == 1 for i in range(6))
    assert HTTPXCrawler().max_per_host == 6
