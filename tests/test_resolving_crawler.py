from factassessor import CascadedCrawler, Crawler, Fetch, ResolvingCrawler

LONG = " ".join(["word"] * 400)


class Pages(Crawler):
    """A crawler over a fixed {url: text}; records what it was asked for."""

    def __init__(self, texts):
        super().__init__()
        self.texts, self.asked = texts, []

    async def crawl(self, url):
        self.asked.append(url)
        text = self.texts.get(url)
        page = {"url": url, "title": "t", "text": text} if text else None
        return self.mark(Fetch(url=url, page=page, status=200 if text else 403, words=len(text.split()) if text else 0))


class Copies:
    def __init__(self, found=None, fail=False):
        self.found, self.fail, self.asked = found or {}, fail, []

    async def resolve(self, url):
        self.asked.append(url)
        if self.fail:
            raise RuntimeError("resolver down")
        return self.found.get(url, [])


async def test_reads_the_first_readable_copy_and_cites_it_under_the_hit_url():
    hit, copy = "https://doi.org/10.1000/x", "https://pmc.example/x"
    web = Pages({copy: LONG})
    f = await ResolvingCrawler(Copies({hit: [copy]}), web).crawl(hit)
    assert f and f.url == hit and f.page["url"] == hit and f.page["text"] == LONG
    assert web.asked == [copy]


async def test_a_copy_shorter_than_min_copy_words_is_skipped_for_the_hit_page():
    hit, copy = "https://doi.org/10.1000/x", "https://pmc.example/x"
    web = Pages({copy: "Please verify you are human", hit: "the publisher page itself"})
    f = await ResolvingCrawler(Copies({hit: [copy]}), web, min_copy_words=300).crawl(hit)
    assert f and f.page["text"] == "the publisher page itself"
    assert web.asked == [copy, hit]


async def test_a_failing_resolver_still_reads_the_hit_itself():
    hit = "https://example.org/page"
    f = await ResolvingCrawler(Copies(fail=True), Pages({hit: "text"})).crawl(hit)
    assert f and f.page["text"] == "text"


async def test_nothing_readable_is_a_fetch_without_a_page_for_the_hit():
    hit = "https://doi.org/10.1000/x"
    f = await ResolvingCrawler(Copies({hit: ["https://pmc.example/x"]}), Pages({})).crawl(hit)
    assert not f and f.url == hit and f.page is None


async def test_after_a_first_crawler_in_a_cascade_it_only_resolves_what_the_first_could_not_read():
    first = Pages({"https://doi.org/10.1000/a": LONG})
    copies = Copies({"https://doi.org/10.1000/b": ["https://pmc.example/b"]})
    cascade = CascadedCrawler(first, ResolvingCrawler(copies, Pages({"https://pmc.example/b": LONG})))
    assert await cascade.crawl("https://doi.org/10.1000/a")
    assert await cascade.crawl("https://doi.org/10.1000/b")
    assert copies.asked == ["https://doi.org/10.1000/b"]  # the hit the first crawler read was never resolved
