from factassessor.utils import locate, sentences

TEXT = "It was believed that Nepal's earthquake in 2017 of 7.8 magnitude scale caused massive damage. Total lives lost were 1 million people."


def test_sentences_end_at_punctuation_and_keep_decimals_whole():
    assert [TEXT[s:e] for s, e in sentences(TEXT)] == [
        "It was believed that Nepal's earthquake in 2017 of 7.8 magnitude scale caused massive damage.",
        "Total lives lost were 1 million people.",
    ]
    assert sentences("") == []


def test_locate_picks_the_sentence_the_claim_was_made_from():
    span = lambda claim: TEXT[slice(*locate(claim, TEXT))]
    assert span("The Nepal earthquake had a magnitude of 7.8.").startswith("It was believed")
    assert span("The Nepal earthquake killed 1 million people.") == "Total lives lost were 1 million people."
    assert span("Something about nothing here.").startswith("It was believed")  # no match: the first sentence


def test_locate_weighs_down_context_the_atomizer_repeats_in_every_claim():
    # every claim of a scientific text names the study; the claim's own detail must decide the sentence
    text = ("A 2025 study in Connecticut modelled aboveground biomass with random forests. "
            "The 2025 Connecticut study used 67 explanatory variables from LiDAR and Sentinel-2. "
            "The 2025 Connecticut study validated its models on 142 FIA subplots.")
    claim = "The 2025 study in Connecticut used 67 explanatory variables."
    assert text[slice(*locate(claim, text))].startswith("The 2025 Connecticut study used 67")


def test_locate_on_text_without_sentence_punctuation_is_the_whole_text():
    assert locate("anything", "no punctuation at all") == (0, len("no punctuation at all"))


# --- cache: the async method cache -----------------------------------------------------------------------------

import asyncio

import pytest

from factassessor.utils import cache


class Fetcher:
    def __init__(self, delay=0.05, fail=()):
        self.calls, self.delay, self.fail = [], delay, set(fail)

    @cache(maxsize=2, ttl=60)
    async def get(self, url):
        self.calls.append(url)
        await asyncio.sleep(self.delay)
        if url in self.fail:
            raise RuntimeError("boom")
        return f"page of {url}"


async def test_concurrent_callers_share_one_call_and_later_ones_hit_the_cache():
    f = Fetcher()
    pages = await asyncio.gather(*(f.get("u") for _ in range(5)))  # claims asking for the same page at once
    assert pages == ["page of u"] * 5 and f.calls == ["u"]
    assert await f.get("u") == "page of u" and f.calls == ["u"]  # finished: served from the cache
    assert await f.get("v") == "page of v" and f.calls == ["u", "v"]


async def test_instances_have_their_own_cache():
    a, b = Fetcher(), Fetcher()
    await a.get("u"); await b.get("u")
    assert a.calls == ["u"] and b.calls == ["u"]


async def test_a_cancelled_caller_does_not_cancel_the_shared_call():
    f = Fetcher(delay=0.1)
    doomed = asyncio.create_task(f.get("u"))
    alive = asyncio.create_task(f.get("u"))
    await asyncio.sleep(0.02)
    doomed.cancel()  # a claim hitting its deadline mid-crawl
    assert await alive == "page of u" and f.calls == ["u"]
    with pytest.raises(asyncio.CancelledError):
        await doomed


async def test_failures_are_not_kept_and_the_size_is_bounded():
    f = Fetcher(fail={"bad"})
    with pytest.raises(RuntimeError):
        await f.get("bad")
    with pytest.raises(RuntimeError):
        await f.get("bad")
    assert f.calls == ["bad", "bad"]  # retried, not remembered
    for url in ("a", "b", "c"):  # maxsize 2: the least recently used is dropped
        await f.get(url)
    await f.get("a")
    assert f.calls.count("a") == 2


async def test_crawlers_and_the_openalex_resolver_are_cached():
    from factassessor import Crawl4AICrawler, HTTPXCrawler, OpenAlexResolver

    for cls, name in ((HTTPXCrawler, "fetch"), (Crawl4AICrawler, "crawl"), (OpenAlexResolver, "resolve")):
        assert hasattr(getattr(cls, name), "__cached__"), cls.__name__


def test_sync_functions_and_methods_are_cached_too():
    calls = []

    @cache(maxsize=8, ttl=60)
    def square(x):
        calls.append(x)
        return x * x

    assert [square(3), square(3), square(4)] == [9, 9, 16] and calls == [3, 4]

    class Parser:
        def __init__(self):
            self.n = 0

        @cache(maxsize=8, ttl=60)
        def parse(self, s):
            self.n += 1
            return s.upper()

    p = Parser()
    assert p.parse("a") == p.parse("a") == "A" and p.n == 1


async def test_the_class_form_caches_the_named_methods():
    @cache(["get"], maxsize=8, ttl=60)
    class Plain:
        def __init__(self):
            self.calls = []

        async def get(self, url):
            self.calls.append(url)
            return url

        async def other(self, url):
            self.calls.append(url)
            return url

    p = Plain()
    await p.get("u"); await p.get("u"); await p.other("u"); await p.other("u")
    assert p.calls == ["u", "u", "u"]  # get cached, other not


def test_misuse_fails_when_the_decorator_is_applied():
    with pytest.raises(TypeError, match="class form"):
        @cache(["crawl"])
        async def crawl(url): ...
    with pytest.raises(TypeError, match="name the methods"):
        @cache(maxsize=8)
        class A: ...
    with pytest.raises(AttributeError, match="no method 'crawll'"):
        @cache(["crawll"])
        class B:
            async def crawl(self, url): ...
    with pytest.raises(ValueError):
        cache(maxsize=0)
    with pytest.raises(TypeError, match="non-empty list"):
        cache("crawl")


def test_cached_async_methods_still_look_async():
    # the pipeline's Map awaits a function only if inspect says it is a coroutine function
    import inspect

    @cache(maxsize=4)
    async def f(x):
        return x

    assert inspect.iscoroutinefunction(f) and inspect.iscoroutinefunction(Fetcher().get)
