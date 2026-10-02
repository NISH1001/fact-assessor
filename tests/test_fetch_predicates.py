from factassessor import Filter, collect
from factassessor.crawlers import Fetch, HasPage, StatusIn

PAGE = {"url": "https://a.org", "title": "A", "text": "word " * 300}

article = Fetch(url="https://a.org", page=PAGE, status=200, content_type="text/html", words=300)
js_shell = Fetch(url="https://b.org", status=200, content_type="text/html", words=12)
bad_pdf = Fetch(url="https://c.org/x.pdf", status=200, content_type="application/pdf")
refused = Fetch(url="https://d.org", status=403, content_type="text/html", words=40)
gone = Fetch(url="https://e.org", status=404)
timed_out = Fetch(url="https://f.org", error="TimeoutError")


def test_status_in_takes_codes_and_ranges():
    assert StatusIn(404, 410)(gone) and not StatusIn(404, 410)(refused)
    success = StatusIn(range(200, 300))
    assert [success(f) for f in (article, js_shell, bad_pdf, refused, gone, timed_out)] == [True, True, True, False, False, False]
    assert not StatusIn(200)(timed_out)  # no response: no status, never in any set


def test_has_page_means_a_2xx_with_text_not_just_a_2xx():
    assert [HasPage()(f) for f in (article, js_shell, bad_pdf, refused, gone, timed_out)] == [True, False, False, False, False, False]
    assert not HasPage()(Fetch(url="https://g.org", page=PAGE, status=403))  # a page under an error status doesn't count


def test_they_combine_and_negate_like_any_predicate():
    js_like = StatusIn(range(200, 300)) & ~HasPage()  # the server said yes, nothing came out
    assert [js_like(f) for f in (article, js_shell, bad_pdf, refused)] == [False, True, True, False]
    assert (HasPage() | StatusIn(404))(gone) and not (HasPage() | StatusIn(404))(refused)


async def test_in_a_chain_a_predicate_filters_fetches():
    async def stream():
        for f in (article, js_shell, refused, gone):
            yield f

    kept = await collect(Filter(HasPage() | StatusIn(403))(stream()))
    assert [f.url for f in kept] == ["https://a.org", "https://d.org"]
