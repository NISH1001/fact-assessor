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


# --- the predicates behind NeedsBrowser: the cases from the 1,500-URL measurement ---------------------------------

from factassessor.crawlers import BotChallenge, ContentType, JavaScriptShell, MinWords, NeedsBrowser

loading = Fetch(url="https://spa.org", page={"url": "https://spa.org", "title": "", "text": "Loading..."}, status=200,
                content_type="text/html", words=1, body_head="<div id='root'></div><p>Loading...</p>")
challenge = Fetch(url="https://pub.org", status=403, content_type="text/html",
                  body_head="<title>Just a moment...</title><body>Checking your browser before accessing. cf-chl</body>")
plain_403 = Fetch(url="https://pub2.org", status=403, content_type="text/html", body_head="<h1>403 Forbidden</h1>")
method_405 = Fetch(url="https://odd.org", status=405, content_type="text/html")


def test_min_words_and_content_type():
    assert MinWords(100)(article) and not MinWords(100)(loading) and not MinWords(1)(timed_out)
    assert ContentType("html")(article) and not ContentType("html")(bad_pdf) and ContentType("pdf")(bad_pdf)
    assert not ContentType("html")(timed_out)  # no response: no type


def test_bot_challenge_is_a_refusal_with_a_challenge_page():
    assert BotChallenge()(challenge)
    assert not BotChallenge()(plain_403)  # refused, but nothing a browser could pass
    assert not BotChallenge()(Fetch(url="x", status=200, body_head="please solve the captcha below"))  # not a refusal


def test_javascript_shell_is_a_2xx_html_page_with_almost_no_text():
    assert JavaScriptShell()(loading) and JavaScriptShell()(js_shell)
    assert not JavaScriptShell()(article)  # real text
    assert not JavaScriptShell()(bad_pdf)  # a 2xx with no text, but a PDF: a browser can't parse it better
    assert not JavaScriptShell()(refused) and not JavaScriptShell()(timed_out)
    assert JavaScriptShell(max_words=500)(article)  # the threshold is a setting


def test_needs_browser_is_exactly_the_cases_a_browser_rescued():
    yes = [loading, js_shell, challenge, method_405]
    no = [article, bad_pdf, plain_403, gone, timed_out, Fetch(url="x", status=401), Fetch(url="x", status=410),
          Fetch(url="x", status=500)]
    assert all(NeedsBrowser()(f) for f in yes), [f.url for f in yes if not NeedsBrowser()(f)]
    assert not any(NeedsBrowser()(f) for f in no), [f.url for f in no if NeedsBrowser()(f)]


def test_paywalled_is_a_401_or_402_or_paywall_wording():
    from factassessor.crawlers import Paywalled

    teaser = Fetch(url="https://journal.org/a", status=200, content_type="text/html", words=40,
                   page={"url": "https://journal.org/a", "title": "", "text": "Abstract ..."},
                   body_head="<p>Abstract ...</p><div>Subscribe to read the full article. Purchase this article</div>")
    assert Paywalled()(teaser) and Paywalled()(Fetch(url="x", status=402)) and Paywalled()(Fetch(url="x", status=401))
    assert not Paywalled()(article) and not Paywalled()(loading) and not Paywalled()(challenge)
    assert not NeedsBrowser()(teaser)  # short, html, 200: shaped like a JavaScript shell, but a browser renders the same paywall
    assert JavaScriptShell()(teaser)   # the shell rule alone would have sent it
