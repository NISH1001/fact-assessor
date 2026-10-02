"""What a crawl produced, and the conditions that decide what to do with it.

`Fetch` records one crawl of one URL: the page when text came out, and the facts about the response either way.
The predicates read a `Fetch`; in a chain each becomes a `Filter` (`>>`), and they combine with `&`, `|`, `~`.
"""

from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel

from factassessor.pipeline import Predicate


class Fetch(BaseModel):
    """One crawl of one URL: the page, or the facts about why there isn't one."""

    url: str
    page: dict[str, Any] | None = None  # {"url", "title", "text"}: the extracted text, when any came out
    status: int | None = None  # the HTTP status; None when no response came back (timeout, DNS, refused connection)
    content_type: str = ""  # "text/html", "application/pdf", ... without parameters
    words: int = 0  # words of extracted text (0 when nothing came out)
    body_head: str = ""  # the start of the response, decoded, for markers (bot checks)
    error: str = ""  # the exception's name when there was no response


class StatusIn(Predicate):
    """The server answered with one of these codes: `StatusIn(404, 410)`, `StatusIn(range(200, 300))`.
    False when there was no response at all."""

    def __init__(self, *codes: int | range) -> None:
        allowed = frozenset(c for code in codes for c in (code if isinstance(code, range) else (code,)))
        super().__init__(lambda f: f.status in allowed)


class HasPage(Predicate):
    """The crawl produced a page: a 2xx response whose text could be extracted. A 2xx with nothing readable
    (a JavaScript shell, a PDF that didn't parse) is not a page."""

    def __init__(self) -> None:
        success = StatusIn(range(200, 300))
        super().__init__(lambda f: f.page is not None and success(f))


class MinWords(Predicate):
    """At least `n` words of text came out."""

    def __init__(self, n: int) -> None:
        super().__init__(lambda f: f.words >= n)


class ContentType(Predicate):
    """The response's content type contains `kind`: `ContentType("html")`, `ContentType("pdf")`."""

    def __init__(self, kind: str) -> None:
        kind = kind.lower()
        super().__init__(lambda f: kind in f.content_type)


# what bot-check pages say (Cloudflare's "Just a moment...", captchas); read from the start of the response body
_CHALLENGE = re.compile(
    r"just a moment|cf-chl|cf_chl|challenge-platform|captcha|are you a robot|verify you are human"
    r"|checking your browser|enable javascript and cookies",
    re.IGNORECASE,
)


class BotChallenge(Predicate):
    """The server refused (403, 503 by default) with a bot-check page. A real browser sometimes passes: 104 of 453
    such pages in a 1,500-URL sample of the eval's search hits."""

    def __init__(self, statuses: tuple[int, ...] = (403, 503)) -> None:
        super().__init__(lambda f: f.status in statuses and bool(_CHALLENGE.search(f.body_head)))


class JavaScriptShell(Predicate):
    """A 2xx HTML page with almost no text: the content is probably built by JavaScript, which a browser runs. A
    heuristic (it reads the word count, not the scripts): the browser rescued 60 of 119 in the sample, the rest were
    pages that are short anyway (redirect stubs, landing pages)."""

    def __init__(self, max_words: int = 100) -> None:
        p = StatusIn(range(200, 300)) & ContentType("html") & ~MinWords(max_words)
        super().__init__(p.fn)




# what paywall pages say; read from the start of the response body
_PAYWALL = re.compile(
    r"subscribe to (read|continue)|purchase (this )?article|buy (this )?article|rent (this )?article"
    r"|institutional access|log ?in to (read|access|view)",
    re.IGNORECASE,
)


class Paywalled(Predicate):
    """The content needs a login or payment: a 401 or 402, or paywall wording at the start of the page. No crawler
    gets past it; the resolvers look for a free copy of the paper instead."""

    def __init__(self) -> None:
        super().__init__(lambda f: f.status in (401, 402) or bool(_PAYWALL.search(f.body_head)))


class NeedsBrowser(Predicate):
    """An HTTP failure a browser can plausibly fix: a JavaScript shell (unless it is a short paywall page, which a
    browser renders the same), a bot check, or a 405 sent to non-browsers. In a 1,500-URL sample of the eval's hits
    these held 170 of the browser's 180 rescues, and the failures left out (404, 401/402, timeouts, PDFs that didn't
    parse, plain 403s) a third of its attempts."""

    def __init__(self) -> None:
        p = (JavaScriptShell() & ~Paywalled()) | BotChallenge() | StatusIn(405)
        super().__init__(p.fn)
