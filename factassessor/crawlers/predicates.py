"""What a crawl produced, and the conditions that decide what to do with it.

`Fetch` records one crawl of one URL: the page when text came out, and the facts about the response either way.
The predicates read a `Fetch`; in a chain each becomes a `Filter` (`>>`), and they combine with `&`, `|`, `~`.
"""

from __future__ import annotations

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
