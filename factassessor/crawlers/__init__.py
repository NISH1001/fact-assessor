"""Crawlers: url -> `Fetch` (the page `{"url", "title", "text"}`, clean plain text, and the facts about the
response). A step: usable pages come out, the rest are dropped.

`Crawler` is the role: implement `crawl(url) -> Fetch` (never raise; end with `self.mark(f)`, which applies `accept`).
- `Crawl4AICrawler`: a headless browser; renders JavaScript; ~0.6-1.6s per page.
- `HTTPXCrawler`: a plain HTTP fetch + `extract()` (HTML or PDF); no JavaScript; much faster for ordinary pages.
  (Papers in full from their free copies: give the pipeline a resolver, `FactAssessor(resolver=...)`.)
- `CascadedCrawler(fast, thorough, when=...)`: the first usable page wins; `when` says which failures go on.
"""

from factassessor.crawlers._base import Crawler, CascadedCrawler, NoCrawler
from factassessor.crawlers.browser import Crawl4AICrawler
from factassessor.crawlers.plain_http import HTTPXCrawler
from factassessor.crawlers.predicates import (
    BodyMatches, BotChallenge, ContentType, Fetch, HasPage, JavaScriptShell, MinWords, NeedsBrowser, Paywalled, StatusIn,
)
from factassessor.resolvers import doi_in

__all__ = [
    "Crawler", "Crawl4AICrawler", "HTTPXCrawler", "CascadedCrawler", "NoCrawler", "doi_in",
    "Fetch", "StatusIn", "HasPage", "MinWords", "ContentType", "BodyMatches", "BotChallenge", "JavaScriptShell", "NeedsBrowser", "Paywalled",
]
