"""Crawlers: url -> page `{"url", "title", "text"}` (clean plain text). A step; failed pages are dropped.

`Crawler` is the role: implement `crawl(url)` (return None on failure).
- `Crawl4AICrawler`: a headless browser; renders JavaScript; ~0.6-1.6s per page.
- `HTTPXCrawler`: a plain HTTP fetch + `extract()` (HTML or PDF); no JavaScript; much faster for ordinary pages.
  (Papers in full from their free copies: give the pipeline a resolver, `FactAssessor(resolver=...)`.)
- `CascadedCrawler(fast, thorough)`: the first crawler that gets text wins.
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
