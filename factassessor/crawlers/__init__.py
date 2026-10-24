"""Crawlers: url -> `Fetch` (the page `{"url", "title", "text"}`, clean plain text, and the facts about the
response). A step: usable pages come out, the rest are dropped.

`Crawler` is the role: implement `crawl(url) -> Fetch` (never raise; end with `self.mark(f)`, which applies `accept`).
- `Crawl4AICrawler`: a headless browser; renders JavaScript; ~0.6-1.6s per page.
- `HTTPXCrawler`: a plain HTTP fetch + `extract()` (HTML or PDF); no JavaScript; much faster for ordinary pages.
  (Papers in full from their free copies: give the pipeline a resolver, `FactAssessor(resolver=...)`.)
- `ImpitCrawler`: the same over impit, which looks like a real browser at the TLS level: gets past many 403s.
- `AKDDoiCrawler`: a paper's full text from AKD Scholar Search by the DOI in its url; no web fetch.
- `ResolvingCrawler(resolver, crawler)`: free copies first (arXiv, OpenAlex, PMC), then the url, read with `crawler`.
- `CascadedCrawler(fast, thorough, when=...)`: the first usable page wins; `when` says which failures go on.
"""

from factassessor.crawlers._base import Crawler, CascadedCrawler, NoCrawler
from factassessor.crawlers.akd import AKDDoiCrawler
from factassessor.crawlers.browser import Crawl4AICrawler
from factassessor.crawlers.plain_http import HTTPXCrawler, ImpitCrawler
from factassessor.crawlers.resolving import ResolvingCrawler
from factassessor.crawlers.predicates import (
    BodyMatches, BotChallenge, ContentType, Fetch, HasPage, JavaScriptShell, MinWords, NeedsBrowser, Paywalled, StatusIn,
)
from factassessor.resolvers import doi_in

__all__ = [
    "Crawler", "Crawl4AICrawler", "HTTPXCrawler", "ImpitCrawler", "CascadedCrawler", "NoCrawler", "AKDDoiCrawler", "ResolvingCrawler", "doi_in",
    "Fetch", "StatusIn", "HasPage", "MinWords", "ContentType", "BodyMatches", "BotChallenge", "JavaScriptShell", "NeedsBrowser", "Paywalled",
]
