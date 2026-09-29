"""Crawlers: url -> page `{"url", "title", "text"}` (clean plain text). A step; failed pages are dropped.

`Crawler` is the role: implement `crawl(url)` (return None on failure).
- `Crawl4AICrawler`: a headless browser; renders JavaScript; ~0.6-1.6s per page.
- `HTTPXCrawler`: a plain HTTP fetch + HTML-to-text; no JavaScript; much faster for ordinary pages.
- `FallbackCrawler(fast, thorough)`: the first crawler that gets text wins.
- `OpenAccessCrawler`: for a URL with a DOI, the paper's free full text (via OpenAlex); for publishers that block
  crawlers, as the last crawler in a `FallbackCrawler`.
"""

from factassessor.crawlers._base import Crawler, FallbackCrawler, NoCrawler
from factassessor.crawlers.browser import Crawl4AICrawler
from factassessor.crawlers.open_access import OpenAccessCrawler, arxiv_pdf_url, doi_in, looks_like_pdf
from factassessor.crawlers.plain_http import HTTPXCrawler

__all__ = [
    "Crawler", "Crawl4AICrawler", "HTTPXCrawler", "OpenAccessCrawler", "FallbackCrawler", "NoCrawler",
    "doi_in", "arxiv_pdf_url", "looks_like_pdf",
]
