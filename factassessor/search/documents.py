"""DocumentSearcher: search given documents instead of the web (in-domain checks)."""

from __future__ import annotations

from typing import Any

from factassessor.passages import BM25Index, word_windows
from factassessor.search._base import Searcher


class DocumentSearcher(Searcher):
    """Search given documents instead of the web: in-domain checks against the papers or reports a text was
    written from. Documents `{"url", "title", "text"}` are split once
    into overlapping `passage_words`-word passages; `search(claim)` returns the `num` best (BM25) as ordinary hits
    `{"url": "<doc url>#p<n>", "title", "snippet"}`, so any judge takes them. Nothing to crawl: pair it with
    `NoCrawler()`: `FactAssessor(searcher=DocumentSearcher(docs), crawler=NoCrawler())`.
    """

    def __init__(self, documents: list[dict[str, Any]], passage_words: int = 120, overlap: int = 30, num: int = 5) -> None:
        self.num = num
        self.passages: list[tuple[dict[str, Any], int, str]] = []
        for doc in documents:
            for n, text in enumerate(word_windows(doc["text"], passage_words, overlap)):
                self.passages.append((doc, n, text))
        self.index = BM25Index([text for _, _, text in self.passages])

    async def search(self, query: str) -> list[dict[str, Any]]:
        best = self.index.top(query, self.num, matching_only=True)
        return [
            {"url": f"{doc['url']}#p{n}", "title": doc.get("title") or "", "snippet": text}
            for doc, n, text in (self.passages[i] for i in best)
        ]
