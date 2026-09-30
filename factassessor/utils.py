"""Small text helpers shared across components: sentence spans, and where in a text a piece of it came from."""

from __future__ import annotations

import re

from factassessor.passages import BM25Index

_SENTENCE = re.compile(r"\S.*?(?:[.!?]+(?=\s|$)|$)", re.S)  # ends at .!? + space, so "7.8" stays whole


def sentences(text: str) -> list[tuple[int, int]]:
    """Character spans of the sentences of `text`, in order."""
    return [(m.start(), m.start() + len(m.group().rstrip())) for m in _SENTENCE.finditer(text)]


def locate(query: str, source: str) -> tuple[int, int]:
    """The character span of the sentence in `source` that best matches `query`: the one sharing the most, and
    rarest, words with it (BM25 over the sentences, so wording every sentence repeats counts for little next to
    the query's own detail). The first sentence when nothing matches. The atomizer uses it to find the sentence a
    claim was made from; it works the same for a quote in a page or a title in a document."""
    spans = sentences(source) or [(0, len(source))]
    [best] = BM25Index([source[s:e] for s, e in spans]).top(query, 1)
    return spans[best]
