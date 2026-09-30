"""Small text helpers shared across components: sentence spans, and where in a text a claim was made."""

from __future__ import annotations

import re

from factassessor.passages import BM25Index

_SENTENCE = re.compile(r"\S.*?(?:[.!?]+(?=\s|$)|$)", re.S)  # ends at .!? + space, so "7.8" stays whole


def sentences(text: str) -> list[tuple[int, int]]:
    """Character spans of the sentences of `text`, in order."""
    return [(m.start(), m.start() + len(m.group().rstrip())) for m in _SENTENCE.finditer(text)]


def locate(claim: str, text: str) -> tuple[int, int]:
    """The span of the sentence of `text` that `claim` was made from: the one sharing the most, and rarest, words
    with it (BM25 over the sentences, so context the atomizer repeats in every claim, like a study's name, counts
    for little next to the claim's own detail). The first sentence when nothing matches."""
    spans = sentences(text) or [(0, len(text))]
    [best] = BM25Index([text[s:e] for s, e in spans]).top(claim, 1)
    return spans[best]
