"""Cut crawled pages into model-sized passages and keep the ones relevant to a claim.

Judge-agnostic: pass the judge model's own tokenizer and token budget.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any

_WORD = re.compile(r"\w+")
_CITATION = re.compile(r"\[(?:\d+|[a-z]{1,2})\]")  # [6], [c]
_EMPHASIS = re.compile(r"\*\*|__|(?<!\w)_(?=\S)|(?<=\S)_(?!\w)")  # **bold**, __bold__, _italic_
_TABLE_RULE = re.compile(r"^\|?[\s:|-]+\|?$")  # |---|:---:|
_BULLET = re.compile(r"^[*+-]\s+")
_TAG = re.compile(r"</?[a-zA-Z][^>]*>")
_JSON_LINE = re.compile(r'^\s*[\[{].*["\]}]\s*$')  # a line that's a JSON object/array, e.g. JSON-LD metadata


def clean_text(markdown: str) -> str:
    """Crawled markdown -> plain text: no formatting, citations, table pipes, or menu/share-button lines."""
    lines = []
    for line in markdown.splitlines():
        if "<script" in line.lower() or "<style" in line.lower():
            continue  # a leaked <script>/<style> block (e.g. JSON-LD): markup, not content
        line = _TAG.sub("", line).strip()
        if _JSON_LINE.match(line) and line.count('"') >= 4:
            continue
        if _TABLE_RULE.match(line):
            continue
        if line.count("|") >= 2:  # table row, wherever it starts
            line = " ".join(cell.strip() for cell in line.split("|") if cell.strip())
        if _BULLET.match(line):
            line = _BULLET.sub("", line)
            if len(line.split()) <= 2:  # "Facebook", "Next page": navigation, not content
                continue
        line = line.lstrip("#").strip()
        line = _EMPHASIS.sub("", _CITATION.sub("", line))
        line = re.sub(r"\s+([,.;:!?])", r"\1", re.sub(r"\s+", " ", line)).strip()
        if any(ch.isalnum() for ch in line):
            lines.append(line)
    return "\n".join(lines)


def chunk(text: str, tokenizer: Any, max_tokens: int, overlap: int) -> list[str]:
    """Overlapping windows of at most `max_tokens` real tokens, as exact slices of `text`.

    The page is tokenized once; offsets map each window back to the original characters.
    """
    offsets = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)["offset_mapping"]
    if not offsets:
        return []
    step = max(1, max_tokens - overlap)
    chunks = []
    for start in range(0, len(offsets), step):
        window = offsets[start : start + max_tokens]
        chunks.append(text[window[0][0] : window[-1][1]])
        if start + max_tokens >= len(offsets):
            break
    return chunks


def top_passages(claim: str, chunks: list[str], k: int, k1: float = 1.5, b: float = 0.75) -> list[str]:
    """The `k` chunks with the highest BM25 score against `claim`, best first."""
    if len(chunks) <= k:
        return chunks
    docs = [Counter(_words(c)) for c in chunks]
    avg_len = sum(sum(d.values()) for d in docs) / len(docs)
    n_containing = Counter(w for d in docs for w in d)

    def score(doc: Counter[str]) -> float:
        length = sum(doc.values())
        total = 0.0
        for w in set(_words(claim)):
            if tf := doc.get(w):
                idf = math.log(1 + (len(docs) - n_containing[w] + 0.5) / (n_containing[w] + 0.5))
                total += idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * length / avg_len))
        return total

    ranked = sorted(range(len(chunks)), key=lambda i: score(docs[i]), reverse=True)
    return [chunks[i] for i in ranked[:k]]


def _words(text: str) -> list[str]:
    return _WORD.findall(text.lower())
