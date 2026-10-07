"""Cut crawled pages into model-sized passages and keep the ones relevant to a claim.

Judge-agnostic: pass the judge model's own tokenizer and token budget.
"""

from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter
from typing import Any

_WORD = re.compile(r"\w+")
_CITATION = re.compile(r"\[(?:\d+|[a-z]{1,2})\]")  # [6], [c]
_EMPHASIS = re.compile(r"\*\*|__|(?<!\w)_(?=\S)|(?<=\S)_(?!\w)")  # **bold**, __bold__, _italic_
_TABLE_RULE = re.compile(r"^\|?[\s:|-]+\|?$")  # |---|:---:|
_BULLET = re.compile(r"^[*+-]\s+")
_TAG = re.compile(r"</?[a-zA-Z][^>]*>")
_JSON_LINE = re.compile(r'^\s*[\[{].*["\]}]\s*$')  # a line that's a JSON object/array, e.g. JSON-LD metadata
_ALNUM = re.compile(r"[^\W_]")  # a letter or digit (str.isalnum), found in C instead of a Python loop
_SPACE_PUNCT = re.compile(r" ([,.;:!?])")


_MINUS = str.maketrans({"\u2212": "-", "\u2010": "-", "\u2011": "-"})  # minus sign, hyphen, non-breaking hyphen


def normalize_text(text: str) -> str:
    """One way to write the same characters (Unicode NFKC, minus signs as "-"), for comparing a claim with evidence:
    "247 Mg ha⁻¹" and a paper's "247 Mg ha−1" both become "247 Mg ha-1". Laya judged exactly that pair a
    refutation (0.84): different tokens looked like a different number. Also ligatures (ﬁ -> fi), subscripts
    (CO₂ -> CO2). Superscripts lose their raise (10⁵ -> 105), as they already do in text extracted from PDFs."""
    return unicodedata.normalize("NFKC", text).translate(_MINUS)


def clean_text(markdown: str) -> str:
    """Crawled markdown -> plain text: no formatting, citations, table pipes, or menu/share-button lines.

    Every rule only removes characters, so a line without a letter or digit is dropped before any of them run, and
    each rule runs only on lines with its marker: the same output as running all of them, 5x faster on 258 pages."""
    lines = []
    for line in markdown.splitlines():
        if not _ALNUM.search(line):
            continue
        if "<" in line:
            low = line.lower()
            if "<script" in low or "<style" in low:
                continue  # a leaked <script>/<style> block (e.g. JSON-LD): markup, not content
            line = _TAG.sub("", line)
        line = line.strip()
        if line[:1] in ("[", "{") and _JSON_LINE.match(line) and line.count('"') >= 4:
            continue
        if "|" in line:
            if _TABLE_RULE.match(line):
                continue
            if line.count("|") >= 2:  # table row, wherever it starts
                line = " ".join(cell.strip() for cell in line.split("|") if cell.strip())
        if line[:1] in ("*", "+", "-") and _BULLET.match(line):
            line = _BULLET.sub("", line)
            if len(line.split()) <= 2:  # "Facebook", "Next page": navigation, not content
                continue
        line = line.lstrip("#").strip()
        if "[" in line:
            line = _CITATION.sub("", line)
        if "*" in line or "_" in line:
            line = _EMPHASIS.sub("", line)
        line = _SPACE_PUNCT.sub(r"\1", " ".join(line.split()))
        if any(ch.isalnum() for ch in line):
            lines.append(line)
    return "\n".join(lines)


def word_windows(text: str, size: int, overlap: int, max_chars: int | None = None) -> list[str]:
    """Overlapping windows of `size` words, as exact slices of `text` (for judges without a tokenizer budget).

    `max_chars`: a window longer than this is cut into pieces of at most `max_chars` characters. Words are counted
    by whitespace, so text with almost none (a minified file, a data dump) would otherwise make one huge window."""
    spans = [m.span() for m in re.finditer(r"\S+", text)]
    step = max(1, size - overlap)
    windows = []
    for start in range(0, len(spans), step):
        part = spans[start : start + size]
        windows.append(text[part[0][0] : part[-1][1]])
        if start + size >= len(spans):
            break
    if max_chars is None:
        return windows
    return [w[i : i + max_chars] for w in windows for i in range(0, len(w), max_chars)]


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
    return [chunks[i] for i in BM25Index(chunks, k1, b).top(claim, k)]


class BM25Index:
    """BM25 over a fixed list of passages, built once and queried many times (e.g. every claim against a paper).
    `top(query, k)`: indices of the k best passages, best first (ties keep passage order)."""

    def __init__(self, chunks: list[str], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1, self.b = k1, b
        self.docs = [Counter(_words(c)) for c in chunks]
        self.lengths = [sum(d.values()) for d in self.docs]
        self.avg_len = sum(self.lengths) / len(self.docs) if self.docs else 0.0
        self.postings: dict[str, list[int]] = {}  # word -> passages containing it: only those get scored
        for i, doc in enumerate(self.docs):
            for w in doc:
                self.postings.setdefault(w, []).append(i)

    def scores(self, query: str) -> dict[int, float]:
        n, total = len(self.docs), {}
        for w in set(_words(query)):
            ids = self.postings.get(w, [])
            idf = math.log(1 + (n - len(ids) + 0.5) / (len(ids) + 0.5))
            for i in ids:
                tf = self.docs[i][w]
                norm = tf + self.k1 * (1 - self.b + self.b * self.lengths[i] / self.avg_len)
                total[i] = total.get(i, 0.0) + idf * tf * (self.k1 + 1) / norm
        return total

    def top(self, query: str, k: int, matching_only: bool = False) -> list[int]:
        """`matching_only`: leave out passages sharing no word with the query (instead of ranking them last)."""
        scores = self.scores(query)
        pool = scores if matching_only else range(len(self.docs))
        return sorted(pool, key=lambda i: scores.get(i, 0.0), reverse=True)[:k]


def _words(text: str) -> list[str]:
    return _WORD.findall(text.lower())
