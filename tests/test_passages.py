import re

from factassessor.passages import chunk, top_passages


class WordTokenizer:
    """One token per word, with char offsets, like a HF fast tokenizer."""

    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
        spans = [m.span() for m in re.finditer(r"\S+", text)]
        out = {"input_ids": list(range(len(spans)))}
        if return_offsets_mapping:
            out["offset_mapping"] = spans
        return out


TOK = WordTokenizer()


def n_tokens(text):
    return len(TOK(text)["input_ids"])


def test_chunks_respect_the_token_budget_and_overlap():
    text = " ".join(f"w{i}" for i in range(25))
    chunks = chunk(text, TOK, max_tokens=10, overlap=3)
    assert all(n_tokens(c) <= 10 for c in chunks)
    assert chunks[0].split()[-3:] == chunks[1].split()[:3]  # 3 tokens of overlap
    assert chunks[0].startswith("w0") and chunks[-1].endswith("w24")  # nothing lost


def test_chunks_are_exact_slices_of_the_original_text():
    text = "Marie   Curie was born\nin Warsaw in 1867.  She moved to Paris."
    for c in chunk(text, TOK, max_tokens=4, overlap=1):
        assert c in text


def test_short_text_is_one_chunk_and_empty_text_is_none():
    assert chunk("Short page.", TOK, max_tokens=10, overlap=3) == ["Short page."]
    assert chunk("   ", TOK, max_tokens=10, overlap=3) == []


def test_top_passages_ranks_by_relevance_to_the_claim():
    chunks = [
        "The tower was painted brown in 1968.",
        "Marie Curie won the Nobel Prize in Physics in 1903 with Pierre Curie.",
        "Paris is the capital of France.",
        "Curie later won a second Nobel Prize, in Chemistry.",
    ]
    top = top_passages("Marie Curie won the Nobel Prize in Physics in 1903.", chunks, k=2)
    assert top == [chunks[1], chunks[3]]


def test_top_passages_with_fewer_chunks_than_k():
    assert top_passages("claim", ["only one"], k=3) == ["only one"]
    assert top_passages("claim", [], k=3) == []


def test_clean_text_strips_markdown_and_citations_but_keeps_content():
    from factassessor.passages import clean_text

    md = """## The most visited monument
**As France's symbol**, it welcomes _almost_ 7 million visitors.[6][7] She named it _polonium_.[c]
| Height | 330 m |
|---|---|
| Opened | 1889 |
Visitors a year
7 000 000
  * Facebook
  * Twitter
  * Paris is the capital and largest city of France
---
"""
    assert clean_text(md) == (
        "The most visited monument\n"
        "As France's symbol, it welcomes almost 7 million visitors. She named it polonium.\n"
        "Height 330 m\n"
        "Opened 1889\n"
        "Visitors a year\n"
        "7 000 000\n"
        "Paris is the capital and largest city of France"
    )
