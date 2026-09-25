import numpy as np

from factassessor.gliner import LABELS, GlinerJudge

CLAIM = "Marie Curie won the Nobel Prize in Physics."  # no year: the fake keys "supports" off "1903" in the evidence


class WordPieces:
    """Stands in for `tokenizers.Tokenizer`: one id per whitespace word, with char offsets."""

    def __init__(self):
        self.vocab = {}

    def encode(self, text, add_special_tokens=False):
        import re

        words = [(m.group(), m.span()) for m in re.finditer(r"\S+", text)]
        ids = [self.vocab.setdefault(w, len(self.vocab) + 1) for w, _ in words]
        return type("Encoding", (), {"ids": ids, "offsets": [span for _, span in words]})()


class FakeSession:
    """Logits favour 'supports' when the evidence mentions 1903; records every batch it runs."""

    def __init__(self, tok):
        self.tok, self.batches = tok, []
        self.id_1903 = tok.encode("1903")  # registers the word

    def run(self, outputs, feeds):
        self.batches.append({k: v.copy() for k, v in feeds.items()})
        rows = []
        for ids in feeds["input_ids"]:
            has_1903 = self.tok.vocab["1903"] in ids.tolist()
            rows.append([3.0, 0.0, 0.0] if has_1903 else [0.0, 0.0, 3.0])  # order: supports, refutes, nei
        return [np.asarray(rows, dtype=np.float32)]


def judge(**kwargs):
    j = GlinerJudge(**kwargs)
    tok = WordPieces()
    j._tok, j._session = tok, FakeSession(tok)
    j._prompt_ids, j._label_positions = j._encode_prompt()
    return j


async def test_snippets_are_judged_and_labels_map_through_softmax():
    j = judge()
    ev = await j.judge(CLAIM, [
        {"url": "u1", "title": "Nobel", "snippet": "Curie shared the 1903 Nobel Prize in Physics."},
        {"url": "u2", "title": "Paris", "snippet": "Paris is in France."},
        {"url": "u3", "title": "Empty", "snippet": ""},
    ])
    assert [(e.url, e.source, e.label) for e in ev] == [("u1", "snippet", "supports"), ("u2", "snippet", "not_enough_info")]
    expected = float(np.exp(3) / (np.exp(3) + 2))
    assert abs(ev[0].prob - expected) < 1e-6
    assert len(j._session.batches) == 1  # both pairs in one forward pass


async def test_each_row_is_prompt_then_sep_text_then_evidence_then_claim():
    j = judge()
    await j.judge(CLAIM, [{"url": "u", "title": "t", "snippet": "Curie won in 1903"}])
    row = j._session.batches[0]["input_ids"][0].tolist()
    prompt = j._prompt_ids
    assert row[: len(prompt)] == prompt
    words = row[len(prompt) + 1 :]  # after [SEP_TEXT]
    evidence_word = j._tok.vocab["evidence"]  # the word splitter separates "evidence:" into "evidence" + ":"
    claim_word = j._tok.vocab["claim"]
    assert words.index(evidence_word) < words.index(claim_word)  # evidence first: 12/15 vs 11/15 claim-first
    positions = j._session.batches[0]["label_positions"][0].tolist()
    assert len(positions) == len(LABELS) and all(p < len(prompt) for p in positions)


async def test_rows_are_padded_and_batches_capped():
    j = judge(batch_size=2)
    docs = [{"url": f"u{i}", "title": "t", "snippet": "word " * (i + 1)} for i in range(5)]
    await j.judge(CLAIM, docs)
    assert [len(b["input_ids"]) for b in j._session.batches] == [2, 2, 1]
    b = j._session.batches[0]
    assert (b["attention_mask"] * (b["input_ids"] == 0)).sum() == 0  # no attention on padding


async def test_pages_are_cut_to_their_most_relevant_passage():
    filler = " ".join(f"filler{i}" for i in range(2000))
    page = {"url": "wiki", "title": "Marie Curie", "text": f"{filler} Curie won the Nobel Prize in Physics in 1903. {filler}"}
    j = judge(passage_tokens=40)
    ev = await j.judge(CLAIM, [page])
    assert len(ev) == 1 and ev[0].source == "page" and ev[0].label == "supports" and "1903" in ev[0].text
    assert len(ev[0].text.split()) <= 40


async def test_no_docs_skips_the_model():
    j = judge()
    assert await j.judge(CLAIM, []) == []
    assert j._session.batches == []
