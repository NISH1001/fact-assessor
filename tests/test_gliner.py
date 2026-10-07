import asyncio
import threading
import time

import numpy as np

from factassessor import Atom, DecisionClaimFilter, DecisionJudge, DecisionRequest, DecisionRunner, GlinerRunner, Question
from factassessor.claim_filters.decision import QUESTION as KIND
from factassessor.decisions.gliner import GlinerModel, gliner_model
from factassessor.judges.decision import QUESTION as STANCE

CLAIM = "Marie Curie won the Nobel Prize in Physics."  # no year: the fake keys "supports" off "1903" in the evidence
LABELS = list(STANCE["stance"].criteria)


class WordPieces:
    """Stands in for `tokenizers.Tokenizer`: one id per whitespace word."""

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


def fake_model():
    model = GlinerModel("fake/repo")
    model.tok = WordPieces()
    model.session = FakeSession(model.tok)
    return model


def runner(**kwargs):
    r = GlinerRunner(**kwargs)
    r._model = fake_model()
    return r


def judge(runner_kwargs=None, **kwargs):
    return DecisionJudge(runner(**(runner_kwargs or {})), **kwargs)


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
    assert len(j.runner.gliner.session.batches) == 1  # both pairs in one forward pass
    assert isinstance(j.runner, DecisionRunner)


async def test_each_row_is_the_question_as_a_task_then_the_state_fields_in_order():
    r = runner()
    [res] = await r.predict([DecisionRequest(state={"evidence": "Curie won in 1903", "claim": CLAIM}, questions=STANCE)])
    assert res.answers["stance"].label == "supports"
    session, tok = r.gliner.session, r.gliner.tok
    row = session.batches[0]["input_ids"][0].tolist()
    task = {"task": "stance", "instruction": "According to evidence, is claim true or false?", "labels": STANCE["stance"].criteria}
    prompt, positions = r.gliner._prompt(task)  # the question's backticks dropped, its criteria as the labels
    assert row[: len(prompt)] == prompt
    words = row[len(prompt) + 1 :]  # after [SEP_TEXT]
    assert words.index(tok.vocab["evidence"]) < words.index(tok.vocab["claim"])  # evidence first: 12/15 vs 11/15
    assert session.batches[0]["label_positions"][0].tolist() == positions and len(positions) == len(LABELS)


async def test_rows_are_padded_and_batches_capped():
    j = judge({"batch_size": 2})
    docs = [{"url": f"u{i}", "title": "t", "snippet": "word " * (i + 1)} for i in range(5)]
    await j.judge(CLAIM, docs)
    batches = j.runner.gliner.session.batches
    assert [len(b["input_ids"]) for b in batches] == [2, 2, 1]
    b = batches[0]
    assert (b["attention_mask"] * (b["input_ids"] == 0)).sum() == 0  # no attention on padding


async def test_pages_are_cut_to_their_most_relevant_passage():
    filler = " ".join(f"filler{i}" for i in range(2000))
    page = {"url": "wiki", "title": "Marie Curie", "text": f"{filler} Curie won the Nobel Prize in Physics in 1903. {filler}"}
    ev = await judge(passage_words=40, passages_per_page=1).judge(CLAIM, [page])
    assert len(ev) == 1 and ev[0].source == "page" and ev[0].label == "supports" and "1903" in ev[0].text
    assert len(ev[0].text.split()) <= 40


async def test_no_docs_skips_the_model():
    j = judge()
    assert await j.judge(CLAIM, []) == []
    assert j.runner.gliner.session.batches == []


async def test_the_claim_filter_scores_p_factual_claim():
    class KindSession:
        def run(self, outputs, feeds):
            return [np.asarray([[2.0, 0.0, 0.0, 0.0]] * len(feeds["input_ids"]), dtype=np.float32)]

    r = runner()
    r.gliner.session = KindSession()
    score = await DecisionClaimFilter(r, threshold=0.4).score(Atom(id=0, text="NASA was founded in 1958.", span=(0, 25)))
    assert abs(score - float(np.exp(2) / (np.exp(2) + 3))) < 1e-6
    assert list(KIND["kind"].criteria)[0] == "factual_claim"


async def test_questions_sharing_a_task_share_forward_passes_and_others_do_not():
    r = runner()
    other = {"kind": Question(type="choice", instructions="What kind of statement is `claim`?", criteria={"a": "x", "b": "y", "c": "z"})}
    res = await r.predict([
        DecisionRequest(state={"evidence": "1903", "claim": CLAIM}, questions=STANCE),
        DecisionRequest(state={"claim": "x"}, questions=other),
        DecisionRequest(state={"evidence": "nope", "claim": CLAIM}, questions=STANCE),
    ])
    assert [len(b["input_ids"]) for b in r.gliner.session.batches] in ([2, 1], [1, 2])  # one pass per task
    assert [x.answers["stance"].label for x in (res[0], res[2])] == ["supports", "not_enough_info"]
    assert set(res[1].answers["kind"].probabilities) == {"a", "b", "c"}


def test_runners_share_one_model_per_model_and_variant():
    assert GlinerRunner().gliner is GlinerRunner().gliner is gliner_model("2.5-decide", "fp32")
    assert GlinerRunner(variant="int8").gliner is not GlinerRunner().gliner


async def test_two_model_calls_run_at_once():
    # one caller left the CPU underused: 5.8 rows/s vs 7.4 with 2 callers x 7 intra-op threads (M3 Max, 14 cores)
    class SlowSession(FakeSession):
        def __init__(self, tok):
            super().__init__(tok)
            self.running = self.peak = 0
            self.lock = threading.Lock()

        def run(self, outputs, feeds):
            with self.lock:
                self.running += 1
                self.peak = max(self.peak, self.running)
            time.sleep(0.2)
            with self.lock:
                self.running -= 1
            return super().run(outputs, feeds)

    model = GlinerModel("fake/repo", workers=2)
    model.tok = WordPieces()
    model.session = SlowSession(model.tok)
    task = {"task": "stance", "instruction": "x", "labels": STANCE["stance"].criteria}
    await asyncio.gather(model.probabilities(task, ["evidence: a claim: b"]), model.probabilities(task, ["evidence: c claim: d"]))
    assert model.session.peak == 2


def test_intra_op_threads_split_the_cores_between_workers(monkeypatch):
    import factassessor.decisions.gliner as g

    monkeypatch.setattr(g.os, "cpu_count", lambda: 14)
    assert GlinerModel("fake/repo").threads == 7  # default: 2 workers x 7 threads
    assert GlinerModel("fake/repo", workers=1).threads == 14
    assert GlinerModel("fake/repo", threads=3).threads == 3


async def test_long_snippets_are_cut_to_their_most_relevant_passage_too():
    # DuckDuckGo snippets can run to 3,000+ tokens; batches pad to the longest row, so one such snippet made a
    # 5-snippet call take 8.3s instead of 0.9s (and put the row far past what the model was trained on)
    filler = " ".join(f"filler{i}" for i in range(2000))
    long_hit = {"url": "ddg", "title": "", "snippet": f"{filler} Curie won the Nobel Prize in Physics in 1903. {filler}"}
    short_hit = {"url": "short", "title": "", "snippet": "Curie shared the 1903 Nobel Prize."}
    j = judge(passage_words=40)
    ev = await j.judge(CLAIM, [long_hit, short_hit])
    assert [(e.url, e.source, e.label) for e in ev] == [("ddg", "snippet", "supports"), ("short", "snippet", "supports")]
    assert len(ev[0].text.split()) <= 40 and "1903" in ev[0].text
    assert ev[1].text == short_hit["snippet"]  # short snippets stay whole
    assert max(len(b["input_ids"][0]) for b in j.runner.gliner.session.batches) < 200


def test_a_judge_on_gliner_takes_three_claims_at_once_by_default():
    # 20-claim text on recorded evidence: no limit -> 20 of 20 timed out; limit 3/5/8 -> 0 timed out, ~35s either
    # way, first verdict at 4.8s / 7.0s / 9.6s
    from factassessor import Verify

    assert DecisionJudge(GlinerRunner()).concurrency == 3 and DecisionJudge(GlinerRunner(concurrency=None)).concurrency is None
    assert Verify(searcher=None, crawler=None, judge=DecisionJudge(GlinerRunner())).concurrency == 3
