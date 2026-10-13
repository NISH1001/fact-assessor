"""SpanAtomizer: the local span atomizer (the `atomizer` package) behind the Atomizer role. A fake model stands in."""

import asyncio
import sys
from types import SimpleNamespace

import pytest

from factassessor import FactAssessor, Map, SpanAtomizer, Step

TEXT = "Marie Curie won the Nobel Prize in Physics in 1903. She was born in Warsaw. What a life!"


def atom(text, span, verifiable):
    return SimpleNamespace(text=text, span=span, verifiable=verifiable, checkworthy=0.9, score=0.9)


ATOMS = [
    atom("Marie Curie won the Nobel Prize in Physics in 1903.", (0, 51), 1.0),
    atom("Marie Curie was born in Warsaw.", (52, 75), 0.98),
    atom("What a life!", (76, 88), 0.02),
]


class FakeModel:
    """Stands in for `atomizer.Atomizer`: the same atoms for every text; records each `atomize_many` call."""

    def __init__(self, atoms=ATOMS):
        self.atoms, self.calls = atoms, []

    def atomize_many(self, texts, batch_size=16, anchor_threshold=None, ante_threshold=None):
        self.calls.append({"texts": list(texts), "batch_size": batch_size, "anchor_threshold": anchor_threshold,
                           "ante_threshold": ante_threshold})
        return [list(self.atoms) for _ in texts]


def span_atomizer(model=None, **kwargs):
    atomizer = SpanAtomizer(**kwargs)
    atomizer._model = model or FakeModel()
    return atomizer


async def test_atoms_keep_the_models_text_and_span_and_its_verifiable_score_as_the_claim_score():
    atoms = await span_atomizer().atomize(TEXT)  # every atom by default, like the atomizer package
    assert [(a.id, a.text, a.span, a.claim_score) for a in atoms] == [
        (0, "Marie Curie won the Nobel Prize in Physics in 1903.", (0, 51), 1.0),
        (1, "Marie Curie was born in Warsaw.", (52, 75), 0.98),
        (2, "What a life!", (76, 88), 0.02),
    ]
    assert all(a.source_queries == [] for a in atoms)  # fully local: no LLM call, no key


async def test_atoms_below_min_verifiable_are_dropped():
    atoms = await span_atomizer(min_verifiable=0.5).atomize(TEXT)
    assert [a.text for a in atoms] == ["Marie Curie won the Nobel Prize in Physics in 1903.", "Marie Curie was born in Warsaw."]


async def test_min_checkworthy_filters_too_and_auto_is_the_packages_cut_off():
    model = FakeModel([atom("A.", (0, 2), 0.9), atom("B.", (3, 5), 0.3)])
    model.atoms[1].checkworthy = 0.2
    assert [a.text for a in await span_atomizer(model, min_checkworthy=0.5).atomize("A. B.")] == ["A."]
    assert [a.text for a in await span_atomizer(model, min_verifiable="auto").atomize("A. B.")] == ["A."]  # auto: 0.5


async def test_the_decoding_settings_go_to_the_model():
    atomizer = span_atomizer(batch_size=4, anchor_threshold=0.3, ante_threshold=0.2)
    await atomizer.atomize(TEXT)
    assert atomizer._model.calls == [{"texts": [TEXT], "batch_size": 4, "anchor_threshold": 0.3, "ante_threshold": 0.2}]


async def test_an_empty_text_never_reaches_the_model():
    atomizer = span_atomizer()
    assert await atomizer.atomize("  ") == [] and atomizer._model.calls == []


async def test_texts_atomized_at_the_same_time_share_one_model_call():
    atomizer = span_atomizer()
    results = await asyncio.gather(*(atomizer.atomize(f"{TEXT} ({i})") for i in range(3)))
    assert [len(r) for r in results] == [3, 3, 3]
    assert len(atomizer._model.calls) == 1 and len(atomizer._model.calls[0]["texts"]) == 3


# --- in the pipeline ----------------------------------------------------------------------------------------------

class NoHits(Step):
    def __call__(self, queries):
        return Map(lambda q: None)(queries)


class NoJudge:
    async def judge(self, claim, docs):
        return []


async def test_in_the_pipeline_dropped_atoms_are_reported_as_skipped():
    fa = FactAssessor(atomizer=span_atomizer(min_verifiable=0.5), claim_filter=None, searcher=NoHits(), crawler=NoHits(), resolver=None,
                      judge=NoJudge())
    result = await fa.assess(TEXT)
    assert [r.atom.text for r in result.atoms] == ["Marie Curie won the Nobel Prize in Physics in 1903.", "Marie Curie was born in Warsaw."]
    assert [a.text for a in result.skipped] == ["What a life!"]


async def test_without_the_atomizer_package_loading_says_how_to_install_it(monkeypatch):
    monkeypatch.setitem(sys.modules, "atomizer", None)  # import atomizer -> ImportError
    with pytest.raises(ImportError, match="uv add"):
        await SpanAtomizer(model="test/missing-package").aload()
