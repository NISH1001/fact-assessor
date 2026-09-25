from pydantic_ai.models.function import FunctionModel
from pydantic_ai.models.test import TestModel

from factassessor import LLMAtomizer

TEXT = "It was believed that Nepal's earthquake in 2017 of 7.8 magnitude scale caused massive damage. Total lives lost were 1 million people."


def returning(*atoms):
    return TestModel(custom_output_args={"atoms": [{"text": t, "source": s} for t, s in atoms]})


async def test_llm_atoms_get_ids_and_spans_of_their_source_quote():
    atomizer = LLMAtomizer()
    with atomizer.agent.override(model=returning(
        ("A major earthquake struck Nepal in 2017.", "Nepal's earthquake in 2017"),
        ("The Nepal earthquake had a magnitude of 7.8.", "7.8 magnitude scale"),
        ("The Nepal earthquake killed 1 million people.", "Total lives lost were 1 million people."),
    )):
        atoms = await atomizer.atomize(TEXT)
    assert [a.id for a in atoms] == [0, 1, 2]
    assert [a.text for a in atoms] == [
        "A major earthquake struck Nepal in 2017.",
        "The Nepal earthquake had a magnitude of 7.8.",
        "The Nepal earthquake killed 1 million people.",
    ]
    assert [TEXT[s:e] for s, e in (a.span for a in atoms)] == [
        "Nepal's earthquake in 2017", "7.8 magnitude scale", "Total lives lost were 1 million people."
    ]


async def test_source_quote_match_ignores_case_and_whitespace():
    atomizer = LLMAtomizer()
    with atomizer.agent.override(model=returning(("Nepal had a quake in 2017.", "nepal's   EARTHQUAKE in 2017"))):
        [atom] = await atomizer.atomize(TEXT)
    assert TEXT[atom.span[0]:atom.span[1]] == "Nepal's earthquake in 2017"


async def test_unmatched_quote_falls_back_to_the_best_sentence():
    atomizer = LLMAtomizer()
    with atomizer.agent.override(model=returning(("The Nepal earthquake killed 1 million people.", "a paraphrase not in the text"))):
        [atom] = await atomizer.atomize(TEXT)
    assert TEXT[atom.span[0]:atom.span[1]] == "Total lives lost were 1 million people."


async def test_model_failure_falls_back_to_sentences():
    def boom(messages, info):
        raise RuntimeError("429 spend limit")

    atomizer = LLMAtomizer()
    with atomizer.agent.override(model=FunctionModel(boom)):
        atoms = await atomizer.atomize(TEXT)
    assert [a.text for a in atoms] == [
        "It was believed that Nepal's earthquake in 2017 of 7.8 magnitude scale caused massive damage.",
        "Total lives lost were 1 million people.",
    ]
    assert [TEXT[s:e] for s, e in (a.span for a in atoms)] == [a.text for a in atoms]


async def test_empty_text_skips_the_model():
    atomizer = LLMAtomizer()
    model = TestModel()
    with atomizer.agent.override(model=model):
        assert await atomizer.atomize("   ") == []
    assert model.last_model_request_parameters is None
