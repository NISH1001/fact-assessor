from pydantic_ai.models.function import FunctionModel
from pydantic_ai.models.test import TestModel

from factassessor import LLMAtomizer

TEXT = "It was believed that Nepal's earthquake in 2017 of 7.8 magnitude scale caused massive damage. Total lives lost were 1 million people."


def returning(*atoms):
    return TestModel(custom_output_args={"atoms": [{"text": t} for t in atoms]})


async def test_llm_atoms_get_ids_and_the_span_of_the_sentence_they_came_from():
    atomizer = LLMAtomizer()
    with atomizer.agent.override(model=returning(
        "A major earthquake struck Nepal in 2017.",
        "The Nepal earthquake had a magnitude of 7.8.",
        "The Nepal earthquake killed 1 million people.",
    )):
        atoms = await atomizer.atomize(TEXT)
    assert [a.id for a in atoms] == [0, 1, 2]
    assert [a.text for a in atoms] == [
        "A major earthquake struck Nepal in 2017.",
        "The Nepal earthquake had a magnitude of 7.8.",
        "The Nepal earthquake killed 1 million people.",
    ]
    first = "It was believed that Nepal's earthquake in 2017 of 7.8 magnitude scale caused massive damage."
    assert [TEXT[s:e] for s, e in (a.span for a in atoms)] == [first, first, "Total lives lost were 1 million people."]


async def test_the_model_is_asked_for_claims_only_no_source_quote():
    assert "source" not in LLMAtomizer().instructions and "quote" not in LLMAtomizer().instructions


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


async def test_source_query_is_written_once_per_text_and_carried_by_every_atom():
    # the atomizer reads the whole text, so in the same call it can say what document the text is from
    atomizer = LLMAtomizer(source_query=True)
    model = TestModel(custom_output_args={
        "atoms": [{"text": "A"}, {"text": "B"}],
        "source_query": "Nepal earthquake 2017 magnitude damage casualties",
    })
    with atomizer.agent.override(model=model):
        atoms = await atomizer.atomize(TEXT)
    assert [a.source_query for a in atoms] == ["Nepal earthquake 2017 magnitude damage casualties"] * 2
    assert "source_query" in atomizer.instructions  # the model is asked for it


async def test_without_the_flag_no_source_query_is_asked_for_or_kept():
    atomizer = LLMAtomizer()
    assert "source_query" not in atomizer.instructions
    model = TestModel(custom_output_args={"atoms": [{"text": "A"}], "source_query": "ignored"})
    with atomizer.agent.override(model=model):
        [atom] = await atomizer.atomize(TEXT)
    assert atom.source_query is None
