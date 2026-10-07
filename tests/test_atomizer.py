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


async def test_source_queries_are_written_once_per_text_and_carried_by_every_atom():
    # the atomizer reads the whole text, so in the same call it can say what document the text is from
    atomizer = LLMAtomizer(source_queries=2)
    model = TestModel(custom_output_args={
        "atoms": [{"text": "A"}, {"text": "B"}],
        "source_queries": ["Nepal earthquake 2017 magnitude damage casualties", " Gorkha earthquake Nepal 2015 report ", ""],
    })
    with atomizer.agent.override(model=model):
        atoms = await atomizer.atomize(TEXT)
    queries = ["Nepal earthquake 2017 magnitude damage casualties", "Gorkha earthquake Nepal 2015 report"]
    assert [a.source_queries for a in atoms] == [queries, queries]  # stripped, blanks dropped
    assert "source_queries" in atomizer.instructions and "2" in atomizer.instructions  # asked for, and how many


async def test_no_more_source_queries_than_asked_for_and_duplicates_dropped():
    atomizer = LLMAtomizer(source_queries=2)
    model = TestModel(custom_output_args={"atoms": [{"text": "A"}], "source_queries": ["q1", "q1", "q2", "q3"]})
    with atomizer.agent.override(model=model):
        [atom] = await atomizer.atomize(TEXT)
    assert atom.source_queries == ["q1", "q2"]


async def test_zero_source_queries_asks_for_none_and_keeps_none():
    atomizer = LLMAtomizer()  # the atomizer alone: claims only
    assert atomizer.source_queries == 0 and "source_queries" not in atomizer.instructions
    model = TestModel(custom_output_args={"atoms": [{"text": "A"}], "source_queries": ["ignored"]})
    with atomizer.agent.override(model=model):
        [atom] = await atomizer.atomize(TEXT)
    assert atom.source_queries == []


async def test_fallback_false_raises_on_an_llm_error_instead_of_using_sentences():
    # an eval must stop when the LLM is down (no credits): the sentence fallback has no source queries and would
    # quietly measure something else
    import pytest
    from pydantic_ai.models.function import FunctionModel

    def down(messages, info):
        raise RuntimeError("insufficient_quota")

    for fallback, expect_raise in ((True, False), (False, True)):
        atomizer = LLMAtomizer(source_queries=2, fallback=fallback)
        with atomizer.agent.override(model=FunctionModel(down)):
            if expect_raise:
                with pytest.raises(RuntimeError):
                    await atomizer.atomize(TEXT)
            else:
                assert [a.source_queries for a in await atomizer.atomize(TEXT)][0] == []


def test_the_model_is_asked_for_exactly_n_source_queries():
    assert "exactly 2" in LLMAtomizer(source_queries=2).instructions


async def test_source_queries_alone_for_claims_from_elsewhere():
    # given claims skip the atomizer, but the text's source queries still help find its source document
    atomizer = LLMAtomizer(source_queries=2)
    model = TestModel(custom_output_args={"source_queries": ["Nepal earthquake 2015 report", "Gorkha earthquake damage"]})
    with atomizer.source_agent.override(model=model):
        assert await atomizer.source_queries_for(TEXT) == ["Nepal earthquake 2015 report", "Gorkha earthquake damage"]
    assert await LLMAtomizer().source_queries_for(TEXT) == []  # asked for none


async def test_an_llm_outage_with_fallback_is_logged_as_a_warning(logs):
    # the sentence fallback is never silent: a run with degraded claims says so
    from pydantic_ai.models.function import FunctionModel

    def down(messages, info):
        raise RuntimeError("insufficient_quota")

    atomizer = LLMAtomizer(source_queries=2)
    with atomizer.agent.override(model=FunctionModel(down)), atomizer.source_agent.override(model=FunctionModel(down)):
        await atomizer.atomize(TEXT)
        await atomizer.source_queries_for(TEXT)
    warnings = [line for line in logs if line.startswith("WARNING")]
    assert len(warnings) == 2 and all("insufficient_quota" in w for w in warnings)
    assert "falling back to sentences" in warnings[0] and "search on their own" in warnings[1]
