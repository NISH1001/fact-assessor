"""factassessor.decisions is a package: the shared types and helpers, then one module per runner."""

import importlib

import pytest

import factassessor
import factassessor.decisions as decisions

LAYOUT = {
    "types": ["Question", "DecisionRequest", "Answer", "DecisionResponse", "DecisionRunner", "DecisionPacking"],
    "utils": ["Batcher", "pack", "post_with_retries"],
    "systemone": ["SystemOneRunner"],
    "openai": ["OpenAIDecisionRunner"],
    "llm": ["LLMRunner"],
    "laya": ["LayaRunner"],
    "gliner": ["GlinerRunner"],
}


@pytest.mark.parametrize("module", LAYOUT)
def test_each_name_lives_in_its_module_and_imports_the_same_from_the_package(module):
    mod = importlib.import_module(f"factassessor.decisions.{module}")
    for name in LAYOUT[module]:
        assert getattr(decisions, name) is getattr(mod, name)
        if hasattr(factassessor, name):  # the public ones: `from factassessor import LayaRunner` as before
            assert getattr(factassessor, name) is getattr(mod, name)


@pytest.mark.parametrize("old", ["factassessor.laya", "factassessor.gliner"])
def test_the_runners_moved_into_the_package(old):
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(old)
