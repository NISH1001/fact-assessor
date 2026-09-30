from factassessor import LLMAtomizer, LLMRunner
from factassessor._llm import reasoning_off


def test_reasoning_off_matches_what_each_openai_model_accepts():
    # gpt-5 / -mini / -nano reject "none" (400: supported values are minimal, low, medium, high)
    for model in ("openai:gpt-5", "openai:gpt-5-mini", "openai:gpt-5-nano", "openai:gpt-5-nano-2025-08-07"):
        assert reasoning_off(model) == {"openai_reasoning_effort": "minimal"}, model
    for model in ("openai:gpt-5.1", "openai:gpt-5.6-luna", "openai:gpt-6-luna"):  # newer: reasoning can be off
        assert reasoning_off(model) == {"openai_reasoning_effort": "none"}, model
    for model in ("openai:o3", "openai:o4-mini"):  # o-series: no "none" or "minimal"
        assert reasoning_off(model) == {"openai_reasoning_effort": "low"}, model
    for model in ("openai:gpt-4.1-mini", "openai:gpt-4o"):  # not reasoning models: the setting is an error
        assert reasoning_off(model) == {}, model
    assert reasoning_off("anthropic:claude-haiku-4-5") == {}  # other providers: no OpenAI settings


def test_llm_components_default_to_settings_their_model_accepts():
    assert LLMAtomizer("openai:gpt-5-nano").agent.model_settings == {"openai_reasoning_effort": "minimal"}
    assert LLMRunner("openai:gpt-5-nano").agent.model_settings == {"openai_reasoning_effort": "minimal"}
    assert LLMAtomizer().agent.model_settings == {"openai_reasoning_effort": "none"}  # the default model
    assert LLMAtomizer("openai:gpt-5-nano", model_settings={"temperature": 0}).agent.model_settings == {"temperature": 0}
