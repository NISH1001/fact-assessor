"""A component that needs an API key says so when it is created, with what to do instead, not at its first call
(or never: without OPENAI_API_KEY the atomizer used to fall back to sentence atoms silently)."""

import logging

import pytest

from factassessor import FactAssessor, LayaRunner, LLMRunner, MissingAPIKeyError, SerperSearcher, SystemOneRunner
from factassessor.atomizer import LLMAtomizer


def test_jev_without_an_openrouter_key_fails_at_creation_and_points_to_local_laya(monkeypatch, caplog):
    monkeypatch.delenv("OPENROUTER_API_KEY")
    with caplog.at_level(logging.ERROR), pytest.raises(MissingAPIKeyError, match="OPENROUTER_API_KEY") as err:
        SystemOneRunner()
    assert "LayaRunner()" in str(err.value) and "OPENROUTER_API_KEY" in caplog.text  # logged too
    SystemOneRunner(api_key="sk-or-...")  # passed in: fine
    SystemOneRunner(url="http://localhost:8000/v1/systemone", model="english")  # a local Laya server needs none


def test_an_llm_runner_or_atomizer_without_its_providers_key_fails_at_creation(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY")
    with pytest.raises(MissingAPIKeyError, match="OPENAI_API_KEY") as err:
        LLMRunner("openai:gpt-6-luna")
    assert "LayaRunner()" in str(err.value)
    with pytest.raises(MissingAPIKeyError, match="OPENAI_API_KEY"):
        LLMAtomizer("openai:gpt-6-luna")
    LLMRunner("openrouter:openai/gpt-6-luna")  # another provider: its own key (set)
    LLMRunner("ollama:llama3")                 # a local model: no key


def test_serper_without_a_key_fails_at_creation_and_points_to_searxng(monkeypatch):
    monkeypatch.delenv("SERPER_API_KEY")
    with pytest.raises(MissingAPIKeyError, match="SERPER_API_KEY") as err:
        SerperSearcher()
    assert "SearxngSearcher" in str(err.value)


def test_fact_assessor_defaults_to_jev_and_runs_on_laya_with_no_openrouter_key(monkeypatch):
    fa = FactAssessor()
    assert isinstance(fa.judge.runner, SystemOneRunner) and fa.claim_filter.runner is fa.judge.runner
    monkeypatch.delenv("OPENROUTER_API_KEY")
    with pytest.raises(MissingAPIKeyError, match="LayaRunner"):
        FactAssessor()  # at creation, not at the first check
    fa = FactAssessor(runner=LayaRunner())  # local: no OpenRouter key needed
    assert isinstance(fa.judge.runner, LayaRunner) and fa.claim_filter.runner is fa.judge.runner
