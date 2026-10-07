"""A component that needs an API key says so when it is created, with what to do instead, not at its first call
(or never: without OPENAI_API_KEY the atomizer used to fall back to sentence atoms silently)."""

import pytest

from factassessor import FactAssessor, LayaRunner, LLMRunner, MissingAPIKeyError, OpenAIDecisionRunner, SerperSearcher, SystemOneRunner
from factassessor.atomizer import LLMAtomizer
from factassessor.decisions.utils import RUN_LOCALLY


def test_jev_without_an_openrouter_key_fails_at_creation_and_points_to_local_laya(monkeypatch, logs):
    monkeypatch.delenv("OPENROUTER_API_KEY")
    with pytest.raises(MissingAPIKeyError, match="OPENROUTER_API_KEY") as err:
        SystemOneRunner()
    assert "LayaRunner()" in str(err.value) and any(line.startswith("ERROR") and "OPENROUTER_API_KEY" in line for line in logs)  # logged too
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


@pytest.mark.parametrize("make, env", [(lambda: SystemOneRunner(), "OPENROUTER_API_KEY"), (lambda: LLMRunner(), "OPENAI_API_KEY"),
                                       (lambda: OpenAIDecisionRunner(), "OPENAI_API_KEY")])
def test_every_api_runner_without_its_key_points_to_the_same_local_alternative(monkeypatch, make, env):
    monkeypatch.delenv(env, raising=False)
    with pytest.raises(MissingAPIKeyError) as err:
        make()
    assert str(err.value).endswith(RUN_LOCALLY)
