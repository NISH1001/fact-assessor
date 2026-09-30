import asyncio
import re

import pytest
from pydantic_ai.messages import ModelResponse, ToolCallPart
from pydantic_ai.models.function import FunctionModel

from factassessor import DecisionJudge, DecisionRequest, DecisionRunner, LLMRunner, Question

STANCE = {
    "stance": Question(
        type="choice",
        instructions="According to `evidence`, is `claim` true or false?",
        criteria={"supports": "true", "refutes": "false", "not_enough_info": "unknown"},
    )
}


def fake_llm(calls, skip_ids=(), fail=False, choice=None):
    """Answers every [id] in the prompt: 'supports' if its evidence mentions 1958, else not_enough_info."""

    def respond(messages, info):
        prompt = messages[-1].parts[-1].content
        calls.append(prompt)
        if fail:
            raise RuntimeError("429 rate limited")
        items = []
        for block in prompt.split("\n\n"):
            m = re.match(r"\[(\d+)\]\n.*?^evidence: (.*?)$", block, re.S | re.M)
            if m and int(m.group(1)) not in skip_ids:
                label = choice or ("supports" if "1958" in m.group(2) else "not_enough_info")
                items.append({"id": int(m.group(1)), "choice": label, "confidence": 0.9})
        return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, {"items": items})])

    return FunctionModel(respond)


def request(evidence, claim="NASA was founded in 1958."):
    return DecisionRequest(state={"evidence": evidence, "claim": claim}, questions=STANCE)


async def test_each_question_is_one_prompt_item_and_the_pick_becomes_probabilities():
    calls = []
    runner = LLMRunner()
    assert isinstance(runner, DecisionRunner)
    with runner.agent.override(model=fake_llm(calls)):
        a, b = await runner.predict([request("NASA was established in 1958."), request("Paris is in France.")])
    assert len(calls) == 1  # one call for both
    prompt = calls[0]
    assert "[1]\nevidence: NASA was established in 1958.\nclaim: NASA was founded in 1958." in prompt  # state fields in order
    assert "According to `evidence`, is `claim` true or false?" in prompt and "supports: true" in prompt
    assert a.answers["stance"].label == "supports" and a.answers["stance"].confidence == 0.9
    assert a.answers["stance"].probabilities == pytest.approx({"supports": 0.9, "refutes": 0.05, "not_enough_info": 0.05})
    assert b.answers["stance"].label == "not_enough_info"


async def test_big_batches_are_split_into_concurrent_calls():
    calls = []
    runner = LLMRunner(batch_size=2)
    with runner.agent.override(model=fake_llm(calls)):
        res = await runner.predict([request(f"text {i} 1958") for i in range(5)])
    assert len(calls) == 3 and [r.answers["stance"].label for r in res] == ["supports"] * 5


async def test_items_the_model_skips_or_misnames_get_even_probabilities():
    runner = LLMRunner()
    with runner.agent.override(model=fake_llm([], skip_ids={2})):
        a, b = await runner.predict([request("in 1958"), request("also 1958")])
    assert a.answers["stance"].label == "supports"
    assert b.answers["stance"].probabilities == {"supports": 1 / 3, "refutes": 1 / 3, "not_enough_info": 1 / 3}
    assert b.answers["stance"].confidence == 0.0
    with runner.agent.override(model=fake_llm([], choice="Supports")):  # case slips are forgiven
        [c] = await runner.predict([request("x")])
    assert c.answers["stance"].label == "supports"


async def test_an_api_failure_raises_so_the_claim_comes_back_unverified():
    runner = LLMRunner()
    with runner.agent.override(model=fake_llm([], fail=True)):
        with pytest.raises(RuntimeError):
            await runner.predict([request("evidence")])


async def test_the_judge_runs_on_it_and_concurrent_claims_share_a_call():
    calls = []
    runner = LLMRunner()
    judge = DecisionJudge(runner)
    with runner.agent.override(model=fake_llm(calls)):
        a, b = await asyncio.gather(
            judge.judge("NASA was founded in 1958.", [{"url": "s0", "title": "t", "snippet": "NASA was established in 1958."},
                                                      {"url": "s1", "title": "t", "snippet": "Paris is in France."}]),
            judge.judge("Python was created in 1991.", [{"url": "s2", "title": "t", "snippet": "Guido released Python in 1991."}]),
        )
    assert [(e.url, e.label, e.prob) for e in a] == [("s0", "supports", 0.9), ("s1", "not_enough_info", 0.9)]
    assert [e.label for e in b] == ["not_enough_info"]
    assert len(calls) == 1  # both claims' passages in one prompt


def test_reasoning_is_off_by_default_where_the_model_allows():
    assert LLMRunner("openai:gpt-5-nano").agent.model_settings == {"openai_reasoning_effort": "minimal"}
    assert LLMRunner().agent.model_settings == {"openai_reasoning_effort": "none"}
    assert LLMRunner("openai:gpt-5-nano", model_settings={"temperature": 0}).agent.model_settings == {"temperature": 0}
