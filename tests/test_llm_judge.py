import asyncio
import re

from pydantic_ai.messages import ModelResponse, ToolCallPart
from pydantic_ai.models.function import FunctionModel

from factassessor import LLMJudge


def fake_llm(calls, skip_ids=(), fail=False):
    """Answers every [id] in the prompt: 'supports' if its evidence mentions 1958, else not_enough_info."""

    def respond(messages, info):
        prompt = messages[-1].parts[-1].content
        calls.append(prompt)
        if fail:
            raise RuntimeError("429 rate limited")
        items = []
        for block in prompt.split("\n\n"):
            m = re.match(r"\[(\d+)\] claim: .*\n    evidence: (.*)", block, re.S)
            if m and int(m.group(1)) not in skip_ids:
                label = "supports" if "1958" in m.group(2) else "not_enough_info"
                items.append({"id": int(m.group(1)), "label": label, "confidence": 0.9})
        return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, {"items": items})])

    return FunctionModel(respond)


def snippets(*texts):
    return [{"url": f"https://s{i}.org", "title": "t", "snippet": t} for i, t in enumerate(texts)]


async def test_concurrent_requests_from_different_claims_share_one_call():
    calls = []
    j = LLMJudge(window_ms=20)
    with j.agent.override(model=fake_llm(calls)):
        a, b = await asyncio.gather(
            j.judge("NASA was founded in 1958.", snippets("NASA was established in 1958.", "Paris is in France.")),
            j.judge("Python was created in 1991.", snippets("Guido released Python in 1991.")),
        )
    assert len(calls) == 1  # one API call for both claims
    assert [(e.url, e.label, e.prob) for e in a] == [("https://s0.org", "supports", 0.9), ("https://s1.org", "not_enough_info", 0.9)]
    assert [e.label for e in b] == ["not_enough_info"]


async def test_window_zero_sends_each_request_on_its_own():
    calls = []
    j = LLMJudge(window_ms=0)
    with j.agent.override(model=fake_llm(calls)):
        await asyncio.gather(j.judge("c1", snippets("1958")), j.judge("c2", snippets("x")))
    assert len(calls) == 2


async def test_big_batches_are_split_into_concurrent_calls():
    calls = []
    j = LLMJudge(window_ms=20, max_pairs=2)
    with j.agent.override(model=fake_llm(calls)):
        ev = await j.judge("NASA was founded in 1958.", snippets(*[f"text {i} 1958" for i in range(5)]))
    assert len(calls) == 3 and len(ev) == 5 and all(e.label == "supports" for e in ev)


async def test_items_the_model_skips_count_as_no_evidence():
    j = LLMJudge(window_ms=0)
    with j.agent.override(model=fake_llm([], skip_ids={1})):
        ev = await j.judge("NASA was founded in 1958.", snippets("in 1958", "also 1958"))
    assert [(e.label, e.prob) for e in ev] == [("supports", 0.9), ("not_enough_info", 0.0)]


async def test_an_api_failure_degrades_to_no_evidence_instead_of_crashing():
    j = LLMJudge(window_ms=0)
    with j.agent.override(model=fake_llm([], fail=True)):
        ev = await j.judge("claim", snippets("evidence"))
    assert [(e.label, e.prob) for e in ev] == [("not_enough_info", 0.0)]


async def test_pages_are_cut_to_their_most_relevant_passage():
    filler = " ".join(f"filler{i}" for i in range(3000))
    page = {"url": "https://wiki.org", "title": "NASA", "text": f"{filler} NASA was founded in 1958 by an act of Congress. {filler}"}
    calls = []
    j = LLMJudge(window_ms=0, passage_words=50)
    with j.agent.override(model=fake_llm(calls)):
        [ev] = await j.judge("NASA was founded in 1958.", [page])
    assert ev.source == "page" and ev.label == "supports" and "1958" in ev.text and len(ev.text.split()) <= 50
