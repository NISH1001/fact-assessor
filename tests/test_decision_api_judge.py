import asyncio
import json

import httpx
import pytest

from factassessor.judges import DecisionAPIJudge
from factassessor.judges.laya import QUESTION

CLAIM = "Marie Curie won the Nobel Prize in Physics in 1903."


def answer(label, prob, cost=0.00001):
    others = [x for x in ("supports", "refutes", "not_enough_info") if x != label]
    probs = {label: prob, others[0]: round(1 - prob, 3), others[1]: 0.0}
    return {"answers": {"stance": {"type": "choice", "choice": label, "probabilities": probs, "confidence": 0.8}},
            "usage": {"cost": cost}}


def judge(handler, **kwargs):
    j = DecisionAPIJudge(api_key="test-key", **kwargs)
    j._http = httpx.AsyncClient(transport=httpx.MockTransport(handler), headers=j._headers())
    return j


async def test_each_passage_is_one_decision_with_layas_question_and_state():
    bodies = []

    async def handler(request):
        body = json.loads(request.content)
        bodies.append(body)
        assert request.headers["authorization"] == "Bearer test-key"
        return httpx.Response(200, json=answer("supports" if "1903" in body["state"]["evidence"] else "not_enough_info", 0.9))

    j = judge(handler, model="~typesafe/jev-latest")
    ev = await j.judge(CLAIM, [
        {"url": "u1", "title": "Nobel", "snippet": "Curie shared the 1903 Nobel Prize in Physics."},
        {"url": "u2", "title": "Paris", "snippet": "Paris is in France."},
    ])
    assert [(e.url, e.source, e.label, e.prob) for e in ev] == [("u1", "snippet", "supports", 0.9), ("u2", "snippet", "not_enough_info", 0.9)]
    assert bodies[0]["model"] == "~typesafe/jev-latest" and bodies[0]["questions"] == QUESTION
    assert bodies[0]["state"] == {"evidence": "Curie shared the 1903 Nobel Prize in Physics.", "claim": CLAIM}
    assert j.cost == pytest.approx(0.00002)  # usage.cost summed


async def test_pages_are_cut_into_word_windows_and_ranked():
    seen = []

    async def handler(request):
        seen.append(json.loads(request.content)["state"]["evidence"])
        return httpx.Response(200, json=answer("supports", 0.8))

    filler = " ".join(f"filler{i}" for i in range(400))
    page = {"url": "wiki", "title": "Marie Curie", "text": f"{filler} Curie won the Nobel Prize in Physics in 1903. {filler}"}
    ev = await judge(handler, passages_per_page=2, passage_words=50).judge(CLAIM, [page])
    assert len(ev) == 2 and all(e.source == "page" and len(e.text.split()) <= 50 for e in ev)
    assert "1903" in seen[0]  # the best window first


async def test_requests_in_flight_are_capped():
    running = peak = 0

    async def handler(request):
        nonlocal running, peak
        running += 1
        peak = max(peak, running)
        await asyncio.sleep(0.01)
        running -= 1
        return httpx.Response(200, json=answer("refutes", 0.7))

    ev = await judge(handler, max_concurrent=3).judge(CLAIM, [{"url": f"u{i}", "title": "", "snippet": f"s {i}"} for i in range(9)])
    assert len(ev) == 9 and all(e.label == "refutes" for e in ev) and peak == 3


async def test_rate_limits_are_retried_with_backoff():
    calls = 0

    async def handler(request):
        nonlocal calls
        calls += 1
        return httpx.Response(429, text="slow down") if calls == 1 else httpx.Response(200, json=answer("refutes", 0.7))

    [e] = await judge(handler).judge(CLAIM, [{"url": "u", "title": "", "snippet": "s"}])
    assert e.label == "refutes" and calls == 2


async def test_a_missing_key_is_a_clear_error(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY"):
        await DecisionAPIJudge().aload()
