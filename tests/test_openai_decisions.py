"""OpenAIDecisionRunner: gpt-6-luna through OpenAI's Decisions API (POST /v1/decisions), offline against a fake."""

import json

import httpx
import pytest

from factassessor import DecisionRequest, DecisionRunner, MissingAPIKeyError, OpenAIDecisionRunner, Question
from factassessor.judges.decision import QUESTION

CLAIM = "Marie Curie won the Nobel Prize in Physics in 1903."


def request(evidence):
    return DecisionRequest(state={"evidence": evidence, "claim": CLAIM}, questions=QUESTION)


def fake(bodies, status=200, fail_first=0, text=None):
    """OpenAI's Decisions API: 'supports' for a question whose passage mentions 1903."""
    calls = 0

    def handler(req):
        nonlocal calls
        calls += 1
        if calls <= fail_first:
            return httpx.Response(503, text="busy")
        if status != 200:
            return httpx.Response(status, text=text or "")
        body = json.loads(req.content)
        bodies.append(body)
        assert req.headers["authorization"] == "Bearer test-key" and req.url.path == "/v1/decisions"
        answers = []
        for q in body["questions"]:
            index = q["instructions"].split("`evidence[")[1].split("]")[0] if "`evidence[" in q["instructions"] else None
            passage = body["input"].split(f"evidence[{index}]: ")[1].split("\n")[0] if index is not None else body["input"]
            p = 0.9 if "1903" in passage else 0.05
            probs = [{"value": "supports", "probability": p}, {"value": "refutes", "probability": 0.05},
                     {"value": "not_enough_info", "probability": 0.95 - p}]
            answers.append({"type": "choice", "name": q["name"], "choice": max(probs, key=lambda x: x["probability"])["value"],
                            "probabilities": probs, "confidence": 0.9})
        return httpx.Response(200, json={"answers": answers})

    return handler


def runner(handler, **kw):
    r = OpenAIDecisionRunner(**kw)
    r._http = httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="https://api.openai.com",
                                headers={"Authorization": "Bearer test-key"})
    return r


async def test_one_request_becomes_openai_input_text_and_a_choice_question():
    bodies = []
    r = runner(fake(bodies))
    assert isinstance(r, DecisionRunner) and r.model == "gpt-6-luna"
    [res] = await r.predict([request("Curie shared the 1903 Nobel Prize in Physics.")])
    [body] = bodies
    assert body["model"] == "gpt-6-luna"
    assert body["input"] == f"evidence: Curie shared the 1903 Nobel Prize in Physics.\nclaim: {CLAIM}"
    [q] = body["questions"]
    assert (q["type"], q["name"], q["instructions"]) == ("choice", "stance", QUESTION["stance"].instructions)
    assert q["choices"] == [{"value": k, "description": v} for k, v in QUESTION["stance"].criteria.items()]
    assert res.answers["stance"].label == "supports" and res.answers["stance"].probabilities["supports"] == 0.9


async def test_a_claims_passages_are_packed_into_one_call_with_numbered_evidence():
    bodies = []
    res = await runner(fake(bodies)).predict([request("Paris is in France."), request("Curie won it in 1903."), request("Bees.")])
    [body] = bodies  # one call for the claim's three passages
    assert "evidence[0]: Paris is in France.\nevidence[1]: Curie won it in 1903.\nevidence[2]: Bees.\n" in body["input"]
    assert body["input"].endswith(f"claim: {CLAIM}")  # the shared claim once
    assert [q["name"] for q in body["questions"]] == ["stance_0", "stance_1", "stance_2"]
    assert "`evidence[1]`" in body["questions"][1]["instructions"]
    assert [x.answers["stance"].label for x in res] == ["not_enough_info", "supports", "not_enough_info"]


async def test_yes_no_and_score_questions_map_to_openai_predicate_and_score():
    seen = []

    def handler(req):
        body = json.loads(req.content)
        seen.append(body["questions"])
        return httpx.Response(200, json={"answers": [
            {"type": "predicate", "name": "factual", "probability": 0.8},
            {"type": "score", "name": "severity", "score": 1.1, "confidence": 0.5, "probabilities": [
                {"value": 0, "label": "low", "probability": 0.1}, {"value": 1, "label": "mid", "probability": 0.7},
                {"value": 2, "label": "high", "probability": 0.2}]}]})

    q = {"factual": Question(type="noul", instructions="Is `text` factual?"),
         "severity": Question(type="score", instructions="How severe?", criteria={"low": "minor", "mid": "some", "high": "major"})}
    [res] = await runner(handler).predict([DecisionRequest(state={"text": "x"}, questions=q)])
    predicate, score = seen[0]
    assert predicate == {"type": "predicate", "name": "factual", "instructions": "Is `text` factual?"}
    assert score["type"] == "score" and score["levels"] == [{"label": "low", "description": "minor"}, {"label": "mid", "description": "some"},
                                                              {"label": "high", "description": "major"}]
    assert res.answers["factual"].probabilities == {"yes": 0.8, "no": pytest.approx(0.2)} and res.answers["factual"].label == "yes"
    assert res.answers["severity"].probabilities == {"low": 0.1, "mid": 0.7, "high": 0.2} and res.answers["severity"].label == "mid"


async def test_a_busy_service_is_retried_and_a_rejected_request_says_why(monkeypatch):
    import factassessor.decisions as decisions

    async def no_wait(seconds):
        pass

    monkeypatch.setattr(decisions.asyncio, "sleep", no_wait)
    bodies = []
    [res] = await runner(fake(bodies, fail_first=2)).predict([request("1903")])
    assert res.answers["stance"].label == "supports" and len(bodies) == 1
    reason = '{"error": {"message": "Input is too long for this model."}}'
    with pytest.raises(httpx.HTTPStatusError, match="Input is too long"):  # the server's explanation, not just "400"
        await runner(fake([], status=400, text=reason)).predict([request("1903")])


def test_a_missing_openai_key_fails_at_creation_and_points_to_local_laya(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY")
    with pytest.raises(MissingAPIKeyError, match="OPENAI_API_KEY") as err:
        OpenAIDecisionRunner()
    assert "LayaRunner()" in str(err.value)


def test_fact_assessor_runs_its_filter_and_judge_on_it():
    from factassessor import FactAssessor

    fa = FactAssessor(runner=OpenAIDecisionRunner())
    assert isinstance(fa.judge.runner, OpenAIDecisionRunner) and fa.claim_filter.runner is fa.judge.runner
