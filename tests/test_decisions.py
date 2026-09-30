import asyncio
import json
import re

import httpx
import pytest

from factassessor.decisions import Answer, Batcher, DecisionRequest, DecisionResponse, DecisionRunner, DecisionPacking, Question, SystemOneRunner

CLAIM = "Marie Curie won the Nobel Prize in Physics in 1903."
STANCE = {
    "stance": Question(
        type="choice",
        instructions="According to `evidence`, is `claim` true or false?",
        criteria={"supports": "true", "refutes": "false", "not_enough_info": "unknown"},
    )
}


def request(evidence, claim=CLAIM):
    return DecisionRequest(state={"evidence": evidence, "claim": claim}, questions=STANCE)


def answer(label, prob):
    others = [x for x in ("supports", "refutes", "not_enough_info") if x != label]
    return {"type": "choice", "choice": label, "probabilities": {label: prob, others[0]: round(1 - prob, 3), others[1]: 0.0}}


def jev(bodies, cost=0.00001, fail_first=0):
    """A fake System One server: 'supports' when the evidence a question refers to mentions 1903."""
    calls = 0

    async def handler(req):
        nonlocal calls
        calls += 1
        if calls <= fail_first:
            return httpx.Response(429, text="slow down")
        body = json.loads(req.content)
        bodies.append(body)
        assert req.headers["authorization"] == "Bearer test-key"
        answers = {}
        for key, q in body["questions"].items():
            state = body["state"]
            if isinstance(state, dict):  # the first backticked reference is the evidence field, maybe indexed
                name, index = re.search(r"`(\w+)(?:\[(\d+)\])?`", q["instructions"]).groups()
                text = state[name][int(index)] if index is not None else state[name]
            else:
                text = state
            answers[key] = answer("supports" if "1903" in text else "not_enough_info", 0.9)
        return httpx.Response(200, json={"answers": answers, "usage": {"cost": cost, "prompt_tokens": 100}})

    return handler


def runner(handler, **kwargs):
    r = SystemOneRunner(api_key="test-key", **kwargs)
    r._http = httpx.AsyncClient(transport=httpx.MockTransport(handler), headers=r._headers())
    return r


async def test_one_request_goes_out_verbatim_as_layas_request():
    bodies = []
    [res] = await runner(jev(bodies), model="~typesafe/jev-latest").predict([request("Curie shared the 1903 Nobel Prize.")])
    assert bodies == [{
        "model": "~typesafe/jev-latest",
        "state": {"evidence": "Curie shared the 1903 Nobel Prize.", "claim": CLAIM},
        "questions": {"stance": {"type": "choice", "instructions": STANCE["stance"].instructions, "criteria": STANCE["stance"].criteria}},
    }]
    assert isinstance(res, DecisionResponse)
    assert res.answers["stance"].label == "supports" and res.answers["stance"].probabilities["supports"] == 0.9
    assert res.usage == {"cost": 0.00001, "prompt_tokens": 100}


async def test_a_claims_passages_are_packed_into_one_call_as_a_list():
    bodies = []
    r = runner(jev(bodies), batch_size=40)
    res = await r.predict([request("Curie shared the 1903 Nobel Prize."), request("Paris is in France."), request("Nobel 1903 again.")])
    assert len(bodies) == 1  # one HTTP call
    state, questions = bodies[0]["state"], bodies[0]["questions"]
    assert state == {  # the shared claim once, the passages as a list, field order kept (evidence first)
        "evidence": ["Curie shared the 1903 Nobel Prize.", "Paris is in France.", "Nobel 1903 again."],
        "claim": CLAIM,
    }
    assert list(state) == ["evidence", "claim"] and list(questions) == ["stance_0", "stance_1", "stance_2"]
    assert questions["stance_1"]["instructions"] == "According to `evidence[1]`, is `claim` true or false?"
    assert questions["stance_1"]["criteria"] == STANCE["stance"].criteria
    assert [x.answers["stance"].label for x in res] == ["supports", "not_enough_info", "supports"]  # each its own answer
    assert [x.usage["cost"] for x in res] == pytest.approx([0.00001 / 3] * 3)  # the call's cost, shared out
    assert r.cost == pytest.approx(0.00001)


async def test_a_field_that_differs_between_requests_becomes_a_list_too():
    bodies = []
    await runner(jev(bodies)).predict([request("a 1903", claim="Claim one."), request("b", claim="Claim two.")])
    state, questions = bodies[0]["state"], bodies[0]["questions"]
    assert state == {"evidence": ["a 1903", "b"], "claim": ["Claim one.", "Claim two."]}
    assert questions["stance_1"]["instructions"] == "According to `evidence[1]`, is `claim[1]` true or false?"


async def test_packs_are_cut_at_batch_size():
    bodies = []
    res = await runner(jev(bodies), batch_size=2).predict([request(f"passage {i} 1903") for i in range(5)])
    assert [len(b["questions"]) for b in bodies] == [2, 2, 1]
    assert len(res) == 5 and all(x.answers["stance"].label == "supports" for x in res)


async def test_by_default_each_predict_call_is_packed_on_its_own():
    bodies = []
    r = runner(jev(bodies), batch_size=2)
    assert r.packing is DecisionPacking.CALL and SystemOneRunner(api_key="k", packing="none").packing is DecisionPacking.NONE
    a, b = await asyncio.gather(r.predict([request("a 1903"), request("b"), request("c 1903")]), r.predict([request("d")]))
    assert sorted(len(b["questions"]) for b in bodies) == [1, 1, 2]  # the first call's 3 as 2 + 1; the second's 1 alone
    assert [x.answers["stance"].label for x in a] == ["supports", "not_enough_info", "supports"]
    assert [x.answers["stance"].label for x in b] == ["not_enough_info"]


async def test_packing_all_lets_concurrent_callers_share_one_call():
    bodies = []
    r = runner(jev(bodies), packing=DecisionPacking.ALL)
    a, b = await asyncio.gather(r.predict([request("a 1903"), request("b")]), r.predict([request("c 1903")]))
    assert len(bodies) == 1 and len(bodies[0]["questions"]) == 3
    assert [x.answers["stance"].label for x in a] == ["supports", "not_enough_info"]
    assert [x.answers["stance"].label for x in b] == ["supports"]


async def test_string_states_are_sent_one_per_call():
    bodies = []
    q = {"kind": Question(type="choice", instructions="What kind of text is this?", criteria={"a": "x", "b": "y"})}
    res = await runner(jev(bodies)).predict([DecisionRequest(state="1903 text", questions=q), DecisionRequest(state="other", questions=q)])
    assert [b["state"] for b in bodies] == ["1903 text", "other"]
    assert [x.answers["kind"].label for x in res] == ["supports", "not_enough_info"]


async def test_calls_in_flight_are_capped():
    running = peak = 0

    async def handler(req):
        nonlocal running, peak
        running += 1
        peak = max(peak, running)
        await asyncio.sleep(0.01)
        running -= 1
        return httpx.Response(200, json={"answers": {"stance": answer("refutes", 0.7)}})

    res = await runner(handler, packing="none", max_concurrent=3).predict([request(f"p{i}") for i in range(9)])
    assert len(res) == 9 and all(x.answers["stance"].label == "refutes" for x in res) and peak == 3


async def test_rate_limits_are_retried_with_backoff():
    bodies = []
    [res] = await runner(jev(bodies, fail_first=1)).predict([request("1903")])
    assert res.answers["stance"].label == "supports" and len(bodies) == 1


async def test_a_missing_key_is_a_clear_error_but_a_laya_server_needs_none(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY"):
        await SystemOneRunner().aload()
    assert "authorization" not in SystemOneRunner(url="http://localhost:8000/v1/systemone", model="english")._headers()


def test_runners_are_checked_structurally():
    class Conforming:
        batch_size = 8

        async def predict(self, requests):
            return []

    assert isinstance(SystemOneRunner(api_key="k"), DecisionRunner)
    assert isinstance(Conforming(), DecisionRunner)
    assert not isinstance(object(), DecisionRunner)


def test_an_answer_labels_its_most_probable_option():
    assert Answer(probabilities={"a": 0.2, "b": 0.7, "c": 0.1}).label == "b"
    assert Answer(choice="a", probabilities={}).label == "a"


async def test_batcher_merges_callers_and_keeps_each_ones_results_in_order():
    calls = []

    async def run(items):
        calls.append(list(items))
        return [f"r-{x}" for x in items]

    b = Batcher(run, max_wait_ms=5, max_concurrent=1)
    x, y = await asyncio.gather(b.submit(["a1", "a2"]), b.submit(["b1"]))
    assert calls == [["a1", "a2", "b1"]] and x == ["r-a1", "r-a2"] and y == ["r-b1"]


async def test_batcher_failure_reaches_every_caller_and_a_cancelled_caller_is_left_out():
    async def failing(items):
        raise RuntimeError("out of memory")

    b = Batcher(failing, max_wait_ms=1, max_concurrent=1)
    results = await asyncio.gather(b.submit(["a"]), b.submit(["b"]), return_exceptions=True)
    assert all(isinstance(r, RuntimeError) for r in results)

    seen = []

    async def run(items):
        seen.extend(items)
        return items

    b = Batcher(run, max_wait_ms=5, max_concurrent=1)
    doomed = asyncio.create_task(b.submit(["gone"]))
    alive = asyncio.create_task(b.submit(["kept"]))
    await asyncio.sleep(0)
    doomed.cancel()
    assert await alive == ["kept"] and seen == ["kept"]  # the cancelled caller's request never reaches the model
    with pytest.raises(asyncio.CancelledError):
        await doomed
