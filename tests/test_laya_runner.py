import asyncio

import pytest

from factassessor.decisions import DecisionRequest, DecisionResponse, DecisionRunner, Question
from factassessor.decisions.laya import LayaRunner

ECHO = {"echo": Question(type="choice", instructions="Which text is `state`?", criteria={"a": "a", "b": "b"})}


def req(state):
    return DecisionRequest(state=state, questions=ECHO)


class FakeRouter:
    """Answers every row with its own state as the `choice`, so routing back to callers is checked."""

    def __init__(self, fail=False):
        self.calls = []
        self.batch_sizes = []
        self.orders = []
        self.fail = fail

    def predict_batch(self, requests, batch_size=None):
        self.calls.append(len(requests))
        self.batch_sizes.append(batch_size)
        self.orders.append([r["state"] for r in requests])
        if self.fail:
            raise RuntimeError("MPS out of memory")
        return [{"answers": {"echo": {"type": "choice", "choice": str(r["state"]), "probabilities": {}, "confidence": 1.0}}} for r in requests]


def runner(router, **kwargs):
    laya = LayaRunner(**kwargs)
    laya._router = router  # skip loading the real model
    return laya


def echoes(responses):
    return [r.answers["echo"].choice for r in responses]


async def test_concurrent_callers_share_one_forward_pass_and_get_typed_responses():
    router = FakeRouter()
    laya = runner(router)
    a, b, c = await asyncio.gather(
        laya.predict([req("a1"), req("a2")]), laya.predict([req("b1")]), laya.predict([req("c1"), req("c2"), req("c3")])
    )
    assert router.calls == [6]  # one merged batch
    assert echoes(a) == ["a1", "a2"] and echoes(b) == ["b1"] and echoes(c) == ["c1", "c2", "c3"]  # each its own, in order
    assert all(isinstance(r, DecisionResponse) for r in a) and isinstance(laya, DecisionRunner)


async def test_rows_are_layas_wire_shape_with_the_runners_checkpoint():
    class Recording(FakeRouter):
        def predict_batch(self, requests, batch_size=None):
            self.rows = requests
            return super().predict_batch(requests, batch_size)

    router = Recording()
    await runner(router, model="multilingual").predict([DecisionRequest(state={"claim": "x"}, questions=ECHO)])
    assert router.rows == [{  # the state as given, the questions as plain dicts, the runner's checkpoint
        "state": {"claim": "x"},
        "questions": {"echo": {"type": "choice", "instructions": "Which text is `state`?", "criteria": {"a": "a", "b": "b"}}},
        "model": "multilingual",
    }]


async def test_callers_after_a_flush_start_a_new_batch():
    router = FakeRouter()
    laya = runner(router)
    await laya.predict([req("first")])
    await laya.predict([req("second")])
    assert router.calls == [1, 1]


async def test_requests_arriving_during_a_pass_form_one_next_batch():
    # pages land one by one while the model is busy: they must merge into the next pass, not one pass each
    import time

    class Slow(FakeRouter):
        def predict_batch(self, requests, batch_size=None):
            time.sleep(0.05)  # a forward pass on the model's thread
            return super().predict_batch(requests, batch_size)

    router = Slow()
    laya = runner(router)
    first = asyncio.create_task(laya.predict([req("first")]))
    await asyncio.sleep(0.02)  # the first pass is running
    later = []
    for i in range(20):  # trickle in over ~40ms
        later.append(asyncio.create_task(laya.predict([req(f"r{i}")])))
        await asyncio.sleep(0.002)
    await asyncio.gather(first, *later)
    assert router.calls[0] == 1 and sum(router.calls) == 21 and len(router.calls) <= 3


async def test_failure_reaches_every_caller_in_the_batch():
    laya = runner(FakeRouter(fail=True))
    results = await asyncio.gather(laya.predict([req("a")]), laya.predict([req("b")]), return_exceptions=True)
    assert all(isinstance(r, RuntimeError) for r in results)


async def test_empty_request_list_skips_the_model():
    router = FakeRouter()
    assert await runner(router).predict([]) == []
    assert router.calls == []


async def test_cancelled_caller_does_not_break_the_batch():
    router = FakeRouter()
    laya = runner(router)
    doomed = asyncio.create_task(laya.predict([req("gone")]))
    alive = asyncio.create_task(laya.predict([req("kept")]))
    await asyncio.sleep(0)
    doomed.cancel()
    assert echoes(await alive) == ["kept"]
    with pytest.raises(asyncio.CancelledError):
        await doomed


async def test_forward_passes_are_capped_at_batch_size():
    router = FakeRouter()
    laya = runner(router, batch_size=32)
    await laya.predict([req(f"s{i}") for i in range(100)])
    assert router.batch_sizes == [32]  # Laya splits the 100 into passes of <= 32 rows


async def test_rows_are_grouped_by_length_and_results_come_back_in_caller_order():
    router = FakeRouter()
    laya = runner(router)
    short, long_ = "a snippet", "a long crawled page passage " * 20
    a, b = await asyncio.gather(
        laya.predict([req(long_), req(short)]),
        laya.predict([req(short + "!"), req(long_ + "!")]),
    )
    sent = router.orders[0]
    assert sent == sorted(sent, key=len)  # short rows next to short rows: less padding per pass
    assert echoes(a) == [long_, short]
    assert echoes(b) == [short + "!", long_ + "!"]
