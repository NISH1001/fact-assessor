import asyncio

import pytest

from factassessor import LayaRunner


class FakeRouter:
    def __init__(self, fail=False):
        self.calls = []
        self.fail = fail

    def predict_batch(self, requests, batch_size=None):
        self.calls.append(len(requests))
        self.batch_sizes = getattr(self, "batch_sizes", []) + [batch_size]
        self.orders = getattr(self, "orders", []) + [[r["state"] for r in requests]]
        if self.fail:
            raise RuntimeError("MPS out of memory")
        return [{"echo": r["state"]} for r in requests]


def runner(router):
    laya = LayaRunner()
    laya._router = router  # skip loading the real model
    return laya


async def test_concurrent_callers_share_one_forward_pass():
    router = FakeRouter()
    laya = runner(router)
    a, b, c = await asyncio.gather(
        laya.predict_batch([{"state": "a1"}, {"state": "a2"}]),
        laya.predict_batch([{"state": "b1"}]),
        laya.predict_batch([{"state": "c1"}, {"state": "c2"}, {"state": "c3"}]),
    )
    assert router.calls == [6]  # one merged batch
    assert [x["echo"] for x in a] == ["a1", "a2"]  # each caller gets exactly its own results, in order
    assert [x["echo"] for x in b] == ["b1"]
    assert [x["echo"] for x in c] == ["c1", "c2", "c3"]


async def test_callers_after_a_flush_start_a_new_batch():
    router = FakeRouter()
    laya = runner(router)
    await laya.predict_batch([{"state": "first"}])
    await laya.predict_batch([{"state": "second"}])
    assert router.calls == [1, 1]


async def test_failure_reaches_every_caller_in_the_batch():
    laya = runner(FakeRouter(fail=True))
    results = await asyncio.gather(
        laya.predict_batch([{"state": "a"}]), laya.predict_batch([{"state": "b"}]), return_exceptions=True
    )
    assert all(isinstance(r, RuntimeError) for r in results)


async def test_empty_request_list_skips_the_model():
    router = FakeRouter()
    assert await runner(router).predict_batch([]) == []
    assert router.calls == []


async def test_cancelled_caller_does_not_break_the_batch():
    router = FakeRouter()
    laya = runner(router)
    doomed = asyncio.create_task(laya.predict_batch([{"state": "gone"}]))
    alive = asyncio.create_task(laya.predict_batch([{"state": "kept"}]))
    await asyncio.sleep(0)
    doomed.cancel()
    assert [x["echo"] for x in await alive] == ["kept"]
    with pytest.raises(asyncio.CancelledError):
        await doomed


async def test_forward_passes_are_capped_at_max_batch():
    router = FakeRouter()
    laya = runner(router)
    laya.max_batch = 32
    await laya.predict_batch([{"state": f"s{i}"} for i in range(100)])
    assert router.batch_sizes == [32]  # Laya splits the 100 into passes of <= 32 rows


async def test_rows_are_grouped_by_length_and_results_come_back_in_caller_order():
    router = FakeRouter()
    laya = runner(router)
    short, long_ = "a snippet", "a long crawled page passage " * 20
    a, b = await asyncio.gather(
        laya.predict_batch([{"state": long_}, {"state": short}]),
        laya.predict_batch([{"state": short + "!"}, {"state": long_ + "!"}]),
    )
    sent = router.orders[0]
    assert sent == sorted(sent, key=len)  # short rows next to short rows: less padding per pass
    assert [x["echo"] for x in a] == [long_, short]
    assert [x["echo"] for x in b] == [short + "!", long_ + "!"]
