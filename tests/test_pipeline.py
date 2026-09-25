import asyncio
import time

import pytest

from factassessor.pipeline import Chain, Filter, FlatMap, Map, Step, Take, collect, once


async def items(*xs, delay=0.0):
    for x in xs:
        if delay:
            await asyncio.sleep(delay)
        yield x


async def test_map_runs_items_concurrently_and_emits_in_completion_order():
    async def slow(x):
        await asyncio.sleep(x / 100)
        return x * 10

    start = time.perf_counter()
    out = await collect(Map(slow)(items(5, 1, 3)))
    assert time.perf_counter() - start < 0.09  # ~0.05s in parallel, not 0.09s in sequence
    assert out == [10, 30, 50]  # fastest first


async def test_map_drops_none_and_accepts_sync_functions():
    assert await collect(Map(lambda x: x if x % 2 else None)(items(1, 2, 3))) == [1, 3]


async def test_map_respects_its_concurrency_limit():
    running, peak = 0, 0

    async def work(x):
        nonlocal running, peak
        running += 1
        peak = max(peak, running)
        await asyncio.sleep(0.01)
        running -= 1
        return x

    assert sorted(await collect(Map(work, concurrency=2)(items(*range(8))))) == list(range(8))
    assert peak == 2


async def test_flatmap_fans_out():
    async def explode(x):
        for i in range(x):
            yield f"{x}.{i}"

    assert sorted(await collect(FlatMap(explode)(items(1, 2)))) == ["1.0", "2.0", "2.1"]


async def test_filter_takes_sync_or_async_predicates():
    async def is_even(x):
        return x % 2 == 0

    assert await collect(Filter(lambda x: x > 1)(items(1, 2, 3))) == [2, 3]
    assert sorted(await collect(Filter(is_even)(items(1, 2, 3, 4)))) == [2, 4]


async def test_rshift_chains_steps_and_chains_are_steps():
    double = Map(lambda x: x * 2)
    pipeline = double >> Filter(lambda x: x > 2) >> Map(lambda x: x + 1)
    assert isinstance(pipeline, Chain) and isinstance(pipeline, Step)
    assert sorted(await collect(pipeline(items(1, 2, 3)))) == [5, 7]
    assert sorted(await collect((pipeline >> Take(1))(items(1, 2, 3)))) in ([5], [7])


async def test_results_flow_downstream_before_upstream_finishes():
    seen_at = []
    start = time.perf_counter()

    async def record(x):
        seen_at.append(time.perf_counter() - start)
        return x

    await collect((Map(lambda x: x) >> Map(record))(items(1, 2, 3, delay=0.05)))
    assert seen_at[0] < 0.1  # the first item was processed while the source was still producing


async def test_take_stops_early_and_cancels_unfinished_upstream_work():
    started, cancelled, source_closed = [], [], []

    async def endless():
        try:
            i = 0
            while True:
                yield i
                i += 1
        finally:
            source_closed.append(True)

    async def work(x):
        started.append(x)
        try:
            await asyncio.sleep(0 if x < 3 else 10)
        except asyncio.CancelledError:
            cancelled.append(x)
            raise
        return x

    out = await collect((Map(work, concurrency=5) >> Take(3))(endless()))
    assert sorted(out) == [0, 1, 2]
    assert source_closed == [True]
    assert cancelled and all(x >= 3 for x in cancelled)


async def test_breaking_out_early_cancels_in_flight_work():
    cancelled = []

    async def work(x):
        try:
            await asyncio.sleep(0 if x == 0 else 10)
        except asyncio.CancelledError:
            cancelled.append(x)
            raise
        return x

    stream = Map(work)(items(0, 1, 2))
    async for first in stream:
        break
    await stream.aclose()
    assert first == 0 and sorted(cancelled) == [1, 2]


async def test_errors_reach_the_consumer():
    def boom(x):
        raise ValueError(f"bad {x}")

    with pytest.raises(ValueError, match="bad 1"):
        await collect(Map(boom)(items(1)))


async def test_once_and_collect():
    assert await collect(once("text")) == ["text"]


async def test_lifecycle_reaches_every_component_once():
    events = []

    class Resource:
        async def aload(self):
            events.append("load")

        async def aclose(self):
            events.append("close")

    class Uses(Step):  # a component step holding a resource, like a searcher holding its HTTP client
        def __init__(self, resource):
            self.resource = resource

        def __call__(self, items):
            return items

    shared = Resource()
    pipeline = Map(lambda x: x) >> Uses(shared) >> Filter(lambda x: True) >> Uses(shared)
    await pipeline.aload()
    await pipeline.aclose()
    assert events == ["load", "close"]


async def test_sync_steps_keep_input_order():
    ranked = [f"hit{i}" for i in range(20)]
    assert await collect((Filter(lambda h: h != "hit3") >> Map(str.upper))(items(*ranked))) == [
        h.upper() for h in ranked if h != "hit3"
    ]


async def test_steps_start_and_stop_their_own_resources_once():
    events = []

    class Browser(Step):
        async def start(self):
            events.append("start")

        async def stop(self):
            events.append("stop")

        def __call__(self, items):
            return items

    browser = Browser()
    pipeline = browser >> Map(lambda x: x) >> browser
    await pipeline.aload()
    await pipeline.aclose()
    assert events == ["start", "stop"]


# --- conditions: &, |, ~ -------------------------------------------------------------------------------

from factassessor.pipeline import Pred, Scan, TakeUntil, last  # noqa: E402


async def test_conditions_combine_with_and_or_not():
    even, big = Pred(lambda n: n % 2 == 0), Pred(lambda n: n > 5)
    assert await collect(Filter(even & big)(items(*range(10)))) == [6, 8]
    assert await collect(Filter(even | big)(items(*range(10)))) == [0, 2, 4, 6, 7, 8, 9]
    assert await collect(Filter(~even)(items(*range(5)))) == [1, 3]
    assert await collect(Filter(even & ~big)(items(*range(10)))) == [0, 2, 4]


async def test_conditions_mix_sync_and_async_and_short_circuit():
    calls = []

    async def slow_check(n):
        calls.append(n)
        return n > 2

    cond = Pred(lambda n: n % 2 == 0) & Pred(slow_check)
    assert sorted(await collect(Filter(cond)(items(1, 2, 3, 4)))) == [4]
    assert sorted(calls) == [2, 4]  # odd numbers never reached the async check
    assert sorted(await collect(Filter(Pred(slow_check) | (lambda n: n == 0))(items(0, 1, 3)))) == [0, 3]


# --- bare functions and conditions in >> ---------------------------------------------------------------


async def test_bare_functions_in_a_chain_act_as_map_and_conditions_as_filter():
    pipeline = Map(lambda n: n + 1) >> (lambda n: n * 10) >> Pred(lambda n: n > 20) >> Take(2)
    assert await collect(pipeline(items(0, 1, 2, 3))) == [30, 40]


async def test_a_chain_can_start_with_a_function_or_condition():
    assert await collect(((lambda n: n * 2) >> Take(2))(items(1, 2, 3))) == [2, 4]
    assert await collect((Pred(lambda n: n > 1) >> Map(str))(items(1, 2, 3))) == ["2", "3"]


# --- Scan, TakeUntil, last -----------------------------------------------------------------------------


async def test_scan_yields_running_totals():
    assert await collect(Scan(lambda total, n: total + n, 0)(items(1, 2, 3))) == [1, 3, 6]


async def test_take_until_includes_the_item_that_satisfied_it_and_cancels_upstream():
    cancelled = []

    async def work(n):
        try:
            await asyncio.sleep(0.01 * n if n < 4 else 10)
        except asyncio.CancelledError:
            cancelled.append(n)
            raise
        return n

    running = Map(work) >> Scan(lambda total, n: total + n, 0) >> TakeUntil(lambda total: total >= 3)
    assert await collect(running(items(1, 2, 3, 4, 5))) == [1, 3]  # stops at the running total that reached 3
    assert sorted(cancelled) == [3, 4, 5]  # everything still running when it stopped (3 needed 0.03s; we stopped at 0.02s)


async def test_last_returns_the_final_item_or_a_default():
    assert await last(items(1, 2, 3)) == 3
    assert await last(items(), default="none") == "none"


def test_sync_conditions_evaluate_each_side_at_most_once():
    calls = []

    def left(n):
        calls.append("left")
        return True

    (Pred(left) | Pred(lambda n: True))(1)
    (Pred(left) & Pred(lambda n: True))(1)
    assert calls == ["left", "left"]  # once per combined check, not twice
