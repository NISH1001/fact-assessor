"""Composable async stream steps: `a >> b >> c`.

A step turns an async stream of items into an async stream of items. Per-item steps (`Map`, `FlatMap`,
`Filter`) process items concurrently and pass each result on the moment it's ready, so a chain streams:
item 1 can be three steps downstream while item 5 is still in the first. Nothing waits for a whole stage.

Closing a stream (a consumer breaking out, `Take(n)` reaching n, a timeout) cancels the unfinished work in
every step feeding it.
"""

from __future__ import annotations

import asyncio
import contextvars
import inspect
from collections.abc import AsyncIterable, AsyncIterator, Callable
from typing import Any

_END = object()

# Atoms a Filter dropped during the current run; FactAssessor reads it to report `skipped`.
dropped: contextvars.ContextVar[list[Any] | None] = contextvars.ContextVar("dropped", default=None)


class Step:
    """Base class: implement `__call__(items) -> AsyncIterator`. Compose with `>>`."""

    def __call__(self, items: AsyncIterator[Any]) -> AsyncIterator[Any]:
        raise NotImplementedError

    def __rshift__(self, other: Step) -> Chain:
        return Chain(self, other)

    def parts(self) -> list[Any]:
        """Sub-steps and resources (anything with `aload`/`aclose`) this step holds, for lifecycle management."""
        return [v for v in vars(self).values() if isinstance(v, Step) or hasattr(v, "aload") or hasattr(v, "aclose")]

    async def start(self) -> None:
        """Acquire this step's own resources (browser, HTTP pool, model). Default: none."""

    async def stop(self) -> None:
        """Release this step's own resources. Default: none."""

    async def aload(self) -> None:
        """Warm everything reachable from this step (nested steps and shared resources), each once."""
        for node in _nodes(self):
            await (node.start() if isinstance(node, Step) else _call(node, "aload"))

    async def aclose(self) -> None:
        for node in _nodes(self):
            await (node.stop() if isinstance(node, Step) else _call(node, "aclose"))


class Chain(Step):
    def __init__(self, *steps: Step) -> None:
        self.steps: list[Step] = [s for step in steps for s in (step.steps if isinstance(step, Chain) else [step])]

    def __call__(self, items: AsyncIterator[Any]) -> AsyncIterator[Any]:
        for step in self.steps:
            items = step(items)
        return items

    def parts(self) -> list[Any]:
        return list(self.steps)


class FlatMap(Step):
    """Each item -> 0..n items. `fn(item)` returns an async iterable."""

    def __init__(self, fn: Callable[[Any], AsyncIterable[Any]], concurrency: int | None = None) -> None:
        self.fn = fn
        self.concurrency = concurrency

    def __call__(self, items: AsyncIterator[Any]) -> AsyncIterator[Any]:
        return _concurrently(items, self.fn, self.concurrency)


class Map(Step):
    """Each item -> one item, or dropped if `fn` returns None. `fn` may be sync or async."""

    def __init__(self, fn: Callable[[Any], Any], concurrency: int | None = None) -> None:
        self.fn = fn
        self.concurrency = concurrency

    def __call__(self, items: AsyncIterator[Any]) -> AsyncIterator[Any]:
        if not _is_async(self.fn):  # sync: nothing to overlap, so keep input order and skip the tasks
            return _in_order(items, lambda item: [] if (r := self.fn(item)) is None else [r])

        async def one(item: Any) -> AsyncIterator[Any]:
            result = await self.fn(item)
            if result is not None:
                yield result

        return _concurrently(items, one, self.concurrency)


class Filter(Step):
    """Keep items where `pred(item)` is true. `pred` may be sync or async; it sees whatever flows here."""

    def __init__(self, pred: Callable[[Any], Any], concurrency: int | None = None) -> None:
        self.pred = pred
        self.concurrency = concurrency

    def __call__(self, items: AsyncIterator[Any]) -> AsyncIterator[Any]:
        if not _is_async(self.pred):  # sync: keeps input order (e.g. search rank)
            return _in_order(items, self._keep)

        async def one(item: Any) -> AsyncIterator[Any]:
            for kept in self._keep(item, await self.pred(item)):
                yield kept

        return _concurrently(items, one, self.concurrency)

    def _keep(self, item: Any, verdict: Any = None) -> list[Any]:
        if (self.pred(item) if verdict is None else verdict):
            return [item]
        report_dropped(item)
        return []


class Take(Step):
    """The first n items; then closes the stream upstream (cancelling what's still running)."""

    def __init__(self, n: int) -> None:
        self.n = n

    async def __call__(self, items: AsyncIterator[Any]) -> AsyncIterator[Any]:
        try:
            if self.n <= 0:
                return
            count = 0
            async for item in items:
                yield item
                count += 1
                if count >= self.n:
                    return
        finally:
            await _aclose(items)


async def once(item: Any) -> AsyncIterator[Any]:
    """A one-item stream: the usual source for a chain (`chain(once(text))`)."""
    yield item


async def collect(stream: AsyncIterator[Any]) -> list[Any]:
    return [x async for x in stream]


def report_dropped(item: Any) -> None:
    from factassessor.schema import Atom

    if isinstance(item, Atom) and (sink := dropped.get()) is not None:
        sink.append(item)


async def _concurrently(
    items: AsyncIterator[Any], run: Callable[[Any], AsyncIterable[Any]], limit: int | None
) -> AsyncIterator[Any]:
    """Run `run(item)` for every item concurrently (at most `limit` at once); yield outputs as they appear."""
    queue: asyncio.Queue[Any] = asyncio.Queue()
    slots = asyncio.Semaphore(limit) if limit else None
    tasks: set[asyncio.Task[None]] = set()

    async def work(item: Any) -> None:
        try:
            async for out in run(item):
                queue.put_nowait((None, out))
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            queue.put_nowait((exc, None))
        finally:
            if slots:
                slots.release()

    async def feed() -> None:
        try:
            async for item in items:
                if slots:
                    await slots.acquire()  # backpressure: don't pull more input than we can work on
                task = asyncio.create_task(work(item))
                tasks.add(task)
                task.add_done_callback(tasks.discard)
            while tasks:
                await asyncio.wait(set(tasks))
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            queue.put_nowait((exc, None))
        finally:
            queue.put_nowait(_END)

    feeder = asyncio.create_task(feed())
    try:
        while (message := await queue.get()) is not _END:
            error, out = message
            if error is not None:
                raise error
            yield out
    finally:
        feeder.cancel()
        for task in list(tasks):
            task.cancel()
        await asyncio.gather(feeder, *tasks, return_exceptions=True)
        await _aclose(items)


async def _in_order(items: AsyncIterator[Any], fn: Callable[[Any], list[Any]]) -> AsyncIterator[Any]:
    try:
        async for item in items:
            for out in fn(item):
                yield out
    finally:
        await _aclose(items)


def _is_async(fn: Any) -> bool:
    return inspect.iscoroutinefunction(fn) or inspect.iscoroutinefunction(getattr(fn, "__call__", None))


async def _call(obj: Any, method: str) -> None:
    if hasattr(obj, method):
        await getattr(obj, method)()


async def _aclose(items: Any) -> None:
    if hasattr(items, "aclose"):
        try:
            await items.aclose()
        except RuntimeError:  # already closing (cancelled mid-iteration)
            pass


def _nodes(root: Step) -> list[Any]:
    """Every step and resource reachable from `root` through `parts()`, each once, in discovery order."""
    seen: set[int] = set()
    found: list[Any] = []
    queue: list[Any] = [root]
    while queue:
        node = queue.pop(0)
        if id(node) in seen:
            continue
        seen.add(id(node))
        found.append(node)
        if isinstance(node, Step):
            queue.extend(node.parts())
    return found
