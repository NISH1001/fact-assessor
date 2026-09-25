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
        """Components this step holds (anything with `aload`/`aclose`), for lifecycle management."""
        return [v for v in vars(self).values() if hasattr(v, "aload") or hasattr(v, "aclose")]

    async def aload(self) -> None:
        """Warm every resource held anywhere in this step (nested steps included), each once."""
        for c in _components(self):
            if hasattr(c, "aload"):
                await c.aload()

    async def aclose(self) -> None:
        for c in _components(self):
            if hasattr(c, "aclose"):
                await c.aclose()


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
        async def one(item: Any) -> AsyncIterator[Any]:
            result = await _maybe_await(self.fn(item))
            if result is not None:
                yield result

        return _concurrently(items, one, self.concurrency)


class Filter(Step):
    """Keep items where `pred(item)` is true. `pred` may be sync or async; it sees whatever flows here."""

    def __init__(self, pred: Callable[[Any], Any], concurrency: int | None = None) -> None:
        self.pred = pred
        self.concurrency = concurrency

    def __call__(self, items: AsyncIterator[Any]) -> AsyncIterator[Any]:
        async def one(item: Any) -> AsyncIterator[Any]:
            if await _maybe_await(self.pred(item)):
                yield item
            else:
                report_dropped(item)

        return _concurrently(items, one, self.concurrency)


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


async def _maybe_await(value: Any) -> Any:
    return await value if inspect.isawaitable(value) else value


async def _aclose(items: Any) -> None:
    if hasattr(items, "aclose"):
        try:
            await items.aclose()
        except RuntimeError:  # already closing (cancelled mid-iteration)
            pass


def _components(root: Step) -> list[Any]:
    """Every non-step resource reachable from `root` through `parts()`, each once, in discovery order."""
    seen: set[int] = set()
    found: list[Any] = []
    queue: list[Any] = [root]
    while queue:
        node = queue.pop(0)
        if id(node) in seen:
            continue
        seen.add(id(node))
        if isinstance(node, Step):
            queue.extend(node.parts())
        else:
            found.append(node)
    return found
