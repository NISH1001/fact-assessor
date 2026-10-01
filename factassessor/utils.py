"""Small helpers shared across components: sentence spans, where in a text a piece of it came from, and `cache`."""

from __future__ import annotations

import asyncio
import functools
import inspect
import threading
from collections.abc import Callable
from typing import Any, TypeVar, overload

import re

from factassessor.passages import BM25Index

F = TypeVar("F", bound=Callable[..., Any])
C = TypeVar("C", bound=type)

_SENTENCE = re.compile(r"\S.*?(?:[.!?]+(?=\s|$)|$)", re.S)  # ends at .!? + space, so "7.8" stays whole


def sentences(text: str) -> list[tuple[int, int]]:
    """Character spans of the sentences of `text`, in order."""
    return [(m.start(), m.start() + len(m.group().rstrip())) for m in _SENTENCE.finditer(text)]


def locate(query: str, source: str) -> tuple[int, int]:
    """The character span of the sentence in `source` that best matches `query`: the one sharing the most, and
    rarest, words with it (BM25 over the sentences, so wording every sentence repeats counts for little next to
    the query's own detail). The first sentence when nothing matches. The atomizer uses it to find the sentence a
    claim was made from; it works the same for a quote in a page or a title in a document."""
    spans = sentences(source) or [(0, len(source))]
    [best] = BM25Index([source[s:e] for s, e in spans]).top(query, 1)
    return spans[best]


# --- cache ------------------------------------------------------------------------------------------------------


@overload
def cache(methods: list[str] | tuple[str, ...], *, maxsize: int = ..., ttl: float = ...) -> Callable[[C], C]: ...
@overload
def cache(*, maxsize: int = ..., ttl: float = ...) -> Callable[[F], F]: ...
def cache(methods: list[str] | tuple[str, ...] | None = None, *, maxsize: int = 1024, ttl: float = 600.0) -> Any:
    """Remember results by argument for `ttl` seconds (least recently used out past `maxsize`), in memory.

        @cache(maxsize=2048, ttl=600)            # a method or a function, async or sync
        async def crawl(self, url): ...

        @cache(["crawl"], maxsize=2048, ttl=600) # a class: the named methods

    Methods get one cache per instance; functions one cache in all. Async: concurrent callers with the same
    arguments await one call (all of a text's claims asking for the same paper at once), a cancelled caller doesn't
    cancel it, and an exception or cancellation is not remembered. Sync: a plain lookup under a lock; exceptions not
    remembered. Arguments must be hashable. Misuse fails when the decorator is applied, so at import time."""
    if maxsize < 1:
        raise ValueError(f"cache: maxsize must be >= 1, not {maxsize}")
    if ttl <= 0:
        raise ValueError(f"cache: ttl must be > 0, not {ttl}")
    if methods is not None and (isinstance(methods, str) or not all(isinstance(m, str) for m in methods) or not methods):
        raise TypeError(f"cache: methods must be a non-empty list of method names, not {methods!r}")

    def decorate(target: Any) -> Any:
        if isinstance(target, type):
            if methods is None:
                raise TypeError(f"cache: on a class, name the methods to cache: @cache([...], ...) on {target.__name__}")
            for name in methods:
                if not callable(target.__dict__.get(name)) and not callable(getattr(target, name, None)):
                    raise AttributeError(f"cache: {target.__name__} has no method {name!r}")
                setattr(target, name, _Cached(getattr(target, name), maxsize, ttl))
            return target
        if methods is not None:
            raise TypeError(f"cache([...]) is the class form; on {getattr(target, '__qualname__', target)!r} use @cache(maxsize=..., ttl=...)")
        if not callable(target):
            raise TypeError(f"cache: {target!r} is not a function or a class")
        return _Cached(target, maxsize, ttl)

    return decorate


class _Cached:
    """The cached callable: called directly it uses its own cache; looked up on an instance (a method) it binds that
    instance and uses a cache stored on it."""

    def __init__(self, fn: Callable[..., Any], maxsize: int, ttl: float) -> None:
        from cachetools import TTLCache

        if isinstance(fn, _Cached):  # @cache on an already cached function: keep the first
            fn = fn.fn
        self.fn, self.maxsize, self.ttl = fn, maxsize, ttl
        self.is_async = inspect.iscoroutinefunction(fn)
        self.name = f"_cache_{getattr(fn, '__name__', 'fn')}_{id(self)}"
        self.entries: Any = TTLCache(maxsize=maxsize, ttl=ttl)
        self.lock = threading.Lock()
        self.__cached__ = True
        functools.update_wrapper(self, fn)
        if self.is_async:  # a cached async function called directly is still a coroutine function
            inspect.markcoroutinefunction(self)

    def __set_name__(self, owner: type, name: str) -> None:
        self.name = f"_cache_{name}"

    def __get__(self, instance: Any, owner: type | None = None) -> Any:
        if instance is None:
            return self
        from cachetools import TTLCache

        state = instance.__dict__.get(self.name)
        if state is None:
            state = instance.__dict__[self.name] = (TTLCache(maxsize=self.maxsize, ttl=self.ttl), threading.Lock())
        entries, lock = state
        return self._bind(entries, lock, instance)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self._call(self.entries, self.lock, None, *args, **kwargs)

    def _bind(self, entries: Any, lock: threading.Lock, instance: Any) -> Callable[..., Any]:
        """A real function (async def for async), so `inspect.iscoroutinefunction` still sees an async method: the
        pipeline's `Map` decides whether to await by it."""
        if self.is_async:
            async def bound(*args: Any, **kwargs: Any) -> Any:
                return await self._call(entries, lock, instance, *args, **kwargs)
        else:
            def bound(*args: Any, **kwargs: Any) -> Any:
                return self._call(entries, lock, instance, *args, **kwargs)
        functools.update_wrapper(bound, self.fn)
        bound.__cached__ = True  # type: ignore[attr-defined]
        return bound

    def _call(self, entries: Any, lock: threading.Lock, instance: Any, *args: Any, **kwargs: Any) -> Any:
        key = (args, tuple(sorted(kwargs.items())))
        call_args = (instance, *args) if instance is not None else args
        if self.is_async:
            return self._async(entries, key, call_args, kwargs)
        with lock:
            if key in entries:
                return entries[key]
        result = self.fn(*call_args, **kwargs)
        with lock:
            entries[key] = result
        return result

    async def _async(self, entries: Any, key: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
        task = entries.get(key)
        if task is not None and task.done() and not task.cancelled() and task.exception() is None:
            return task.result()  # finished: no await, so a cached result works from any event loop
        if task is None or task.done():
            # its own task: a claim cancelled at its deadline must not take down a crawl other claims are waiting on
            task = entries[key] = asyncio.ensure_future(self.fn(*args, **kwargs))
            task.add_done_callback(functools.partial(_forget_failure, entries, key))
        return await asyncio.shield(task)


def _forget_failure(entries: Any, key: Any, task: asyncio.Future[Any]) -> None:
    """A failed or cancelled call is not remembered (and its exception counts as retrieved)."""
    if task.cancelled() or task.exception() is not None:
        if entries.get(key) is task:
            entries.pop(key, None)
