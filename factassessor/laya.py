"""Laya runtime, internal: shared by every Laya component (LayaClaimFilter, LayaJudge). Nobody passes it around.

- The model (a laya `Router`) and the one thread it runs on are loaded once per device, per process, however many
  components or assessors use them.
- Requests are micro-batched per event loop: everything that arrives within `max_wait_ms` (from any claim, page,
  or component) goes out in one `predict_batch`. `laya_runner(device)` returns this loop's runner.

Measured on MPS (docs/design/decisions.md): capping each forward pass at 32 rows costs no speed and holds GPU
memory flat under load (400 pairs: 7.1 GB uncapped vs 2.0 GB); grouping rows by length avoids padding short
snippets up to long passages (mixed pass 946ms vs 714ms as two passes).
"""

from __future__ import annotations

import asyncio
import json
import threading
import weakref
from concurrent.futures import ThreadPoolExecutor
from typing import Any

_routers: dict[str, Any] = {}  # device -> loaded laya.Router
_threads: dict[str, ThreadPoolExecutor] = {}  # device -> the one thread that model runs on
_process_lock = threading.Lock()
_runners: weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, dict[str, LayaRunner]] = weakref.WeakKeyDictionary()


def laya_runner(device: str = "auto") -> LayaRunner:
    """This event loop's runner for `device` (call from async code). Every Laya component shares it."""
    loop = asyncio.get_running_loop()
    per_loop = _runners.setdefault(loop, {})
    if device not in per_loop:
        per_loop[device] = LayaRunner(device)
    return per_loop[device]


class LayaRunner:
    """Micro-batching front end to the process-wide Laya model for one device and one event loop."""

    def __init__(self, device: str = "auto", max_wait_ms: float = 5.0, batch_size: int = 32) -> None:
        self.device = device
        self.max_wait_ms = max_wait_ms
        self.batch_size = batch_size  # rows per forward pass (Laya's own name): bounds GPU memory, free in speed
        self._router: Any = None
        self._lock = asyncio.Lock()
        self._pending: list[tuple[list[dict[str, Any]], asyncio.Future[list[dict[str, Any]]]]] = []
        self._flush: asyncio.Task[None] | None = None

    @property
    def _thread(self) -> ThreadPoolExecutor:
        with _process_lock:
            if self.device not in _threads:
                _threads[self.device] = ThreadPoolExecutor(max_workers=1, thread_name_prefix=f"laya-{self.device}")
            return _threads[self.device]

    async def aload(self) -> None:
        """Load the Router (once per process per device)."""
        async with self._lock:
            if self._router is None:
                self._router = await asyncio.get_running_loop().run_in_executor(self._thread, _load_router, self.device)

    async def agent(self, model: str) -> Any:
        """The loaded Laya Agent for a checkpoint: `.tok` is its tokenizer, `.cfg` its token limits."""
        await self.aload()
        return await asyncio.get_running_loop().run_in_executor(self._thread, self._router.load, model)

    async def predict_batch(self, requests: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """`Router.predict_batch` for {"state", "questions", "model"} requests, micro-batched with every other
        caller's requests on this loop into shared forward passes."""
        if not requests:
            return []
        await self.aload()
        future: asyncio.Future[list[dict[str, Any]]] = asyncio.get_running_loop().create_future()
        self._pending.append((requests, future))
        if self._flush is None:
            self._flush = asyncio.create_task(self._flush_after_wait())
        return await future

    async def _flush_after_wait(self) -> None:
        await asyncio.sleep(self.max_wait_ms / 1000)
        batch, self._pending, self._flush = self._pending, [], None
        merged = [r for requests, _ in batch for r in requests]
        order = sorted(range(len(merged)), key=lambda i: _length(merged[i]))  # short rows with short rows
        try:
            ordered = await asyncio.get_running_loop().run_in_executor(
                self._thread, lambda: self._router.predict_batch([merged[i] for i in order], batch_size=self.batch_size)
            )
        except Exception as exc:
            for _, future in batch:
                if not future.done():
                    future.set_exception(exc)
            return
        results: list[dict[str, Any]] = [{}] * len(merged)
        for position, i in enumerate(order):
            results[i] = ordered[position]
        start = 0
        for requests, future in batch:
            if not future.done():  # the caller may have been cancelled (e.g. a claim's timeout)
                future.set_result(results[start : start + len(requests)])
            start += len(requests)


def _load_router(device: str) -> Any:
    """Runs on the model's thread: load the Router once per device for the whole process."""
    with _process_lock:
        if device in _routers:
            return _routers[device]
    from laya import Router

    router = Router(device=resolve_device(device))
    with _process_lock:
        return _routers.setdefault(device, router)


def _length(request: dict[str, Any]) -> int:
    state = request.get("state")
    return len(state) if isinstance(state, str) else len(json.dumps(state, default=str))


def resolve_device(device: str) -> str:
    """"auto" -> cuda > mps > cpu."""
    if device != "auto":
        return device
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"
