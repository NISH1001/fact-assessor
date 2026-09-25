"""One shared Laya model for every step that needs it (atom filter, evidence judge).

Jev-style parallelism: every request that arrives within `max_wait_ms` -- from any atom,
any page, any step -- is merged into a single `Router.predict_batch` forward pass. The
Router lives on one dedicated thread (MPS is happiest with one model on one thread), loads
on first use, and never blocks the event loop. While one batch runs, the next one fills.
"""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Any


class LayaRunner:
    def __init__(self, device: str = "auto", max_wait_ms: float = 5.0) -> None:
        self.device = device
        self.max_wait_ms = max_wait_ms
        self._router: Any = None
        self._lock = asyncio.Lock()
        self._thread = ThreadPoolExecutor(max_workers=1, thread_name_prefix="laya")
        self._pending: list[tuple[list[dict[str, Any]], asyncio.Future[list[dict[str, Any]]]]] = []
        self._flush: asyncio.Task[None] | None = None

    async def aload(self) -> None:
        """Load the model now instead of on the first request (~2s on MPS)."""
        async with self._lock:
            if self._router is None:
                self._router = await asyncio.get_running_loop().run_in_executor(self._thread, self._load)

    async def predict_batch(self, requests: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """`Router.predict_batch` for {"state", "questions", "model"} requests, micro-batched with
        every other caller's requests into one forward pass."""
        if not requests:
            return []
        await self.aload()
        future: asyncio.Future[list[dict[str, Any]]] = asyncio.get_running_loop().create_future()
        self._pending.append((requests, future))
        if self._flush is None:
            self._flush = asyncio.create_task(self._flush_after_wait())
        return await future

    async def agent(self, model: str) -> Any:
        """The loaded Laya Agent for a checkpoint: `.tok` is its tokenizer, `.cfg` its token limits."""
        await self.aload()
        return await asyncio.get_running_loop().run_in_executor(self._thread, self._router.load, model)

    async def _flush_after_wait(self) -> None:
        await asyncio.sleep(self.max_wait_ms / 1000)
        batch, self._pending, self._flush = self._pending, [], None
        merged = [r for requests, _ in batch for r in requests]
        try:
            results = await asyncio.get_running_loop().run_in_executor(self._thread, self._router.predict_batch, merged)
        except Exception as exc:
            for _, future in batch:
                if not future.done():
                    future.set_exception(exc)
            return
        start = 0
        for requests, future in batch:
            if not future.done():  # the caller may have been cancelled (e.g. atom timeout)
                future.set_result(results[start : start + len(requests)])
            start += len(requests)

    def _load(self) -> Any:
        from laya import Router

        return Router(device=resolve_device(self.device))


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
