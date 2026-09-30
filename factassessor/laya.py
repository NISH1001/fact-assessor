"""LayaRunner: in-process Laya on the local GPU, the default `DecisionRunner`.

- The model (a laya `Router`) and the one thread it runs on are loaded once per device, per process, however many
  runners or assessors use them. `model` picks the checkpoint: english (421M, 512 tokens), multilingual (322M,
  1,024 tokens, ~2.2x faster), typed-decisions.
- Requests are micro-batched (`decisions.Batcher`): everything that arrives within `max_wait_ms` from any claim,
  page or component goes out in one `predict_batch`, and everything arriving during a pass forms the next one.

Measured on MPS (docs/design/decisions.md): capping each forward pass at 32 rows costs no speed and holds GPU
memory flat under load (400 pairs: 7.1 GB uncapped vs 2.0 GB); grouping rows by length avoids padding short
snippets up to long passages (mixed pass 946ms vs 714ms as two passes).
"""

from __future__ import annotations

import asyncio
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from factassessor.decisions import Batcher, DecisionRequest, DecisionResponse, DecisionRunner

_routers: dict[str, Any] = {}  # device -> loaded laya.Router
_threads: dict[str, ThreadPoolExecutor] = {}  # device -> the one thread that model runs on
_process_lock = threading.Lock()


class LayaRunner(DecisionRunner):
    """Micro-batching front end to the process-wide Laya model for one device."""

    def __init__(self, model: str = "english", device: str = "auto", batch_size: int = 32, max_wait_ms: float = 5.0) -> None:
        self.model = model  # Laya checkpoint: english | multilingual | typed-decisions
        self.device = device
        self.batch_size = batch_size  # rows per forward pass (Laya's own name): bounds GPU memory, free in speed
        self._router: Any = None
        self._lock = asyncio.Lock()
        self._batch = Batcher(self._run, max_wait_ms, max_concurrent=1)  # one GPU: passes run one after another

    @property
    def _thread(self) -> ThreadPoolExecutor:
        with _process_lock:
            if self.device not in _threads:
                _threads[self.device] = ThreadPoolExecutor(max_workers=1, thread_name_prefix=f"laya-{self.device}")
            return _threads[self.device]

    async def aload(self) -> None:
        """Load the Router (once per process per device) and this checkpoint's weights."""
        async with self._lock:
            if self._router is None:
                loop = asyncio.get_running_loop()
                self._router = await loop.run_in_executor(self._thread, _load_router, self.device)
                await loop.run_in_executor(self._thread, self._router.load, self.model)

    async def predict(self, requests: list[DecisionRequest]) -> list[DecisionResponse]:
        if not requests:
            return []
        await self.aload()
        return await self._batch.submit(requests)

    async def _run(self, requests: list[DecisionRequest]) -> list[DecisionResponse]:
        """One `Router.predict_batch` on the model's thread (Laya cuts it into passes of `batch_size` rows)."""
        rows = [{"state": r.state, "questions": {k: q.wire() for k, q in r.questions.items()}, "model": r.model or self.model} for r in requests]
        order = sorted(range(len(rows)), key=lambda i: _length(rows[i]))  # short rows with short rows
        ordered = await asyncio.get_running_loop().run_in_executor(
            self._thread, lambda: self._router.predict_batch([rows[i] for i in order], batch_size=self.batch_size)
        )
        results: list[DecisionResponse] = [None] * len(rows)  # type: ignore[list-item]
        for position, i in enumerate(order):
            results[i] = DecisionResponse.model_validate(ordered[position])
        return results


def _load_router(device: str) -> Any:
    """Runs on the model's thread: load the Router once per device for the whole process."""
    with _process_lock:
        if device in _routers:
            return _routers[device]
    from laya import Router

    router = Router(device=resolve_device(device))
    with _process_lock:
        return _routers.setdefault(device, router)


def _length(row: dict[str, Any]) -> int:
    state = row["state"]
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
