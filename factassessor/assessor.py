"""FactAssessor: the ready-made fact-checking pipeline, and the facade that runs any pipeline.

    atomizer                      searcher (per claim)             crawler (per hit)
    Atomizer >> LayaCheckworthy   Serper >> Filter(not_blocked)    Crawl4ai
             >> Take(n_atoms)            >> Take(top_k)
                    \\                                  judge: LayaJudge   policy: WeightedPolicy
                     `-> Verify(searcher, crawler, judge, policy) -> results -> fact score + graph

Everything streams: a claim is verified the moment the atomizer emits it, each page is judged the moment its
crawl lands, and results come out as they settle. `assess` is `stream` read to the end.
"""

from __future__ import annotations

import asyncio
import threading
import time
from collections.abc import AsyncIterator
from typing import Any

from factassessor.aggregate import build_graph, fact_score
from factassessor.atom_filter import LayaCheckworthy
from factassessor.atomizer import DEFAULT_MODEL as DEFAULT_ATOMIZER_MODEL
from factassessor.atomizer import Atomizer
from factassessor.crawl import Crawl4ai
from factassessor.evidence_judge import LayaJudge
from factassessor.laya import LayaRunner
from factassessor.pipeline import Filter, Map, Step, Take, dropped, once
from factassessor.schema import AtomResult, CheckResult, ClaimFound, ClaimVerified, Done, Event
from factassessor.search import BLOCKED_DOMAINS, Serper, not_blocked
from factassessor.verify import Verify, WeightedPolicy

_END = object()


class FactAssessor:
    """Pass components to replace any part (`atomizer=`, `searcher=`, `crawler=`, `judge=`, `policy=`, `laya=`);
    the other arguments configure the defaults and are ignored for a component you pass yourself."""

    def __init__(
        self,
        n_atoms: int = 5,
        top_k: int = 5,
        *,
        device: str = "auto",
        serper_api_key: str | None = None,
        laya_model: str = "english",  # Laya checkpoint: english | multilingual | typed-decisions
        checkworthy_threshold: float = 0.4,  # P(factual_claim); low on purpose: a dropped real claim is never checked
        early_exit_conf: float = 0.9,  # 2+ passages this sure, none against -> stop gathering evidence
        strong_evidence: float = 0.7,  # a passage counts toward a verdict at or above this prob
        crawl_timeout: float = 2.5,
        max_concurrent_crawls: int = 10,
        search_timeout: float = 5.0,
        search_hedge_after: float = 1.2,
        blocked_domains: tuple[str, ...] = BLOCKED_DOMAINS,
        timeout: float = 15.0,  # per claim; a claim still running then comes back unverified
        atomizer_model: str = DEFAULT_ATOMIZER_MODEL,
        laya: LayaRunner | None = None,
        atomizer: Step | None = None,
        searcher: Step | None = None,
        crawler: Step | None = None,
        judge: Any = None,
        policy: Any = None,
    ) -> None:
        self.laya = laya or LayaRunner(device)  # one model shared by the filter and the judge
        self.laya_model = laya_model
        self.atomizer = atomizer or (
            Atomizer(atomizer_model) >> LayaCheckworthy(self.laya, checkworthy_threshold, laya_model) >> Take(n_atoms)
        )
        self.searcher = searcher or (
            Serper(serper_api_key, num=2 * top_k, timeout=search_timeout, hedge_after=search_hedge_after)
            >> Filter(not_blocked(blocked_domains))  # over-fetched above so blocked hits don't leave us short
            >> Take(top_k)
        )
        self.crawler = crawler or Crawl4ai(timeout=crawl_timeout, max_concurrent=max_concurrent_crawls)
        self.judge = judge or LayaJudge(self.laya, laya_model)
        self.policy = policy or WeightedPolicy(strong=strong_evidence, early_exit=early_exit_conf)
        self.verify = Verify(self.searcher, self.crawler, self.judge, self.policy, timeout=timeout)
        self._loop: asyncio.AbstractEventLoop | None = None  # background loop behind assess_sync()
        self._loop_thread: threading.Thread | None = None

    # --- running ----------------------------------------------------------------------------------------

    async def stream(self, text: str) -> AsyncIterator[Event]:
        """`ClaimFound` as each claim is found, `ClaimVerified` as each one settles, then `Done`."""
        start = time.perf_counter()
        events: asyncio.Queue[Any] = asyncio.Queue()
        results: list[AtomResult] = []
        skipped: list[Any] = []

        def found(atom: Any) -> Any:
            events.put_nowait(ClaimFound(atom=atom))
            return atom

        async def run() -> None:
            dropped.set(skipped)  # filters report the atoms they drop here (this task's context only)
            try:
                async for result in (self.atomizer >> Map(found) >> self.verify)(once(text)):
                    results.append(result)
                    events.put_nowait(ClaimVerified(result=result))
            finally:
                events.put_nowait(_END)

        task = asyncio.create_task(run())
        try:
            while (event := await events.get()) is not _END:
                yield event
            await task  # surface a pipeline error, if any
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

        results.sort(key=lambda r: r.atom.id)
        yield Done(
            result=CheckResult(
                text=text,
                atoms=results,
                skipped=sorted(skipped, key=lambda a: a.id),
                fact_score=fact_score(results),
                graph=build_graph(results, strong=getattr(self.policy, "strong", 0.7)),
                latency_ms=(time.perf_counter() - start) * 1000,
            )
        )

    async def assess(self, text: str) -> CheckResult:
        """Fact-check `text` and return the full result (`stream` read to the end)."""
        async for event in self.stream(text):
            if isinstance(event, Done):
                return event.result
        raise RuntimeError("stream ended without a result")

    async def acheck(self, text: str) -> CheckResult:
        """Alias for `assess`."""
        return await self.assess(text)

    def assess_sync(self, text: str) -> CheckResult:
        """Blocking `assess` for plain scripts (and Jupyter, where a loop is already running).

        Runs on one background event loop owned by this assessor, so the browser, HTTP pool, and Laya stay warm
        across calls. Use either `assess` or `assess_sync` on a given instance, not both: their resources belong
        to different loops. Call `close()` (or use `with FactAssessor() as fa:`) when done.
        """
        if self._loop is None:
            self._loop = asyncio.new_event_loop()
            self._loop_thread = threading.Thread(target=self._loop.run_forever, name="factassessor-loop", daemon=True)
            self._loop_thread.start()
        return asyncio.run_coroutine_threadsafe(self.assess(text), self._loop).result()

    # --- lifecycle --------------------------------------------------------------------------------------

    async def aload(self) -> None:
        """Warm up every component (Laya weights, browser, HTTP pool) so the first check is fast."""
        await (self.atomizer >> self.verify).aload()

    async def aclose(self) -> None:
        """Close the browser and HTTP pool."""
        await (self.atomizer >> self.verify).aclose()

    async def __aenter__(self) -> FactAssessor:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()

    def close(self) -> None:
        """Blocking `aclose` for `assess_sync` users: closes the browser and HTTP pool, stops the background loop."""
        if self._loop is None:
            return
        asyncio.run_coroutine_threadsafe(self.aclose(), self._loop).result()
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._loop_thread.join()
        self._loop.close()
        self._loop = self._loop_thread = None

    def __enter__(self) -> FactAssessor:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
