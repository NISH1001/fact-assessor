"""FactAssessor: the ready-made fact-checking pipeline, and the facade that runs any pipeline.

    atomizer >> claim_filter              searcher (per claim)                crawler (per hit)
    LLMAtomizer >> DecisionClaimFilter    SerperSearcher >> not_blocked()     Crawl4AICrawler
                >> Take(n_atoms)                         >> Take(top_k)
                    \\                                  judge: DecisionJudge   policy: WeightedPolicy
                     `-> Verify(searcher, crawler, judge, policy) -> results -> fact score
                                                            (knowledge graph: kg.build(result), on demand)

The default filter and judge share one `LayaRunner` (the local model, loaded once, one batch queue); pass
`claim_filter=DecisionClaimFilter(runner)` / `judge=DecisionJudge(runner, ...)` to put either on another runner.

Everything streams: a claim is verified the moment the atomizer emits it, each page is judged the moment its
crawl lands, and results come out as they settle. `assess` is `stream` read to the end.
"""

from __future__ import annotations

import asyncio
import math
import threading
import time
from collections.abc import AsyncIterator, Iterable
from typing import Any

from factassessor.atomizer import DEFAULT_MODEL as DEFAULT_ATOMIZER_MODEL
from factassessor.atomizer import LLMAtomizer
from factassessor.claim_filters import ClaimFilter, DecisionClaimFilter
from factassessor.crawlers import CascadedCrawler, ContentType, Crawl4AICrawler, HTTPXCrawler, ImpitCrawler, StatusIn
from factassessor.judges import DecisionJudge, Judge
from factassessor.laya import LayaRunner
from factassessor.pipeline import Cache, Map, Step, Take, dropped, once
from factassessor.schema import AtomResult, CheckResult, ClaimFound, ClaimVerified, Done, Event
from factassessor.search import BLOCKED_DOMAINS, SerperSearcher, not_blocked
from factassessor.verify import Policy, Verify, WeightedPolicy
from factassessor.resolvers import ArxivResolver, CompositeResolver, OpenAlexResolver, Resolver

_END = object()
_DEFAULT: Any = object()  # "build the default" (so claim_filter=None can mean "no filter")


class FactAssessor:
    """Pass components to replace any part (`atomizer=`, `claim_filter=`, `searcher=`, `crawler=`, `judge=`,
    `policy=`); the other arguments configure the defaults and are ignored for a component you pass yourself.
    The default filter and judge share one Laya runner (one model, one batch queue)."""

    def __init__(
        self,
        n_atoms: int = 5,
        top_k: int = 5,
        *,
        overfetch: float = 1.0,  # keep twice as many hits as pages; the first readable ones are judged (+0.015 F1, same credits)
        device: str = "auto",
        serper_api_key: str | None = None,
        laya_model: str = "english",  # Laya checkpoint: english | multilingual | typed-decisions
        claim_threshold: float = 0.4,  # min claim_score to check an atom; low on purpose: a dropped real claim is never checked
        early_exit_conf: float = 0.9,  # 2+ passages this sure, none against -> stop gathering evidence
        strong_evidence: float = 0.7,  # a passage counts toward a verdict at or above this prob
        crawl_timeout: float = 2.5,
        max_concurrent_crawls: int = 10,
        max_concurrent_claims: int | None | Any = _DEFAULT,  # default: the judge's `concurrency`; None: no limit
        search_timeout: float = 5.0,
        search_hedge_after: float = 1.2,
        blocked_domains: tuple[str, ...] = BLOCKED_DOMAINS,
        timeout: float = 30.0,  # per claim; a claim still running then is decided on the evidence it has (p90 22s alone)
        atomizer_model: str = DEFAULT_ATOMIZER_MODEL,
        source_query: bool = True,  # the atomizer also writes one search for the text's source document (papers: +0.04 F1)
        passages_per_page: int = 3,  # windows of each page the judge sees (3 vs 1: +0.10 F1 on the paper eval)
        atomizer: Step | None = None,  # an Atomizer, or any chain starting with one (text -> atoms)
        claim_filter: ClaimFilter | Step | None = _DEFAULT,  # None: no filter (e.g. your atomizer chain already filters)
        searcher: Step | None = None,  # a Searcher, or any chain starting with one (query -> hits)
        crawler: Step | None = None,  # a Crawler, or any step url -> page
        resolver: Resolver | None | Any = _DEFAULT,  # default: arXiv + OpenAlex, papers read from their free copies; None: off
        judge: Judge | None = None,
        policy: Policy | None = None,
    ) -> None:
        self.atomizer = atomizer or LLMAtomizer(atomizer_model, source_query=source_query)
        laya = LayaRunner(laya_model, device) if claim_filter is _DEFAULT or judge is None else None  # shared
        self.claim_filter = DecisionClaimFilter(laya, threshold=claim_threshold) if claim_filter is _DEFAULT else claim_filter
        self.atoms = (  # text -> the atoms worth checking
            self.atomizer >> self.claim_filter >> Take(n_atoms) if self.claim_filter else self.atomizer >> Take(n_atoms)
        )
        candidates = math.ceil(top_k * (1 + overfetch))  # hits kept per claim; pages read: top_k
        self.searcher = Cache(  # a query is searched once per 10 minutes: a text's claims share their source query
            searcher
            or SerperSearcher(serper_api_key, num=max(10, candidates), timeout=search_timeout, hedge_after=search_hedge_after,
                              exclude=blocked_domains)  # excluded in the query, so Google fills the slots with usable hits
            >> not_blocked(blocked_domains)  # and dropped after search as the guarantee
            >> Take(candidates)
        )
        self.crawler = crawler or CascadedCrawler(  # honest HTTP, then HTTP that looks like Firefox, then a real browser
            HTTPXCrawler(timeout=crawl_timeout),
            ImpitCrawler(timeout=crawl_timeout),
            Crawl4AICrawler(timeout=crawl_timeout, max_concurrent=max_concurrent_crawls),
            when=~(StatusIn(404, 410) | ContentType("pdf")),  # not for a page that is gone, or a PDF (no browser reads those)
        )
        resolver = CompositeResolver(ArxivResolver(), OpenAlexResolver()) if resolver is _DEFAULT else resolver
        self.judge = judge or DecisionJudge(laya, passages_per_page=passages_per_page)
        self.policy = policy or WeightedPolicy(strong=strong_evidence, early_exit=early_exit_conf)
        claims: dict[str, Any] = {} if max_concurrent_claims is _DEFAULT else {"concurrency": max_concurrent_claims}
        self.verify = Verify(
            self.searcher, self.crawler, self.judge, self.policy, timeout=timeout, resolver=resolver,
            pages_per_claim=top_k * (2 if source_query else 1) if overfetch else None,  # own hits + the source query's
            **claims,
        )
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
                async for result in (self.atoms >> Map(found) >> self.verify)(once(text)):
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
                latency_ms=(time.perf_counter() - start) * 1000,
            )
        )

    async def assess(self, text: str) -> CheckResult:
        """Fact-check `text` and return the full result (`stream` read to the end)."""
        async for event in self.stream(text):
            if isinstance(event, Done):
                return event.result
        raise RuntimeError("stream ended without a result")

    async def assess_many(self, texts: Iterable[str], concurrency: int = 3) -> list[CheckResult]:
        """Fact-check several texts on this assessor, at most `concurrency` at once; results in input order.

        One assessor shares its caches across texts (search results, pages, OpenAlex lookups: the texts' claims often
        hit the same pages). The cap is admission control: every text in flight shares one crawler, so starting all
        of them at once makes each wait for the others (10 texts at once on a laptop: ~60s each, vs ~19s alone)."""
        if concurrency < 1:
            raise ValueError(f"concurrency must be >= 1, not {concurrency}")
        slots = asyncio.Semaphore(concurrency)

        async def one(text: str) -> CheckResult:
            async with slots:
                return await self.assess(text)

        return list(await asyncio.gather(*(one(t) for t in texts)))

    def assess_many_sync(self, texts: Iterable[str], concurrency: int = 3) -> list[CheckResult]:
        """Blocking `assess_many`, on the same background loop as `assess_sync` (call `close()` when done)."""
        return self._run_sync(self.assess_many(list(texts), concurrency))

    async def acheck(self, text: str) -> CheckResult:
        """Alias for `assess`."""
        return await self.assess(text)

    def assess_sync(self, text: str) -> CheckResult:
        """Blocking `assess` for plain scripts (and Jupyter, where a loop is already running).

        Runs on one background event loop owned by this assessor, so the browser, HTTP pool, and Laya stay warm
        across calls. Use either `assess` or `assess_sync` on a given instance, not both: their resources belong
        to different loops. Call `close()` (or use `with FactAssessor() as fa:`) when done.
        """
        return self._run_sync(self.assess(text))

    def _run_sync(self, coro: Any) -> Any:
        if self._loop is None:
            self._loop = asyncio.new_event_loop()
            self._loop_thread = threading.Thread(target=self._loop.run_forever, name="factassessor-loop", daemon=True)
            self._loop_thread.start()
        return asyncio.run_coroutine_threadsafe(coro, self._loop).result()

    # --- lifecycle --------------------------------------------------------------------------------------

    async def aload(self) -> None:
        """Warm up every component (Laya weights, browser, HTTP pool) so the first check is fast."""
        await (self.atoms >> self.verify).aload()

    async def aclose(self) -> None:
        """Close the browser and HTTP pool."""
        await (self.atoms >> self.verify).aclose()

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
