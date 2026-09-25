"""FactAssessor: text -> atoms (decontextualized) -> filter -> search -> crawl -> judge -> graph -> score.

Stages run in order; inside a stage, atoms run concurrently, and each page is judged as soon as its
own crawl finishes. Laya does both the filter and the judge calls, micro-batched across all atoms.
"""

from __future__ import annotations

import asyncio
import os
import threading
import time
from typing import Any
from urllib.parse import urlparse

import httpx

from factassessor.atom_filter import AtomFilter
from factassessor.atomizer import DEFAULT_MODEL as DEFAULT_ATOMIZER_MODEL
from factassessor.atomizer import Atomizer
from factassessor.evidence_judge import LayaJudge
from factassessor.laya import LayaRunner
from factassessor.passages import clean_text
from factassessor.schema import Atom, AtomResult, CheckResult, Evidence, Verdict

SERPER_URL = "https://google.serper.dev/search"

# Social media pages are mostly reposts/comments and often crawl badly. Twitter/X and LinkedIn are kept:
# they're primary sources for people and organisations. Subdomains (m.facebook.com) are blocked too.
BLOCKED_DOMAINS = (
    "facebook.com", "fb.com", "instagram.com", "tiktok.com", "pinterest.com", "threads.net",
    "youtube.com", "youtu.be",  # video pages: crawl ~1.5s for no usable text
)


class FactAssessor:
    def __init__(
        self,
        n_atoms: int = 5,
        top_k: int = 5,
        *,
        device: str = "auto",
        serper_api_key: str | None = None,
        laya_model: str = "english",  # Laya Router checkpoint: english | multilingual | typed-decisions
        checkworthy_threshold: float = 0.4,  # P(factual_claim); low on purpose: dropping a real claim costs more than one extra search
        early_exit_conf: float = 0.9,  # 2+ snippets this sure, none against -> skip crawling
        strong_evidence: float = 0.7,  # a passage counts toward a verdict at or above this prob
        crawl_timeout: float = 2.5,  # good pages crawl in ~0.6-1.6s; a 6s timeout let one dead site set the latency
        max_concurrent_crawls: int = 10,
        search_timeout: float = 5.0,
        search_hedge_after: float = 1.2,  # Serper's p50 is ~0.8s but outliers hit 3s+: fire a duplicate, take the first
        blocked_domains: tuple[str, ...] = BLOCKED_DOMAINS,
        timeout: float = 15.0,
        atomizer: Any = None,
        atomizer_model: str = DEFAULT_ATOMIZER_MODEL,
        atom_filter: Any = None,
        judge: Any = None,
    ) -> None:
        self.n_atoms = n_atoms
        self.top_k = top_k
        self.device = device
        self.serper_api_key = serper_api_key or os.environ.get("SERPER_API_KEY")
        self.laya_model = laya_model
        self.checkworthy_threshold = checkworthy_threshold
        self.early_exit_conf = early_exit_conf
        self.strong_evidence = strong_evidence
        self.crawl_timeout = crawl_timeout
        self._crawl_slots = asyncio.Semaphore(max_concurrent_crawls)
        self.search_timeout = search_timeout
        self.search_hedge_after = search_hedge_after
        self.blocked_domains = blocked_domains
        self.timeout = timeout
        self.atomizer = atomizer or Atomizer(atomizer_model)
        self.laya = LayaRunner(device)  # one model shared by the atom filter and the evidence judge
        self.atom_filter = atom_filter or AtomFilter(self.laya, n_atoms, checkworthy_threshold, laya_model)
        self.judge = judge or LayaJudge(self.laya, laya_model)
        self._http: httpx.AsyncClient | None = None  # shared connection pool, created on first use
        self._crawler: Any = None  # one shared headless browser (crawl4ai), started on first use
        self._crawler_lock = asyncio.Lock()
        self._loop: asyncio.AbstractEventLoop | None = None  # background loop behind assess_sync()
        self._loop_thread: threading.Thread | None = None

    async def __aenter__(self) -> FactAssessor:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()

    async def aload(self) -> None:
        """Warm the local pieces (Laya, browser) up front so the first check is fast."""
        await asyncio.gather(self.laya.agent(self.laya_model), self._start_crawler())

    async def aclose(self) -> None:
        if self._http is not None:
            await self._http.aclose()
            self._http = None
        if self._crawler is not None:
            await self._crawler.close()
            self._crawler = None

    def __enter__(self) -> FactAssessor:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    async def assess(self, text: str) -> CheckResult:
        """Fact-check `text`: atomize, filter, verify every claim concurrently, then score and build the graph."""
        start = time.perf_counter()

        atoms = await self.atomizer.aatomize(text)
        kept, skipped = await self.atom_filter.afilter(atoms)
        results = await self._verify_all(kept)
        graph = self._build_graph(results)
        score = self._score(results)

        return CheckResult(
            text=text,
            atoms=results,
            skipped=skipped,
            fact_score=score,
            graph=graph,
            latency_ms=(time.perf_counter() - start) * 1000,
        )

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

    def close(self) -> None:
        """Blocking `aclose` for `assess_sync` users: closes the browser and HTTP pool, stops the background loop."""
        if self._loop is None:
            return
        asyncio.run_coroutine_threadsafe(self.aclose(), self._loop).result()
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._loop_thread.join()
        self._loop.close()
        self._loop = self._loop_thread = None

    # --- stages ---------------------------------------------------------------

    async def _verify_all(self, atoms: list[Atom]) -> list[AtomResult]:
        """Verify every atom concurrently; atoms still running at `timeout` come back unverified."""
        tasks = [asyncio.create_task(self._verify(a)) for a in atoms]
        if not tasks:
            return []
        done, pending = await asyncio.wait(tasks, timeout=self.timeout)
        for t in pending:
            t.cancel()
        return [
            t.result() if t in done else AtomResult(atom=a, verdict="unverified", error="timeout")
            for t, a in zip(tasks, atoms)
        ]

    async def _verify(self, atom: Atom) -> AtomResult:
        """Search, judge the snippets, and only crawl if the snippets aren't conclusive. Pages are judged as
        each crawl lands, and the rest are abandoned as soon as the evidence settles the atom."""
        try:
            hits = await self._search(atom.text)
            evidence = await self.judge.ajudge(atom.text, hits)
            if not self._is_confident(evidence):
                pending = [asyncio.create_task(self._crawl_and_judge(atom, h["url"])) for h in hits]
                try:
                    for next_page in asyncio.as_completed(pending):
                        evidence += await next_page
                        if self._is_confident(evidence):
                            break
                finally:
                    for task in pending:
                        task.cancel()
            verdict, confidence = self._aggregate(evidence)
            return AtomResult(atom=atom, verdict=verdict, confidence=confidence, evidence=evidence)
        except Exception as exc:  # one atom failing never sinks the whole check
            return AtomResult(atom=atom, verdict="unverified", error=repr(exc))

    async def _search(self, query: str) -> list[dict[str, Any]]:
        """Serper: top_k organic results as [{"url", "title", "snippet"}], blocked domains removed.
        Hedged against Serper's slow outliers."""
        if not self.serper_api_key:
            raise RuntimeError("SERPER_API_KEY is not set (pass serper_api_key= or add it to .env)")
        return await _hedged(lambda: self._serper(query), self.search_hedge_after)

    async def _serper(self, query: str) -> list[dict[str, Any]]:
        if self._http is None:
            self._http = httpx.AsyncClient(timeout=self.search_timeout)
        response = await self._http.post(
            SERPER_URL,
            headers={"X-API-KEY": self.serper_api_key},
            json={"q": query, "num": 2 * self.top_k},  # over-fetch so blocked results don't leave us short
        )
        response.raise_for_status()
        hits = [
            {"url": r["link"], "title": r.get("title", ""), "snippet": r.get("snippet", "")}
            for r in response.json().get("organic", [])
            if not self._is_blocked(r["link"])
        ]
        return hits[: self.top_k]

    def _is_blocked(self, url: str) -> bool:
        host = urlparse(url).netloc.lower()
        return any(host == d or host.endswith("." + d) for d in self.blocked_domains)

    async def _crawl(self, url: str) -> dict[str, Any] | None:
        """crawl4ai page as {"url", "title", "text"} (clean text), or None on failure/timeout."""
        from crawl4ai import CacheMode, CrawlerRunConfig, DefaultMarkdownGenerator

        config = CrawlerRunConfig(
            # plain text: markdown links/images were ~2/3 of the characters and wasted the judge's token budget
            markdown_generator=DefaultMarkdownGenerator(options={"ignore_links": True, "ignore_images": True}),
            cache_mode=CacheMode.BYPASS,
            wait_until="domcontentloaded",  # don't wait for ads/trackers to finish loading
            page_timeout=int(self.crawl_timeout * 1000),
            excluded_tags=["nav", "footer", "header", "aside", "form", "script", "style"],
            exclude_all_images=True,
            verbose=False,
        )
        try:
            crawler = await self._start_crawler()
            async with self._crawl_slots:
                result = await asyncio.wait_for(crawler.arun(url, config=config), self.crawl_timeout)
        except Exception:  # timeouts, dead hosts, browser hiccups: the snippet is still there
            return None
        text = clean_text(result.markdown.raw_markdown) if result.success and result.markdown else ""
        if not text:
            return None
        return {"url": url, "title": (result.metadata or {}).get("title", ""), "text": text}

    async def _crawl_and_judge(self, atom: Atom, url: str) -> list[Evidence]:
        """One page: judged the moment its own crawl finishes, not after the slowest page."""
        page = await self._crawl(url)
        return await self.judge.ajudge(atom.text, [page]) if page else []

    def _build_graph(self, results: list[AtomResult]) -> dict[str, Any]:
        """Atom and source nodes; source -> atom edges for strong supports/refutes (max prob per pair)."""
        nodes: dict[str, dict[str, Any]] = {}
        edges: dict[tuple[str, str, str], float] = {}
        for r in results:
            atom_id = f"atom:{r.atom.id}"
            nodes[atom_id] = {"id": atom_id, "kind": "atom", "label": r.atom.text, "verdict": r.verdict}
            for e in r.evidence:
                if e.label == "not_enough_info" or e.prob < self.strong_evidence:
                    continue
                source_id = f"source:{urlparse(e.url).netloc.removeprefix('www.') or e.url}"
                nodes.setdefault(source_id, {"id": source_id, "kind": "source", "label": source_id[7:], "url": e.url})
                key = (source_id, atom_id, e.label)
                edges[key] = max(edges.get(key, 0.0), e.prob)
        return {
            "nodes": list(nodes.values()),
            "edges": [{"source": s, "target": t, "relation": rel, "weight": round(w, 3)} for (s, t, rel), w in edges.items()],
        }

    def _score(self, results: list[AtomResult]) -> float | None:
        """supported / (supported + refuted + contested); None if nothing was decided."""
        decided = [r for r in results if r.verdict != "unverified"]
        return sum(r.verdict == "supported" for r in decided) / len(decided) if decided else None

    # --- helpers --------------------------------------------------------------

    async def _start_crawler(self) -> Any:
        async with self._crawler_lock:
            if self._crawler is None:
                from crawl4ai import AsyncWebCrawler, BrowserConfig

                crawler = AsyncWebCrawler(
                    config=BrowserConfig(headless=True, text_mode=True, light_mode=True, verbose=False)
                )
                await crawler.start()
                self._crawler = crawler
        return self._crawler

    def _is_confident(self, evidence: list[Evidence]) -> bool:
        """Early exit: 2+ passages agree at >= early_exit_conf and none strongly disagree.
        One sure passage isn't enough: the judge can call a related-but-different fact a refutation."""
        sure = [e.label for e in evidence if e.label != "not_enough_info" and e.prob >= self.early_exit_conf]
        against = {e.label for e in evidence if e.label != "not_enough_info" and e.prob >= self.strong_evidence}
        return any(sure.count(side) >= 2 and against == {side} for side in ("supports", "refutes"))

    def _aggregate(self, evidence: list[Evidence]) -> tuple[Verdict, float]:
        """Weigh strong evidence on each side; a side wins if it has at least 2x the other's weight."""
        support = [e.prob for e in evidence if e.label == "supports" and e.prob >= self.strong_evidence]
        refute = [e.prob for e in evidence if e.label == "refutes" and e.prob >= self.strong_evidence]
        s, r = sum(support), sum(refute)
        if s == r == 0:
            return "unverified", 0.0
        if s >= 2 * r:
            return "supported", max(support)
        if r >= 2 * s:
            return "refuted", max(refute)
        return "contested", max(s, r) / (s + r)



async def _hedged(make: Any, after: float) -> Any:
    """Await `make()`; if it hasn't answered within `after` seconds, race a duplicate and take the first success."""
    first = asyncio.ensure_future(make())
    done, _ = await asyncio.wait({first}, timeout=after)
    if done:
        return first.result()
    racing = {first, asyncio.ensure_future(make())}
    error: BaseException | None = None
    try:
        while racing:
            done, racing = await asyncio.wait(racing, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                if task.exception() is None:
                    return task.result()
                error = task.exception()
        raise error  # type: ignore[misc]  # both attempts failed
    finally:
        for task in racing:
            task.cancel()
