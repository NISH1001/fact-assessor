"""Atom -> AtomResult: search, judge, resolve and crawl only if needed, stop as soon as the evidence settles it."""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator
from typing import Any, Protocol, runtime_checkable

from factassessor.keys import auth_error
from factassessor.pipeline import Map, Scan, Slots, Step, Take, TakeUntil, collect, last, once
from factassessor.resolvers import Resolver, locations
from factassessor.schema import Atom, AtomResult, Evidence, Verdict

_FROM_JUDGE: Any = object()  # "use the judge's concurrency" (so concurrency=None can mean "no limit")


@runtime_checkable
class Policy(Protocol):
    """Role: turns evidence into a verdict, and decides when there's enough evidence to stop looking."""

    def settled(self, evidence: list[Evidence]) -> bool: ...

    def verdict(self, evidence: list[Evidence]) -> tuple[Verdict, float]: ...


class WeightedPolicy(Policy):
    """Strong evidence is weighed per side: a support from `strong`, a refutation from `strong_refute`. A side wins
    with at least 2x the other side's weight, otherwise the claim is contested. A refutation needs more confidence
    because the judge calls a related-but-different fact (another paper by the same authors, another year) a
    refutation: on the fully live run's evidence, refutations from 0.9 took recall 0.632 -> 0.690 on FactReasoner's
    atoms (0.718 -> 0.770 on the synthetic set) for 3 (12) more false atoms through."""

    def __init__(self, strong: float = 0.7, early_exit: float = 0.9, strong_refute: float = 0.9) -> None:
        self.strong = strong
        self.early_exit = early_exit
        self.strong_refute = strong_refute

    def _strong(self, e: Evidence) -> bool:
        return (e.label == "supports" and e.prob >= self.strong) or (e.label == "refutes" and e.prob >= self.strong_refute)

    def settled(self, evidence: list[Evidence]) -> bool:
        """Early exit: 2+ passages agree at >= early_exit and none strongly disagree."""
        sure = [e.label for e in evidence if e.label != "not_enough_info" and e.prob >= self.early_exit]
        against = {e.label for e in evidence if self._strong(e)}
        return any(sure.count(side) >= 2 and against == {side} for side in ("supports", "refutes"))

    def verdict(self, evidence: list[Evidence]) -> tuple[Verdict, float]:
        support = [e.prob for e in evidence if e.label == "supports" and e.prob >= self.strong]
        refute = [e.prob for e in evidence if e.label == "refutes" and e.prob >= self.strong_refute]
        s, r = sum(support), sum(refute)
        if s == r == 0:
            return "unverified", 0.0
        if s >= 2 * r:
            return "supported", max(support)
        if r >= 2 * s:
            return "refuted", max(refute)
        return "contested", max(s, r) / (s + r)


class Verify(Step):
    """Per claim: judge the search snippets; if they don't settle it, crawl every hit concurrently, judge each
    page the moment it lands, and stop (cancelling the remaining crawls) once the policy says settled.

        search -> judge snippets -> [resolve ->] crawl -> judge each page -> running total -> stop when settled

    searcher: step, query -> hits      crawler: step, url -> pages (a Crawler, `crawl(url)`, when resolving)
    judge:    a Judge (`judge(claim, docs)`)      policy: a Policy (`settled(ev)`, `verdict(ev)`)
    resolver: optional Resolver (`resolve(url) -> urls`; `CompositeResolver` for several). Every hit is resolved at
        once; its locations (`resolvers.locations`: a direct-PDF hit, the free copies, the hit's own page) are then
        crawled in order and the first readable one is judged, cited under the hit's URL. A copy claims to be the
        full document, so it needs `min_copy_words` (bot-check pages are shorter); the hit's own page is taken as the
        crawler returns it. A hit has no clock of its own: each fetch is bounded by the crawler's timeouts, a hit has
        a handful of locations, and the claim's `timeout` caps the rest. (A per-hit deadline used to start when the
        hit was handed to the crawler, so under load it timed the wait for a connection: with 6 texts checked at
        once, 93% of crawls expired before starting and 571 of 1,667 claims lost pages that were readable.)
    concurrency: claims verified at once, shared by every stream through this Verify (all texts in flight). Default:
        the judge's `concurrency` (none for Laya and LLM judges); None: no limit. A claim's `timeout` starts when it
        gets its slot, so claims waiting for a slow judge don't time out in the queue.
    """

    def __init__(
        self,
        searcher: Step,
        crawler: Step,
        judge: Any,
        policy: Any = None,
        timeout: float = 15.0,
        concurrency: int | None | Any = _FROM_JUDGE,
        resolver: Resolver | None = None,
        min_copy_words: int = 300,
        pages_per_claim: int | None = None,
    ) -> None:
        if resolver is not None and not callable(getattr(crawler, "crawl", None)):
            raise TypeError("resolving needs a Crawler (crawl(url) -> page) to read each location")
        self.searcher = searcher
        self.crawler = crawler
        self.resolver = resolver
        # overfetch: with more hits than this, the first `pages_per_claim` that turn out readable are judged and the
        # other crawls are cancelled, so a paywalled or blocked hit is replaced by the next one instead of lost
        self.pages_per_claim = pages_per_claim
        self.min_copy_words = min_copy_words
        self.judge = judge
        self.policy = policy or WeightedPolicy()
        self.timeout = timeout
        limit = getattr(judge, "concurrency", None) if concurrency is _FROM_JUDGE else concurrency
        # one pool for every stream through this Verify: several texts at once share it instead of multiplying it
        self.slots = limit if isinstance(limit, Slots) or limit is None else Slots(limit)
        self.concurrency = self.slots.n if self.slots else None

    def __call__(self, atoms: AsyncIterator[Atom]) -> AsyncIterator[AtomResult]:
        return Map(self.verify, concurrency=self.slots)(atoms)  # a claim starts (and its clock) once it has a slot

    async def verify(self, atom: Atom) -> AtomResult:
        """Never raises: a failed claim comes back unverified with `error` set. A slow one is decided on the evidence
        judged before its `timeout` (snippets, pages that landed), with `error="timeout"`."""
        so_far: list[Evidence] = []  # the running total, kept current by _verify
        start = time.perf_counter()
        try:
            result = await asyncio.wait_for(self._verify(atom, so_far), self.timeout)
        except TimeoutError:
            verdict, confidence = self.policy.verdict(so_far)
            result = AtomResult(atom=atom, verdict=verdict, confidence=confidence, evidence=so_far, error="timeout")
        except Exception as exc:
            if err := auth_error(exc):
                raise err from exc  # a rejected key fails every claim the same way: stop the whole check
            result = AtomResult(atom=atom, verdict="unverified", error=repr(exc))
        return result.model_copy(update={"latency_ms": (time.perf_counter() - start) * 1000})

    async def _verify(self, atom: Atom, so_far: list[Evidence]) -> AtomResult:
        # the claim's own search, plus the text's source queries when the atomizer wrote them (a claim about a detail
        # inside a paper rarely finds the paper by itself); the claim's hits come first, each url once
        queries = list(dict.fromkeys([atom.text, *atom.source_queries]))
        found = await asyncio.gather(*(collect(self.searcher(once(q))) for q in queries), return_exceptions=True)
        failed = [f for f in found if isinstance(f, BaseException)]
        if len(failed) == len(found):
            raise failed[0]  # every search failed: the claim comes back unverified with the error
        hits = list({h["url"]: h for hs in found if not isinstance(hs, BaseException) for h in hs}.values())
        evidence = await self.judge.judge(atom.text, hits)  # snippets first: often enough on their own
        so_far[:] = evidence
        if hits and not self.policy.settled(evidence):

            async def judge_page(page: dict[str, Any]) -> list[Evidence]:
                return await self.judge.judge(atom.text, [page])

            def add(total: list[Evidence], new: list[Evidence]) -> list[Evidence]:
                so_far[:] = total + new  # visible to verify() if the timeout strikes mid-crawl
                return so_far[:]

            # crawl every hit at once -> judge each page as it lands -> running total of the evidence ->
            # stop at the first total that settles the claim (TakeUntil cancels the crawls still in flight)
            read = Map(self._resolve) >> Map(self._read) if self.resolver else self.crawler
            if self.pages_per_claim is not None:
                read = read >> Take(self.pages_per_claim)
            gather_evidence = read >> judge_page >> Scan(add, evidence) >> TakeUntil(self.policy.settled)
            evidence = await last(gather_evidence(_urls(hits)), default=evidence)
        verdict, confidence = self.policy.verdict(evidence)
        return AtomResult(atom=atom, verdict=verdict, confidence=confidence, evidence=evidence)


    async def _resolve(self, url: str) -> tuple[str, list[str]]:
        """hit url -> (hit url, the locations to read it from, in order). Never raises."""
        try:
            candidates = await self.resolver.resolve(url)  # type: ignore[union-attr]
        except Exception:
            candidates = []
        return url, locations(url, candidates)

    async def _read(self, source: tuple[str, list[str]]) -> dict[str, Any] | None:
        """The first readable location, as the hit's page; None (dropped) if none is."""
        url, where = source
        return await read_first(self.crawler, url, where, self.min_copy_words)


async def read_first(crawler: Any, url: str, where: list[str], min_copy_words: int = 300) -> dict[str, Any] | None:
    """Crawl `where` (a hit's locations, `resolvers.locations`) in order; the first readable one, as `url`'s page.
    A location other than `url` is a copy claiming to be the full document, so it needs `min_copy_words`."""
    for location in where:
        f = await crawler.crawl(location)
        if f and (location == url or len(f.page["text"].split()) >= min_copy_words):
            return {**f.page, "url": url}
    return None


async def _urls(hits: list[dict[str, Any]]) -> AsyncIterator[str]:
    for hit in hits:
        yield hit["url"]
