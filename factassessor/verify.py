"""Atom -> AtomResult: search, judge, crawl only if needed, stop as soon as the evidence settles the claim."""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from typing import Any

from factassessor.pipeline import Map, Scan, Step, TakeUntil, collect, last, once
from factassessor.schema import Atom, AtomResult, Evidence, Verdict

_FROM_JUDGE: Any = object()  # "use the judge's concurrency" (so concurrency=None can mean "no limit")


class Policy(ABC):
    """Role: turns evidence into a verdict, and decides when there's enough evidence to stop looking."""

    @abstractmethod
    def settled(self, evidence: list[Evidence]) -> bool: ...

    @abstractmethod
    def verdict(self, evidence: list[Evidence]) -> tuple[Verdict, float]: ...


class WeightedPolicy(Policy):
    """Strong evidence (prob >= `strong`, not not_enough_info) is weighed per side; a side wins with at least 2x
    the other side's weight, otherwise the claim is contested. One strong refutation shouldn't flip several
    supports: the judge sometimes calls a related-but-different fact a refutation."""

    def __init__(self, strong: float = 0.7, early_exit: float = 0.9) -> None:
        self.strong = strong
        self.early_exit = early_exit

    def settled(self, evidence: list[Evidence]) -> bool:
        """Early exit: 2+ passages agree at >= early_exit and none strongly disagree."""
        sure = [e.label for e in evidence if e.label != "not_enough_info" and e.prob >= self.early_exit]
        against = {e.label for e in evidence if e.label != "not_enough_info" and e.prob >= self.strong}
        return any(sure.count(side) >= 2 and against == {side} for side in ("supports", "refutes"))

    def verdict(self, evidence: list[Evidence]) -> tuple[Verdict, float]:
        support = [e.prob for e in evidence if e.label == "supports" and e.prob >= self.strong]
        refute = [e.prob for e in evidence if e.label == "refutes" and e.prob >= self.strong]
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

    searcher: step, query -> hits      crawler: step, url -> pages
    judge:    a Judge (`judge(claim, docs)`)      policy: a Policy (`settled(ev)`, `verdict(ev)`)
    concurrency: claims verified at once, like `Map(concurrency=)`. Default: the judge's `concurrency` (none for
        Laya and LLM judges, so every claim starts at once); None: no limit. A claim's `timeout` starts when it
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
    ) -> None:
        self.searcher = searcher
        self.crawler = crawler
        self.judge = judge
        self.policy = policy or WeightedPolicy()
        self.timeout = timeout
        self.concurrency = getattr(judge, "concurrency", None) if concurrency is _FROM_JUDGE else concurrency

    def __call__(self, atoms: AsyncIterator[Atom]) -> AsyncIterator[AtomResult]:
        return Map(self.verify, concurrency=self.concurrency)(atoms)

    async def verify(self, atom: Atom) -> AtomResult:
        """Never raises: a failed claim comes back unverified with `error` set. A slow one is decided on the evidence
        judged before its `timeout` (snippets, pages that landed), with `error="timeout"`."""
        so_far: list[Evidence] = []  # the running total, kept current by _verify
        try:
            return await asyncio.wait_for(self._verify(atom, so_far), self.timeout)
        except TimeoutError:
            verdict, confidence = self.policy.verdict(so_far)
            return AtomResult(atom=atom, verdict=verdict, confidence=confidence, evidence=so_far, error="timeout")
        except Exception as exc:
            return AtomResult(atom=atom, verdict="unverified", error=repr(exc))

    async def _verify(self, atom: Atom, so_far: list[Evidence]) -> AtomResult:
        hits = await collect(self.searcher(once(atom.text)))
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
            gather_evidence = (
                self.crawler >> judge_page >> Scan(add, evidence) >> TakeUntil(self.policy.settled)
            )
            evidence = await last(gather_evidence(_urls(hits)), default=evidence)
        verdict, confidence = self.policy.verdict(evidence)
        return AtomResult(atom=atom, verdict=verdict, confidence=confidence, evidence=evidence)


async def _urls(hits: list[dict[str, Any]]) -> AsyncIterator[str]:
    for hit in hits:
        yield hit["url"]
