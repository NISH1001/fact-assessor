"""Atom -> AtomResult: search, judge, crawl only if needed, stop as soon as the evidence settles the claim."""

from __future__ import annotations

import asyncio
import operator
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from typing import Any

from factassessor.pipeline import Map, Scan, Step, TakeUntil, collect, last, once
from factassessor.schema import Atom, AtomResult, Evidence, Verdict


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
    """

    def __init__(self, searcher: Step, crawler: Step, judge: Any, policy: Any = None, timeout: float = 15.0) -> None:
        self.searcher = searcher
        self.crawler = crawler
        self.judge = judge
        self.policy = policy or WeightedPolicy()
        self.timeout = timeout

    def __call__(self, atoms: AsyncIterator[Atom]) -> AsyncIterator[AtomResult]:
        return Map(self.verify)(atoms)

    async def verify(self, atom: Atom) -> AtomResult:
        """Never raises: a failed or slow claim comes back unverified with `error` set."""
        try:
            return await asyncio.wait_for(self._verify(atom), self.timeout)
        except TimeoutError:
            return AtomResult(atom=atom, verdict="unverified", error="timeout")
        except Exception as exc:
            return AtomResult(atom=atom, verdict="unverified", error=repr(exc))

    async def _verify(self, atom: Atom) -> AtomResult:
        hits = await collect(self.searcher(once(atom.text)))
        evidence = await self.judge.judge(atom.text, hits)  # snippets first: often enough on their own
        if hits and not self.policy.settled(evidence):

            async def judge_page(page: dict[str, Any]) -> list[Evidence]:
                return await self.judge.judge(atom.text, [page])

            # crawl every hit at once -> judge each page as it lands -> running total of the evidence ->
            # stop at the first total that settles the claim (TakeUntil cancels the crawls still in flight)
            gather_evidence = (
                self.crawler >> judge_page >> Scan(operator.add, evidence) >> TakeUntil(self.policy.settled)
            )
            evidence = await last(gather_evidence(_urls(hits)), default=evidence)
        verdict, confidence = self.policy.verdict(evidence)
        return AtomResult(atom=atom, verdict=verdict, confidence=confidence, evidence=evidence)


async def _urls(hits: list[dict[str, Any]]) -> AsyncIterator[str]:
    for hit in hits:
        yield hit["url"]
