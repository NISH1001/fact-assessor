"""Standalone end-to-end latency: FactAssessor.assess(text) on 10 answers at once (5 shortest, 5 longest originals),
the current configuration (Serper, overfetch 1.0, resolvers, top-3, Jev packed per claim). Times only, no scoring.

    uv run python latency_e2e.py                      # fresh: atomizer + Serper live (about 180 credits)
    uv run python latency_e2e.py --cached --claims 40 --http 20 --tabs 10
        --cached: atoms and search hits from the first run (no credits; search time then ~0 of a claim's time)
        --claims: claims verified at once (FactAssessor max_concurrent_claims); the rest queue, their deadline starts at their slot
        --http / --tabs: crawler connections / browser tabs
"""

import argparse, asyncio, json, os, statistics, time
from pathlib import Path

import httpx

from factassessor import (ArxivResolver, Atom, Atomizer, CompositeResolver, Crawl4AICrawler, DecisionClaimFilter, DecisionJudge,
                          FactAssessor, FallbackCrawler, HTTPXCrawler, LLMAtomizer, OpenAlexResolver, Searcher, SerperSearcher,
                          SystemOneRunner, Take, WeightedPolicy, not_blocked)

HERE = Path(__file__).parent
rows = [json.loads(l) for l in Path("tmp/scielf_paired.jsonl").read_text().splitlines() if l.strip()]
originals = sorted((r["original"]["long_form_answer"] for r in rows if not r.get("is_synthetic")), key=lambda t: len(t.split()))
texts = [("short", t) for t in originals[:5]] + [("long", t) for t in originals[-5:]]


class CachedAtomizer(Atomizer):
    """The first run's atoms per text, so reruns search the same queries."""

    def __init__(self, inner: LLMAtomizer, path: Path) -> None:
        self.inner, self.path = inner, path
        self.cache = json.loads(path.read_text()) if path.exists() else {}

    async def atomize(self, text: str) -> list[Atom]:
        if text not in self.cache:
            self.cache[text] = [a.model_dump() for a in await self.inner.atomize(text)]
            self.path.write_text(json.dumps(self.cache))
        return [Atom(**a) for a in self.cache[text]]


class CachedSerper(Searcher):
    """Hits per query from the first run; a miss goes to Serper (1 credit) and is saved."""

    def __init__(self, inner: SerperSearcher, path: Path) -> None:
        self.inner, self.path = inner, path
        self.cache = json.loads(path.read_text()) if path.exists() else {}

    async def search(self, query: str) -> list[dict]:
        if query not in self.cache:
            self.cache[query] = await self.inner.search(query)
            self.path.write_text(json.dumps(self.cache))
        return self.cache[query]


def balance() -> int:
    return httpx.get("https://google.serper.dev/account", headers={"X-API-KEY": os.environ["SERPER_API_KEY"]}, timeout=15).json()["balance"]


async def main(args: argparse.Namespace) -> None:
    jev = SystemOneRunner()
    atomizer = LLMAtomizer("openai:gpt-6-luna", source_query=True)
    serper = SerperSearcher(num=10, hedge_after=None)
    if args.cached:
        atomizer, serper = CachedAtomizer(atomizer, HERE / "lat-atoms.json"), CachedSerper(serper, HERE / "lat-hits.json")
    fa = FactAssessor(
        n_atoms=50, top_k=5, overfetch=1.0, source_query=True, timeout=args.timeout, max_concurrent_claims=args.claims,
        atomizer=atomizer,
        claim_filter=DecisionClaimFilter(jev),
        searcher=serper >> not_blocked() >> Take(10),
        resolver=CompositeResolver(ArxivResolver(), OpenAlexResolver()),
        crawler=FallbackCrawler(HTTPXCrawler(max_concurrent=args.http), Crawl4AICrawler(timeout=2.5, max_concurrent=args.tabs)),
        judge=DecisionJudge(jev, passages_per_page=3),
        policy=WeightedPolicy(strong=0.7, early_exit=0.9),
    )
    await fa.aload()  # browser, HTTP pools: not part of a request's time
    before = balance()
    t0 = time.perf_counter()
    if args.sequential:
        results = [await fa.assess(t) for _, t in texts]
    else:
        results = await asyncio.gather(*(fa.assess(t) for _, t in texts))
    wall = time.perf_counter() - t0
    await fa.aclose()
    after = balance()
    lat, claims, timeouts = [], 0, 0
    for (kind, text), r in zip(texts, results):
        s = r.latency_ms / 1000
        lat.append((kind, s))
        claims += len(r.atoms); timeouts += sum(a.error == "timeout" for a in r.atoms)
        verdicts = {v[:4]: sum(a.verdict == v for a in r.atoms) for v in ("supported", "refuted", "contested", "unverified")}
        print(f"{kind:5s} {len(text.split()):4d} words  {len(r.atoms):2d} claims, {len(r.skipped)} filtered  {s:6.1f}s  "
              f"timeouts {sum(a.error == 'timeout' for a in r.atoms):2d}  {verdicts}")
    for name, sel in (("all 10", [s for _, s in lat]), ("short", [s for k, s in lat if k == "short"]), ("long", [s for k, s in lat if k == "long"])):
        q = sorted(sel)
        print(f"{name:7s} n={len(q)}: p50 {statistics.median(q):.1f}s  p90 {q[max(0, int(round(0.9 * len(q))) - 1)]:.1f}s  max {q[-1]:.1f}s")
    print(f"claims {claims}, timeouts {timeouts} ({100 * timeouts / claims:.0f}%); wall clock {wall:.1f}s ({"sequential" if args.sequential else "all at once"}); serper credits used {before - after} "
          f"(balance {after}); jev ${jev.cost:.3f}; settings: deadline {args.timeout}s, claims at once {args.claims}, http {args.http}, tabs {args.tabs}, cached {args.cached}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cached", action="store_true")
    ap.add_argument("--sequential", action="store_true")
    ap.add_argument("--timeout", type=float, default=42.0)
    ap.add_argument("--claims", type=int, default=None)
    ap.add_argument("--http", type=int, default=20)
    ap.add_argument("--tabs", type=int, default=10)
    asyncio.run(main(ap.parse_args()))
