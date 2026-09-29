"""Atom-level eval on a labelled long-form set (SciELF-style JSONL): the paper's metric, with timings.

Each JSONL row is a pair: an `original` and a `corrupted` long-form answer, each with human-labelled atoms
(`{"text", "label": "S" | "NS"}`). The data is external: keep it in tmp/ (gitignored).

    # live: gpt-6-luna atomizer + Laya filter (timed per answer), then every labelled atom verified live
    # (SearXNG, resolvers, HTTPX -> browser, Laya); search hits and pages are cached for replays
    uv run python scripts/eval_atoms.py live --data tmp/scielf_paired.jsonl --tag web

    # replay: the same cached evidence, another judging setup, in seconds (no network)
    uv run python scripts/eval_atoms.py replay --tag web --out web-top3 --passages 3

    uv run python scripts/eval_atoms.py report --tag web

Metric (the paper's Table 3): atom-level accuracy, precision, recall, F1 with S (true) as the positive class,
computed per answer and macro-averaged over answers, on the corrupted split and on original + corrupted combined.
An atom is predicted S when its verdict is `supported`; unverified, contested and refuted are NS (the reference
system likewise needs P(true) > 0.5 from a 0.5 prior, so no evidence means NS).

Timings are per claim and per answer, as busy wall-clock time per component (overlapping calls count once):
atomize, filter, search, snippet judging, resolve, crawl, page judging, and the totals.
"""

from __future__ import annotations

import argparse
import asyncio
import contextvars
import gzip
import json
import statistics
import time
from pathlib import Path
from typing import Any

from factassessor import (
    ArxivResolver, Atom, CompositeResolver, Crawl4AICrawler, Crawler, FallbackCrawler, HTTPXCrawler, LayaClaimFilter,
    LayaJudge, LLMAtomizer, OpenAlexResolver, SearxngSearcher, Step, Take, Verify, WeightedPolicy, collect, not_blocked,
    once,
)
from factassessor.resolvers import locations

OUT = Path("tmp/eval_atoms")
TOP_K = 5
# the reference system's published numbers (paper Table 3, manual portion): (accuracy, precision, recall, F1)
REFERENCE = {
    ("in-domain", "corrupted"): (0.819, 0.890, 0.837, 0.841),
    ("in-domain", "combined"): (0.819, 0.945, 0.828, 0.867),
    ("web", "corrupted"): (0.703, 0.803, 0.698, 0.694),
    ("web", "combined"): (0.673, 0.902, 0.671, 0.731),
}


# --- data ---------------------------------------------------------------------------------------------------------

def load_answers(path: str, manual_only: bool = True, limit: int | None = None) -> list[dict[str, Any]]:
    """One entry per answer: {id, pair, kind (original | corrupted), text, atoms [{text, label}]}."""
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    if manual_only:
        rows = [r for r in rows if not r.get("is_synthetic")]
    rows = rows[:limit] if limit else rows
    answers = []
    for i, r in enumerate(rows):
        for kind in ("original", "corrupted"):
            a = r[kind]
            answers.append({"id": f"p{i:03d}-{kind}", "pair": i, "kind": kind, "text": a["long_form_answer"],
                            "atoms": [{"text": x["text"], "label": x["label"]} for x in a["atoms"]]})
    return answers


# --- metric -------------------------------------------------------------------------------------------------------

def answer_metrics(labels: list[str], predicted_s: list[bool]) -> tuple[float, float, float, float]:
    """Accuracy, precision, recall, F1 for one answer, S as positive; an undefined ratio is 0 (zero_division=0)."""
    tp = sum(lab == "S" and p for lab, p in zip(labels, predicted_s))
    fp = sum(lab == "NS" and p for lab, p in zip(labels, predicted_s))
    fn = sum(lab == "S" and not p for lab, p in zip(labels, predicted_s))
    acc = sum((lab == "S") == p for lab, p in zip(labels, predicted_s)) / len(labels)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return acc, prec, rec, f1


def macro(results: list[dict[str, Any]], split: str, predict: Any) -> tuple[float, float, float, float, int]:
    """Macro average over the answers in `split` (original | corrupted | combined) of `answer_metrics`."""
    chosen = [r for r in results if split == "combined" or r["kind"] == split]
    per = [answer_metrics([a["label"] for a in r["atoms"]], [predict(a) for a in r["atoms"]]) for r in chosen]
    return (*(statistics.mean(m[i] for m in per) for i in range(4)), len(chosen))


# --- timing -------------------------------------------------------------------------------------------------------

_spans: contextvars.ContextVar[dict[str, list[tuple[float, float]]] | None] = contextvars.ContextVar("spans", default=None)


def _record(component: str, start: float, end: float) -> None:
    if (spans := _spans.get()) is not None:
        spans.setdefault(component, []).append((start, end))


def busy(intervals: list[tuple[float, float]]) -> float:
    """Wall-clock time covered by `intervals` (overlapping calls count once)."""
    total, cur_start, cur_end = 0.0, None, None
    for s, e in sorted(intervals):
        if cur_end is None or s > cur_end:
            if cur_end is not None:
                total += cur_end - cur_start
            cur_start, cur_end = s, e
        else:
            cur_end = max(cur_end, e)
    return total + (cur_end - cur_start if cur_end is not None else 0.0)


class TimedSearch(Step):
    """The searcher, timed; hits cached by query. SearXNG's engines suspend bursts, so at most `slots` at once
    (the wait counts as search time: it's what a self-hosted searcher costs)."""

    def __init__(self, inner: Step, cache: dict[str, list[dict[str, Any]]], slots: int = 4) -> None:
        self.inner, self.cache, self._slots = inner, cache, asyncio.Semaphore(slots)

    async def __call__(self, queries: Any) -> Any:
        async for q in queries:
            start = time.perf_counter()
            if q not in self.cache:
                async with self._slots:
                    try:
                        self.cache[q] = await collect(self.inner(once(q)))
                    except Exception:
                        self.cache[q] = []
            _record("search", start, time.perf_counter())
            for hit in self.cache[q]:
                yield hit


class TimedResolver:
    def __init__(self, inner: Any) -> None:
        self.inner = inner

    async def resolve(self, url: str) -> list[str]:
        start = time.perf_counter()
        try:
            return await self.inner.resolve(url)
        finally:
            _record("resolve", start, time.perf_counter())

    async def aclose(self) -> None:
        await self.inner.aclose()


class TimedCrawler(Crawler):
    def __init__(self, inner: Crawler) -> None:
        self.inner = inner

    async def crawl(self, url: str) -> dict[str, Any] | None:
        start = time.perf_counter()
        try:
            return await self.inner.crawl(url)
        finally:
            _record("crawl", start, time.perf_counter())


class TimedJudge:
    """The judge, timed separately for snippets and pages; records every page it sees (the replay cache)."""

    def __init__(self, inner: Any, pages: dict[str, Any]) -> None:
        self.inner, self.pages = inner, pages

    async def judge(self, claim: str, docs: list[dict[str, Any]]) -> Any:
        start = time.perf_counter()
        is_page = bool(docs) and "text" in docs[0]
        for d in docs if is_page else ():
            self.pages[d["url"]] = d
        try:
            return await self.inner.judge(claim, docs)
        finally:
            _record("judge_pages" if is_page else "judge_snippets", start, time.perf_counter())


# --- replay components --------------------------------------------------------------------------------------------

class CachedSearch(Step):
    def __init__(self, cache: dict[str, list[dict[str, Any]]]) -> None:
        self.cache = cache

    async def __call__(self, queries: Any) -> Any:
        async for q in queries:
            for hit in self.cache.get(q, []):
                yield hit


class CachedCrawler(Crawler):
    """Pages as the live run read them (after resolving), by hit URL; None for hits that couldn't be read."""

    def __init__(self, pages: dict[str, Any]) -> None:
        self.pages = pages

    async def crawl(self, url: str) -> dict[str, Any] | None:
        return self.pages.get(url)


# --- runs ---------------------------------------------------------------------------------------------------------

def _load(path: Path, default: Any) -> Any:
    return json.loads(gzip.decompress(path.read_bytes())) if path.exists() else default


def _save(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(gzip.compress(json.dumps(data, ensure_ascii=False).encode()))


def _atom_record(atom: dict[str, Any], result: Any, spans: dict[str, list[tuple[float, float]]], total: float) -> dict:
    return {**atom, "verdict": result.verdict, "confidence": result.confidence, "error": result.error,
            "evidence": [e.model_dump() for e in result.evidence],
            "time": {**{k: busy(v) for k, v in spans.items()}, "total": total}}


async def verify_answer(verify: Verify, answer: dict[str, Any]) -> tuple[list[dict[str, Any]], float]:
    """Every atom of an answer at once, as the pipeline checks a text's claims; per-atom spans and totals."""

    async def one(a: dict[str, Any]) -> dict[str, Any]:
        spans: dict[str, list[tuple[float, float]]] = {}
        _spans.set(spans)  # this task's context only: each atom's own spans
        start = time.perf_counter()
        result = await verify.verify(Atom(id=0, text=a["text"], span=(0, len(a["text"]))))
        return _atom_record(a, result, spans, time.perf_counter() - start)

    start = time.perf_counter()
    atoms = await asyncio.gather(*(one(a) for a in answer["atoms"]))
    return list(atoms), time.perf_counter() - start


async def fill_pages(cache_hits: dict[str, Any], pages: dict[str, Any], resolver: Any, crawler: Crawler,
                     atoms: list[dict[str, Any]], min_copy_words: int = 300) -> None:
    """Read the hits the live run skipped (early exit), so replays with another judge see every page. Untimed."""
    from factassessor.verify import read_first

    todo = {h["url"] for a in atoms for h in cache_hits.get(a["text"], [])} - set(pages)
    slots = asyncio.Semaphore(8)

    async def read(url: str) -> None:
        async with slots:
            try:
                async with asyncio.timeout(8.0):
                    where = locations(url, await resolver.resolve(url)) if resolver else [url]
                    pages[url] = await read_first(crawler, url, where, min_copy_words)
            except Exception:
                pages[url] = None

    await asyncio.gather(*(read(u) for u in todo))


async def live(args: argparse.Namespace) -> None:
    answers = load_answers(args.data, limit=args.limit)
    run_dir = OUT / args.tag
    hits: dict[str, Any] = _load(run_dir / "hits.json.gz", {})
    pages: dict[str, Any] = _load(run_dir / "pages.json.gz", {})
    done = {r["id"]: r for r in _load(run_dir / "results.json.gz", [])}

    atomizer = LLMAtomizer(args.atomizer)
    claim_filter = LayaClaimFilter()
    judge = LayaJudge(passages_per_page=args.passages)
    searcher = SearxngSearcher(args.searxng, num=2 * TOP_K, timeout=20.0, search_type=args.search_type,
                               hedge_after=None) >> not_blocked() >> Take(TOP_K)
    base_resolver = None if args.no_resolver else CompositeResolver(ArxivResolver(), OpenAlexResolver())
    base_crawler = FallbackCrawler(HTTPXCrawler(), Crawl4AICrawler(timeout=2.5))
    verify = Verify(TimedSearch(searcher, hits), TimedCrawler(base_crawler), TimedJudge(judge, pages),
                    WeightedPolicy(strong=args.strong), resolver=TimedResolver(base_resolver) if base_resolver else None)
    await judge.judge("warm-up", [{"url": "u", "title": "", "snippet": "warm-up"}])  # load Laya before timing

    for n, answer in enumerate(answers, 1):
        if answer["id"] in done:
            continue
        t0 = time.perf_counter()
        ours = await atomizer.atomize(answer["text"])  # timed only: scoring uses the labelled atoms
        t1 = time.perf_counter()
        kept = await collect(claim_filter(_stream(ours)))
        t2 = time.perf_counter()
        atoms, verify_s = await verify_answer(verify, answer)
        done[answer["id"]] = {**{k: answer[k] for k in ("id", "pair", "kind")}, "atoms": atoms,
                              "time": {"atomize": t1 - t0, "filter": t2 - t1, "verify": verify_s,
                                       "total": t1 - t0 + t2 - t1 + verify_s},
                              "our_atoms": len(ours), "our_kept": len(kept)}
        await fill_pages(hits, pages, base_resolver, base_crawler, answer["atoms"])
        _save(run_dir / "hits.json.gz", hits)
        _save(run_dir / "pages.json.gz", pages)
        _save(run_dir / "results.json.gz", list(done.values()))
        t = done[answer["id"]]["time"]
        s = sum(a["verdict"] == "supported" for a in atoms)
        print(f"[{n:3d}/{len(answers)}] {answer['id']}: {len(atoms)} atoms, {s} supported; atomize {t['atomize']:.1f}s "
              f"filter {t['filter']:.2f}s verify {t['verify']:.1f}s = {t['total']:.1f}s", flush=True)

    await base_crawler.aclose()
    if base_resolver:
        await base_resolver.aclose()
    report(list(done.values()), args.tag)


async def _stream(items: list[Any]) -> Any:
    for item in items:
        yield item


async def replay(args: argparse.Namespace) -> None:
    """The cached evidence of run `--tag`, judged again with another setup; saved as run `--out`."""
    run_dir = OUT / args.tag
    hits, pages = _load(run_dir / "hits.json.gz", {}), _load(run_dir / "pages.json.gz", {})
    live_results = _load(run_dir / "results.json.gz", [])
    judge = LayaJudge(passages_per_page=args.passages)
    verify = Verify(CachedSearch(hits), CachedCrawler(pages), judge, WeightedPolicy(strong=args.strong), timeout=120)
    out = []
    start = time.perf_counter()
    for r in live_results:
        atoms, verify_s = await verify_answer(verify, r)
        out.append({**{k: r[k] for k in ("id", "pair", "kind")}, "atoms": atoms, "time": {"verify": verify_s}})
    print(f"replayed {len(out)} answers in {time.perf_counter() - start:.0f}s")
    _save(OUT / args.out / "results.json.gz", out)
    report(out, args.out)


# --- report -------------------------------------------------------------------------------------------------------

def _pct(values: list[float], q: float) -> float:
    return statistics.quantiles(values, n=100)[int(q) - 1] if len(values) > 1 else (values[0] if values else 0.0)


def report(results: list[dict[str, Any]], tag: str) -> None:
    mode = "in-domain" if "in-domain" in tag else "web"
    print(f"\n== {tag}: {len(results)} answers, {sum(len(r['atoms']) for r in results)} atoms")
    print(f"{'system':28s} {'split':10s} {'n':>4s} {'Acc':>6s} {'Prec':>6s} {'Rec':>6s} {'F1':>6s}")
    for split in ("corrupted", "combined"):
        ref = REFERENCE.get((mode, split))
        if ref:
            print(f"{'reference (paper Table 3)':28s} {split:10s} {'':>4s} " + " ".join(f"{x:6.3f}" for x in ref))
        acc, prec, rec, f1, n = macro(results, split, lambda a: True)
        print(f"{'always S':28s} {split:10s} {n:4d} {acc:6.3f} {prec:6.3f} {rec:6.3f} {f1:6.3f}")
        acc, prec, rec, f1, n = macro(results, split, lambda a: a["verdict"] == "supported")
        print(f"{'FactAssessor':28s} {split:10s} {n:4d} {acc:6.3f} {prec:6.3f} {rec:6.3f} {f1:6.3f}")
    atoms = [a for r in results for a in r["atoms"]]
    verdicts: dict[str, int] = {}
    for a in atoms:
        verdicts[a["verdict"]] = verdicts.get(a["verdict"], 0) + 1
    ns = [a for a in atoms if a["label"] == "NS"]
    print(f"verdicts: {verdicts}; false atoms caught (not supported): {sum(a['verdict'] != 'supported' for a in ns)}/{len(ns)}; "
          f"timeouts: {sum(a.get('error') == 'timeout' for a in atoms)}")
    comps = ["search", "judge_snippets", "resolve", "crawl", "judge_pages", "total"]
    print(f"\nper claim (s)   {'':6s}" + "".join(f"{c:>15s}" for c in comps))
    for q in (50, 90, 95):
        vals = [_pct([a["time"].get(c, 0.0) for a in atoms], q) for c in comps]
        print(f"  p{q:<13d}{'':6s}" + "".join(f"{v:15.2f}" for v in vals))
    print(f"  used (share)  {'':6s}" + "".join(f"{sum(c in a['time'] for a in atoms) / len(atoms):15.0%}" for c in comps))
    if all("atomize" in r["time"] for r in results):
        parts = ["atomize", "filter", "verify", "total"]
        print(f"\nper answer (s)  {'':6s}" + "".join(f"{c:>15s}" for c in parts))
        for q in (50, 90, 95):
            vals = [_pct([r["time"][c] for r in results], q) for c in parts]
            print(f"  p{q:<13d}{'':6s}" + "".join(f"{v:15.2f}" for v in vals))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    lv = sub.add_parser("live", help="live run: search, resolve, crawl, judge; caches evidence")
    lv.add_argument("--data", required=True)
    lv.add_argument("--tag", default="web")
    lv.add_argument("--limit", type=int, help="first N pairs")
    lv.add_argument("--searxng", default="http://localhost:8080")
    lv.add_argument("--search-type", default="general", choices=["general", "science"])
    lv.add_argument("--atomizer", default="openai:gpt-6-luna", help="pydantic-ai model for the atomizer (timed only)")
    lv.add_argument("--no-resolver", action="store_true")
    lv.add_argument("--passages", type=int, default=1, help="passages per page for the judge")
    lv.add_argument("--strong", type=float, default=0.7)
    rp = sub.add_parser("replay", help="judge the cached evidence of --tag again, with another setup")
    rp.add_argument("--tag", default="web")
    rp.add_argument("--out", required=True)
    rp.add_argument("--passages", type=int, default=1)
    rp.add_argument("--strong", type=float, default=0.7)
    rt = sub.add_parser("report", help="metrics and timings of a run")
    rt.add_argument("--tag", default="web")
    args = ap.parse_args()
    if args.cmd == "live":
        asyncio.run(live(args))
    elif args.cmd == "replay":
        asyncio.run(replay(args))
    else:
        report(_load(OUT / args.tag / "results.json.gz", []), args.tag)


if __name__ == "__main__":
    main()
