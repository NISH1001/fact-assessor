"""Atom-level eval on a labelled long-form set (SciELF-style JSONL): the paper's metric, with timings.

Each JSONL row is a pair: an `original` and a `corrupted` long-form answer, each with human-labelled atoms
(`{"text", "label": "S" | "NS"}`). The data is external: keep it in tmp/ (gitignored).

    # live: gpt-6-luna atomizer + Laya filter (timed per answer), then every labelled atom verified live
    # (SearXNG, resolvers, HTTPX -> browser, Laya); search hits and pages are cached for replays. The atomizer
    # also writes the text's source query (--no-source-query to skip), searched once per text for every atom
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
import math
import statistics
import time
from pathlib import Path
from typing import Any

from factassessor import (
    ArxivResolver, Atom, CompositeResolver, Crawl4AICrawler, Crawler, DecisionClaimFilter, DecisionJudge, FallbackCrawler,
    HTTPXCrawler, LayaRunner, LLMAtomizer, OpenAlexResolver, DecisionPacking, SearxngSearcher, SerperSearcher, Step,
    SystemOneRunner, Take, Verify, WeightedPolicy, collect, not_blocked, once,
)
from factassessor.rankers import HybridRanker
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

def load_answers(path: str, manual_only: bool = True, limit: int | None = None, sample: int | None = None,
                 seed: int = 0) -> list[dict[str, Any]]:
    """One entry per answer: {id, pair, kind (original | corrupted), text, atoms [{text, label}]}. Ids come from
    the pair's position in the file, so a `sample` or `limit` run and a full run name the same answers the same."""
    import random

    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    if manual_only:
        rows = [r for r in rows if not r.get("is_synthetic")]
    indexed = list(enumerate(rows))
    if limit:
        indexed = indexed[:limit]
    if sample:
        indexed = sorted(random.Random(seed).sample(indexed, sample))
    answers = []
    for i, r in indexed:
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
    (the wait counts as search time: it's what a self-hosted searcher costs). `save` runs after every search, so a
    paid-for result is on disk before the answer it belongs to finishes."""

    def __init__(self, inner: Step, cache: dict[str, list[dict[str, Any]]], slots: int = 4, save: Any = None) -> None:
        self.inner, self.cache, self._slots, self.save = inner, cache, asyncio.Semaphore(slots), save
        self._inflight: dict[str, asyncio.Task[list[dict[str, Any]]]] = {}  # a text's claims share its source query

    async def __call__(self, queries: Any) -> Any:
        async for q in queries:
            if q in self.cache:
                hits = self.cache[q]
            else:
                leader = q not in self._inflight
                if leader:  # its own task: a claim cancelled at its deadline must not kill a search others share
                    self._inflight[q] = asyncio.create_task(self._search(q))
                start = time.perf_counter()
                hits = await asyncio.shield(self._inflight[q])
                if not leader:  # the leader's own wait and request are recorded by _search, in its context
                    _record("search_wait", start, time.perf_counter())
            for hit in hits:
                yield hit

    async def _search(self, q: str) -> list[dict[str, Any]]:
        start = time.perf_counter()
        async with self._slots:
            got_slot = time.perf_counter()
            try:
                hits = await collect(self.inner(once(q)))
            except Exception:
                hits = []
        _record("search_wait", start, got_slot)  # queueing for a SearXNG slot: not the search itself
        _record("search", got_slot, time.perf_counter())
        self.cache[q] = hits
        self._inflight.pop(q, None)
        if self.save:
            self.save()
        return hits


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
    """Hits as the live run got them; `limit` caps every query (a smaller overfetch than the run had, e.g. 5 of
    its 10 hits) and `caps` single queries (fewer source-query hits)."""

    def __init__(self, cache: dict[str, list[dict[str, Any]]], caps: dict[str, int] | None = None, limit: int | None = None) -> None:
        self.cache, self.caps, self.limit = cache, caps or {}, limit

    async def __call__(self, queries: Any) -> Any:
        async for q in queries:
            hits = self.cache.get(q, [])
            for hit in hits[: min(self.caps.get(q, len(hits)), self.limit or len(hits))]:
                yield hit


class CachedCrawler(Crawler):
    """Pages as the live run read them (after resolving), by hit URL; None for hits that couldn't be read."""

    def __init__(self, pages: dict[str, Any]) -> None:
        self.pages = pages

    async def crawl(self, url: str) -> dict[str, Any] | None:
        return self.pages.get(url)


# --- runs ---------------------------------------------------------------------------------------------------------

def make_judge(args: argparse.Namespace) -> DecisionJudge:
    """The same judge on `--judge laya` (local, the default) or `--judge decision` (Jev on OpenRouter: `--judge-model`,
    `--batch-size` requests per call); the live run's claim filter shares the judge's runner."""
    ranker = HybridRanker(alpha=args.alpha) if args.ranker == "hybrid" else None
    if args.judge == "decision":
        runner: Any = SystemOneRunner(model=args.judge_model, packing=args.pack, **({"batch_size": args.batch_size} if args.batch_size else {}))
    else:
        runner = LayaRunner(**({"max_wait_ms": args.laya_wait_ms} if args.laya_wait_ms else {}))
    return DecisionJudge(runner, passages_per_page=args.passages, passage_words=args.passage_words, ranker=ranker)


def _cost(judge: DecisionJudge) -> str:
    return f"; API cost ${judge.runner.cost:.4f}" if hasattr(judge.runner, "cost") else ""


def _load(path: Path, default: Any) -> Any:
    return json.loads(gzip.decompress(path.read_bytes())) if path.exists() else default


def _save(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(gzip.compress(json.dumps(data, ensure_ascii=False).encode()))


def _write_meta(run_dir: Path, args: argparse.Namespace) -> None:
    """When, which code, which settings: `meta.json` next to a run's results (appended per invocation)."""
    import subprocess
    import sys
    from datetime import datetime, timezone

    def git(*cmd: str) -> str:
        try:
            return subprocess.run(["git", *cmd], capture_output=True, text=True, check=True).stdout.strip()
        except Exception:
            return "unknown"

    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / "meta.json"
    meta = json.loads(path.read_text()) if path.exists() else {"runs": []}
    meta.setdefault("runs", []).append({
        "when": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "commit": git("rev-parse", "--short", "HEAD") + (" (dirty)" if git("status", "--porcelain") else ""),
        "argv": sys.argv[1:],
        "args": {k: v for k, v in vars(args).items() if k != "cmd"},
    })
    path.write_text(json.dumps(meta, indent=1))


def _atom_record(atom: dict[str, Any], result: Any, spans: dict[str, list[tuple[float, float]]], total: float) -> dict:
    return {**atom, "verdict": result.verdict, "confidence": result.confidence, "error": result.error,
            "evidence": [e.model_dump() for e in result.evidence],
            "time": {**{k: busy(v) for k, v in spans.items()}, "total": total}}


async def verify_answer(verify: Verify, answer: dict[str, Any]) -> tuple[list[dict[str, Any]], float]:
    """Every atom of an answer at once, as the pipeline checks a text's claims; per-atom spans and totals.
    `answer["source_query"]` (the atomizer's search for the text's source document) goes on every atom."""

    async def one(a: dict[str, Any]) -> dict[str, Any]:
        spans: dict[str, list[tuple[float, float]]] = {}
        _spans.set(spans)  # this task's context only: each atom's own spans
        start = time.perf_counter()
        atom = Atom(id=0, text=a["text"], span=(0, len(a["text"])), source_query=answer.get("source_query"))
        result = await verify.verify(atom)
        return _atom_record(a, result, spans, time.perf_counter() - start)

    start = time.perf_counter()
    atoms = await asyncio.gather(*(one(a) for a in answer["atoms"]))
    return list(atoms), time.perf_counter() - start


async def fill_pages(cache_hits: dict[str, Any], pages: dict[str, Any], resolver: Any, crawler: Crawler,
                     queries: list[str], min_copy_words: int = 300) -> None:
    """Read the hits the live run skipped (early exit), so replays with another judge see every page. Untimed."""
    from factassessor.verify import read_first

    todo = {h["url"] for q in queries for h in cache_hits.get(q, [])} - set(pages)
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
    answers = load_answers(args.data, limit=args.limit, sample=args.sample, seed=args.seed)
    run_dir = OUT / args.tag
    hits: dict[str, Any] = _load(run_dir / "hits.json.gz", {})
    pages: dict[str, Any] = _load(run_dir / "pages.json.gz", {})
    done = {r["id"]: r for r in _load(run_dir / "results.json.gz", [])}

    atomizer = LLMAtomizer(args.atomizer, source_query=not args.no_source_query)
    judge = make_judge(args)
    claim_filter = DecisionClaimFilter(judge.runner)  # timed only; the same model as the judge
    if args.searcher == "serper":  # 10 results = 1 credit; blocked hosts excluded in the query, so all 10 are usable
        base_search: Step = SerperSearcher(num=10, hedge_after=None, search_type=args.search_type)
    else:
        base_search = SearxngSearcher(args.searxng, num=2 * TOP_K, timeout=20.0, search_type=args.search_type, hedge_after=None)
    searcher = base_search >> not_blocked() >> Take(math.ceil(TOP_K * (1 + args.overfetch)))
    base_resolver = None if args.no_resolver else CompositeResolver(ArxivResolver(), OpenAlexResolver())
    base_crawler = FallbackCrawler(HTTPXCrawler(), Crawl4AICrawler(timeout=2.5))
    save_hits = lambda: _save(run_dir / "hits.json.gz", hits)  # noqa: E731  # every paid search on disk at once
    verify = Verify(TimedSearch(searcher, hits, save=save_hits), TimedCrawler(base_crawler), TimedJudge(judge, pages),
                    WeightedPolicy(strong=args.strong), timeout=args.timeout,
                    resolver=TimedResolver(base_resolver) if base_resolver else None,
                    pages_per_claim=TOP_K * (1 if args.no_source_query else 2) if args.overfetch else None)
    await judge.aload()  # the model (or the HTTP pool) before timing
    _write_meta(run_dir, args)
    slots = asyncio.Semaphore(args.parallel)  # answers at once; > 1 contaminates per-answer latency (flagged)
    todo = [a for a in answers if a["id"] not in done]

    async def one(n: int, answer: dict[str, Any]) -> None:
        async with slots:
            t0 = time.perf_counter()
            ours = await atomizer.atomize(answer["text"])  # timed only: scoring uses the labelled atoms
            t1 = time.perf_counter()
            kept = await collect(claim_filter(_stream(ours)))
            t2 = time.perf_counter()
            answer["source_query"] = ours[0].source_query if ours else None  # from the same atomizer call
            searches_cached = all(a["text"] in hits for a in answer["atoms"])  # an earlier run searched them: not live
            atoms, verify_s = await verify_answer(verify, answer)
            done[answer["id"]] = {**{k: answer[k] for k in ("id", "pair", "kind", "source_query")}, "atoms": atoms,
                                  "time": {"atomize": t1 - t0, "filter": t2 - t1, "verify": verify_s,
                                           "total": t1 - t0 + t2 - t1 + verify_s},
                                  "our_atoms": len(ours), "our_kept": len(kept),
                                  "searches_cached": searches_cached or args.parallel > 1}
            queries = [a["text"] for a in answer["atoms"]] + ([answer["source_query"]] if answer["source_query"] else [])
            await fill_pages(hits, pages, base_resolver, base_crawler, queries)
            _save(run_dir / "hits.json.gz", hits)  # synchronous dumps: no other answer mutates the dicts meanwhile
            _save(run_dir / "pages.json.gz", pages)
            _save(run_dir / "results.json.gz", list(done.values()))
            t = done[answer["id"]]["time"]
            s = sum(a["verdict"] == "supported" for a in atoms)
            print(f"[{n:3d}/{len(answers)}] {answer['id']}: {len(atoms)} atoms, {s} supported; atomize {t['atomize']:.1f}s "
                  f"filter {t['filter']:.2f}s verify {t['verify']:.1f}s = {t['total']:.1f}s", flush=True)

    await asyncio.gather(*(one(n, a) for n, a in enumerate(todo, 1 + len(answers) - len(todo))))

    await base_crawler.aclose()
    if base_resolver:
        await base_resolver.aclose()
    await judge.aclose()
    report(list(done.values()), args.tag)
    print(f"judge: {args.judge}{_cost(judge)}")


async def _stream(items: list[Any]) -> Any:
    for item in items:
        yield item


async def replay(args: argparse.Namespace) -> None:
    """The cached evidence of run `--tag`, judged again with another setup; saved as run `--out`."""
    run_dir = OUT / args.tag
    hits, pages = _load(run_dir / "hits.json.gz", {}), _load(run_dir / "pages.json.gz", {})
    live_results = sorted(_load(run_dir / "results.json.gz", []), key=lambda r: r["id"])[: args.limit or None]
    judge = make_judge(args)
    await judge.aload()
    source_queries = [r["source_query"] for r in live_results if r.get("source_query")]
    caps = {q: args.source_hits for q in source_queries} if args.source_hits else {}
    verify = Verify(CachedSearch(hits, caps, limit=args.hits), CachedCrawler(pages), judge, WeightedPolicy(strong=args.strong), timeout=120,
                    pages_per_claim=args.pages_per_claim)
    start = time.perf_counter()
    slots = asyncio.Semaphore(args.parallel)  # answers at once: several answers' passages fill the runner's batches

    done = 0

    async def one(r: dict[str, Any]) -> dict[str, Any]:
        nonlocal done
        if args.no_source_query:
            r = {**r, "source_query": None}
        async with slots:
            atoms, verify_s = await verify_answer(verify, r)
        done += 1
        print(f"[{done:3d}/{len(live_results)}] {r['id']}: {len(atoms)} atoms, {sum(a['verdict'] == 'supported' for a in atoms)} supported; "
              f"{verify_s:.1f}s", flush=True)
        return {**{k: r[k] for k in ("id", "pair", "kind")}, "atoms": atoms, "time": {"verify": verify_s}}

    out = list(await asyncio.gather(*(one(r) for r in live_results)))
    await judge.aclose()
    print(f"replayed {len(out)} answers in {time.perf_counter() - start:.0f}s ({args.parallel} at a time; judge {args.judge}{_cost(judge)})")
    _write_meta(OUT / args.out, args)
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
    comps = ["search_wait", "search", "judge_snippets", "resolve", "crawl", "judge_pages", "total"]
    print(f"\nper claim (s)   {'':6s}" + "".join(f"{c:>15s}" for c in comps))
    for q in (50, 90, 95):
        vals = [_pct([a["time"].get(c, 0.0) for a in atoms], q) for c in comps]
        print(f"  p{q:<13d}{'':6s}" + "".join(f"{v:15.2f}" for v in vals))
    print(f"  used (share)  {'':6s}" + "".join(f"{sum(c in a['time'] for a in atoms) / len(atoms):15.0%}" for c in comps))
    # a corrupted answer shares most atoms with its original, whose searches are then already cached: only the
    # originals' times are live end to end
    fresh = [r for r in results if r["kind"] == "original" and "atomize" in r["time"] and not r.get("searches_cached")]
    if fresh:
        parts = ["atomize", "filter", "verify", "total"]
        print(f"\nper answer (s), {len(fresh)} originals (live searches)\n{'':21s}" + "".join(f"{c:>15s}" for c in parts))
        for q in (50, 90, 95):
            vals = [_pct([r["time"][c] for r in fresh], q) for c in parts]
            print(f"  p{q:<13d}{'':6s}" + "".join(f"{v:15.2f}" for v in vals))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    lv = sub.add_parser("live", help="live run: search, resolve, crawl, judge; caches evidence")
    lv.add_argument("--data", required=True)
    lv.add_argument("--tag", default="web")
    lv.add_argument("--limit", type=int, help="first N pairs")
    lv.add_argument("--sample", type=int, help="N random pairs (same ids as a full run, which then skips them)")
    lv.add_argument("--seed", type=int, default=0)
    lv.add_argument("--searcher", default="searxng", choices=["searxng", "serper"], help="serper: Google via Serper, 1 credit per query")
    lv.add_argument("--searxng", default="http://localhost:8080")
    lv.add_argument("--search-type", default="general", choices=["general", "science"])
    lv.add_argument("--atomizer", default="openai:gpt-6-luna", help="pydantic-ai model for the atomizer (timed only)")
    lv.add_argument("--no-resolver", action="store_true")
    lv.add_argument("--no-source-query", action="store_true", help="claims search on their own only")
    lv.add_argument("--parallel", type=int, default=1, help="answers at once (> 1: per-answer latency not comparable)")
    lv.add_argument("--overfetch", type=float, default=0.0,
                    help="keep this fraction more hits than pages (1.0: 10 hits for 5 pages); the first 5 readable are judged")
    lv.add_argument("--passages", type=int, default=1, help="passages per page for the judge")
    lv.add_argument("--strong", type=float, default=0.7)
    lv.add_argument("--timeout", type=float, default=15.0, help="per-claim deadline (the library default is 15s)")
    lv.add_argument("--ranker", default="bm25", choices=["bm25", "hybrid"], help="which chunks of a page the judge sees")
    lv.add_argument("--alpha", type=float, default=0.5, help="hybrid ranker: BM25's weight (1 - alpha for embeddings)")
    rp = sub.add_parser("replay", help="judge the cached evidence of --tag again, with another setup")
    rp.add_argument("--tag", default="web")
    rp.add_argument("--out", required=True)
    rp.add_argument("--limit", type=int, help="first N answers only (a quick check of a setup)")
    rp.add_argument("--passages", type=int, default=1)
    rp.add_argument("--strong", type=float, default=0.7)
    rp.add_argument("--source-hits", type=int, help="use only the first N hits of each text's source query")
    rp.add_argument("--hits", type=int, help="use only the first N hits of every query (5: the run's evidence without overfetch)")
    rp.add_argument("--pages-per-claim", type=int, help="judge only the first N readable pages per claim (the live run's overfetch cap)")
    rp.add_argument("--no-source-query", action="store_true", help="claims judged on their own hits only")
    rp.add_argument("--ranker", default="bm25", choices=["bm25", "hybrid"], help="which chunks of a page the judge sees")
    rp.add_argument("--alpha", type=float, default=0.5, help="hybrid ranker: BM25's weight (1 - alpha for embeddings)")
    rp.add_argument("--parallel", type=int, default=4, help="answers judged at once (a replay has no network: the judge's runner is the floor)")
    for p in (lv, rp):
        p.add_argument("--passage-words", type=int, default=90, help="words per page window (90 ~ 130 tokens on scientific text)")
        p.add_argument("--judge", default="laya", choices=["laya", "decision"], help="the judge's runner: local Laya, or Jev on OpenRouter")
        p.add_argument("--judge-model", default="~typesafe/jev-latest", help="model id for --judge decision")
        p.add_argument("--pack", default="call", choices=[p.value for p in DecisionPacking],
                       help="--judge decision: what shares a call: one claim's passages (call, the default), every claim in flight (all), nothing (none)")
        p.add_argument("--batch-size", type=int, help="--judge decision: requests packed per call (the runner's default is 40)")
        p.add_argument("--laya-wait-ms", type=float, help="--judge laya: the batch merge window (the runtime default is 5ms)")
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
