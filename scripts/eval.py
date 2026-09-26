"""End-to-end evaluation harness for FactAssessor on synthetic texts with known truth.

    uv run python scripts/eval.py build                          # data/fact_pairs.json -> data/eval_texts.jsonl
    uv run --extra ddg python scripts/eval.py record             # atomize, search, crawl once -> data/evidence/
    uv run --extra gliner python scripts/eval.py run all         # laya, gliner, llm on the recorded evidence + comparison
    uv run --extra ddg python scripts/eval.py run laya --live    # live atomizer/search/crawl (web results drift)
    uv run python scripts/eval.py report                         # rebuild the comparison + plots from data/results/

Texts: each pair in data/fact_pairs.json is a true sentence and a false variant with one detail changed (date,
number, place, person). `build` samples them into 27 texts = {true, false, mixed ~50/50} x {short 2-4, medium 5-10,
long 15-25 sentences} x 3 (seed 7), so every sentence has a label; an atom takes the label of the sentence its span
starts in.

Evidence: live web results change between runs, so `record` fixes the atoms (LLMAtomizer), the top 5 unblocked hits
per atom (DuckDuckGo by default; `--searcher searxng --searxng-url ...` or `--searcher serper`), and every hit's page (crawl4ai, live timeout, so pages that
time out live are missing here too). All atoms are searched, since claim filters differ per variant. Stored in
data/evidence/evidence.json.gz (gitignored: third-party page text). Resumes where it stopped.

Variants (only the models differ):
    laya    LayaClaimFilter + LayaJudge (the default pipeline)
    gliner  GlinerClaimFilter + GlinerJudge
    llm     no claim filter + LLMJudge (gpt-6-luna): everything after the atomizer is the LLM

Warm-up before every run, timed and reported separately. Texts run one at a time. On recorded evidence the atomizer,
search, and crawl return instantly, so latency is the claim filter + judge + policy (the models' cost); `--live`
measures the whole pipeline. Results: data/results/eval-<variant>[-live].{json,md}, eval-comparison.md, and plots
(eval-comparison.png, eval-verdicts.png) of the recorded runs.
"""

from __future__ import annotations

import argparse
import asyncio
import gzip
import json
import random
import statistics
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from factassessor import Atom, Crawl4AICrawler, FactAssessor, LayaClaimFilter, LayaJudge, LLMJudge, SerperSearcher
from factassessor.atomizer import Atomizer, LLMAtomizer
from factassessor.crawl import Crawler
from factassessor.search import DuckDuckGoSearcher, Searcher, SearxngSearcher, is_blocked, not_blocked
from factassessor.pipeline import Take

DATA = Path(__file__).parent.parent / "data"
TEXTS = DATA / "eval_texts.jsonl"
EVIDENCE = DATA / "evidence" / "evidence.json.gz"
RESULTS = DATA / "results"
LENGTHS = {"short": (2, 4), "medium": (5, 10), "long": (15, 25)}  # sentences per text
KINDS = ("true", "false", "mixed")
VARIANTS = {"laya": "LayaClaimFilter + LayaJudge", "gliner": "GlinerClaimFilter + GlinerJudge", "llm": "LLMJudge, no claim filter"}
LLM_MODEL = "openai:gpt-5-nano"  # the cheapest OpenAI model ($0.05 in / $0.40 out per 1M tokens, Sept 2026)
TOP_K = 5
WARM_TEXT = "The Moon orbits the Earth. Mount Fuji is the highest mountain in Japan."  # not in the eval set


# --- texts -----------------------------------------------------------------------------------------------------

def build_texts(pairs: list[dict[str, str]], seed: int = 7, per_cell: int = 3) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    texts = []
    for length, (lo, hi) in LENGTHS.items():
        for kind in KINDS:
            for i in range(per_cell):
                n = rng.randint(lo, hi)
                picked = rng.sample(pairs, n)  # no fact twice in one text
                if kind == "mixed":  # about half and half, at least one of each
                    k = max(1, min(n - 1, round(n / 2)))
                    truths = [True] * k + [False] * (n - k)
                    rng.shuffle(truths)
                else:
                    truths = [kind == "true"] * n
                sentences, offset = [], 0
                for pair, is_true in zip(picked, truths):
                    s = pair["true"] if is_true else pair["false"]
                    sentences.append({"text": s, "true": is_true, "span": [offset, offset + len(s)]})
                    offset += len(s) + 1  # the joining space
                texts.append({"id": f"{length}-{kind}-{i}", "kind": kind, "length": length,
                              "text": " ".join(s["text"] for s in sentences), "sentences": sentences})
    return texts


def load_texts() -> list[dict[str, Any]]:
    return [json.loads(line) for line in TEXTS.read_text().splitlines()]


def label_at(text: dict[str, Any], offset: int) -> bool | None:
    """Ground truth of the sentence containing char `offset` (an atom's span start)."""
    for s in text["sentences"]:
        if s["span"][0] <= offset < s["span"][1]:
            return s["true"]
    return None


# --- evidence: record once, replay per variant -----------------------------------------------------------------

def load_evidence() -> dict[str, Any]:
    if not EVIDENCE.exists():
        return {"searcher": None, "texts": {}, "pages": {}}
    return json.loads(gzip.decompress(EVIDENCE.read_bytes()))


def save_evidence(evidence: dict[str, Any]) -> None:
    EVIDENCE.parent.mkdir(exist_ok=True)
    EVIDENCE.write_bytes(gzip.compress(json.dumps(evidence, ensure_ascii=False).encode()))


def make_searcher(name: str, searxng_url: str) -> Searcher:
    """ddg (no key, default), searxng (self-hosted, no key), or serper (API key and credits)."""
    if name == "searxng":
        return SearxngSearcher(searxng_url, num=2 * TOP_K)
    if name == "serper":
        return SerperSearcher(num=2 * TOP_K)
    return DuckDuckGoSearcher(num=2 * TOP_K)


async def record(searcher_name: str, searxng_url: str, llm_model: str = LLM_MODEL) -> None:
    searcher = make_searcher(searcher_name, searxng_url)
    atomizer, crawler = LLMAtomizer(llm_model, model_settings=llm_settings(llm_model)), Crawl4AICrawler()
    await crawler.start()
    evidence = load_evidence()
    evidence["searcher"] = searcher_name
    slots = asyncio.Semaphore(3)  # DuckDuckGo rate-limits bursts

    async def search(query: str) -> list[dict[str, Any]]:
        for attempt in range(4):
            try:
                async with slots:
                    hits = await searcher.search(query)
                return [h for h in hits if not is_blocked(h["url"])][:TOP_K]
            except Exception as exc:
                print(f"  search retry {attempt + 1} ({exc!r:.60})", flush=True)
                await asyncio.sleep(2 * (attempt + 1))
        return []

    for ex in load_texts():
        if ex["id"] in evidence["texts"]:
            continue
        start = time.perf_counter()
        atoms = await atomizer.atomize(ex["text"])
        hits = await asyncio.gather(*(search(a.text) for a in atoms))
        urls = list({h["url"] for hs in hits for h in hs} - evidence["pages"].keys())
        pages = await asyncio.gather(*(crawler.crawl(u) for u in urls))
        evidence["pages"].update(zip(urls, pages))
        evidence["texts"][ex["id"]] = {"atoms": [{"text": a.text, "span": list(a.span)} for a in atoms],
                                       "hits": {a.text: hs for a, hs in zip(atoms, hits)}}
        save_evidence(evidence)
        print(f"{ex['id']:16s} {len(atoms):2d} atoms, {sum(map(len, hits)):3d} hits, "
              f"{sum(p is not None for p in pages)}/{len(urls)} new pages, {time.perf_counter() - start:.1f}s", flush=True)
    await crawler.stop()


class RecordedAtomizer(Atomizer):
    def __init__(self, evidence: dict[str, Any]) -> None:
        by_id = {ex["id"]: ex["text"] for ex in load_texts()}
        self.atoms = {by_id[i]: t["atoms"] for i, t in evidence["texts"].items()}

    async def atomize(self, text: str) -> list[Atom]:
        return [Atom(id=i, text=a["text"], span=tuple(a["span"])) for i, a in enumerate(self.atoms[text])]


class RecordedSearcher(Searcher):
    def __init__(self, evidence: dict[str, Any]) -> None:
        self.hits = {q: hs for t in evidence["texts"].values() for q, hs in t["hits"].items()}

    async def search(self, query: str) -> list[dict[str, Any]]:
        return self.hits.get(query, [])


class RecordedCrawler(Crawler):
    def __init__(self, evidence: dict[str, Any]) -> None:
        self.pages = evidence["pages"]

    async def crawl(self, url: str) -> dict[str, Any] | None:
        return self.pages.get(url)


# --- runs ------------------------------------------------------------------------------------------------------

def llm_settings(model: str) -> dict[str, Any]:
    """Reasoning off (billed as output tokens); gpt-5 models take "minimal", newer ones "none"."""
    return {"openai_reasoning_effort": "minimal" if model.startswith("openai:gpt-5-") else "none"}


def components(variant: str, llm_model: str = LLM_MODEL) -> tuple[Any, Any]:
    """(claim_filter, judge) for a variant."""
    if variant == "gliner":
        from factassessor.gliner import GlinerClaimFilter, GlinerJudge  # one shared GLiNER model

        return GlinerClaimFilter(), GlinerJudge()
    if variant == "llm":
        return None, LLMJudge(llm_model, model_settings=llm_settings(llm_model))
    return LayaClaimFilter(), LayaJudge()


async def warm_up(fa: FactAssessor, claim_filter: Any, judge: Any, text: str) -> dict[str, Any]:
    """Load and exercise every component once; return the timings (seconds)."""
    t: dict[str, Any] = {}
    start = time.perf_counter()
    await fa.aload()
    t["load"] = time.perf_counter() - start
    atom = Atom(id=0, text="The Moon orbits the Earth.", span=(0, 26))
    for name, call in [
        ("claim_filter", (lambda: claim_filter.score(atom)) if claim_filter else None),
        ("judge", lambda: judge.judge(atom.text, [{"url": "warmup", "title": "", "snippet": "The Moon orbits the Earth."}])),
    ]:
        if call is None:
            continue
        for when in ("first", "warm"):
            start = time.perf_counter()
            await call()
            t[f"{name}_{when}"] = time.perf_counter() - start
    t["assess"] = []
    for _ in range(3):  # until a full check is no slower than the one before (connections, browser, LLM client)
        start = time.perf_counter()
        await fa.assess(text)
        t["assess"].append(time.perf_counter() - start)
        if len(t["assess"]) > 1 and t["assess"][-1] <= t["assess"][-2] * 1.2:
            break
    return t


async def run(variant: str, live: bool, timeout: float = 15.0, searcher: str = "ddg", searxng_url: str = "",
              llm_model: str = LLM_MODEL) -> dict[str, Any]:
    claim_filter, judge = components(variant, llm_model)
    texts = load_texts()
    if live:
        search = make_searcher(searcher, searxng_url) >> not_blocked() >> Take(TOP_K)  # as FactAssessor wires Serper
        fa = FactAssessor(n_atoms=30, claim_filter=claim_filter, judge=judge, timeout=timeout, searcher=search,
                          atomizer=LLMAtomizer(llm_model, model_settings=llm_settings(llm_model)))
        warm_text, source = WARM_TEXT, f"live (LLMAtomizer, {searcher} search, crawl4ai)"
    else:
        evidence = load_evidence()
        missing = [ex["id"] for ex in texts if ex["id"] not in evidence["texts"]]
        if missing:
            raise SystemExit(f"no recorded evidence for {len(missing)} texts (run `eval.py record` first): {missing[:3]}...")
        fa = FactAssessor(n_atoms=30, claim_filter=claim_filter, judge=judge, timeout=timeout, atomizer=RecordedAtomizer(evidence),
                          searcher=RecordedSearcher(evidence), crawler=RecordedCrawler(evidence))
        warm_text, source = texts[0]["text"], f"recorded ({evidence['searcher']} search, crawl4ai)"
    warm = await warm_up(fa, claim_filter, judge, warm_text)
    print(f"[{variant}] warm-up: " + ", ".join(f"{k} {v:.2f}s" if isinstance(v, float) else f"{k} {[round(x, 1) for x in v]}s"
                                             for k, v in warm.items()), flush=True)
    rows = []
    for ex in texts:
        start, first, result = time.perf_counter(), None, None
        async for event in fa.stream(ex["text"]):
            if event.type == "claim_verified" and first is None:
                first = time.perf_counter() - start
            if event.type == "done":
                result = event.result
        total = time.perf_counter() - start
        assert result is not None
        atoms = [
            {"text": a.atom.text, "gold": label_at(ex, a.atom.span[0]), "verdict": a.verdict, "confidence": a.confidence,
             "error": a.error, "pages": sum(e.source == "page" for e in a.evidence)}
            for a in result.atoms
        ] + [{"text": s.text, "gold": label_at(ex, s.span[0]), "verdict": "skipped", "confidence": 0.0, "error": None, "pages": 0}
             for s in result.skipped]
        true_share = sum(s["true"] for s in ex["sentences"]) / len(ex["sentences"])
        rows.append({"id": ex["id"], "kind": ex["kind"], "length": ex["length"], "sentences": len(ex["sentences"]),
                     "true_share": true_share, "fact_score": result.fact_score, "latency_s": total,
                     "first_verdict_s": first, "atoms": atoms})
        decided = [a for a in atoms if a["verdict"] in ("supported", "refuted")]
        right = sum((a["verdict"] == "supported") == a["gold"] for a in decided)
        score = "–" if result.fact_score is None else f"{result.fact_score:.2f}"
        print(f"[{variant}] {ex['id']:16s} {len(ex['sentences']):2d} sent -> {len(atoms):2d} atoms | {right}/{len(decided)} right "
              f"of decided | score {score} vs {true_share:.2f} | first {first or 0:.1f}s total {total:.1f}s", flush=True)
    await fa.aclose()
    return {"variant": variant, "live": live, "source": source, "timeout": timeout,
            "llm_model": llm_model if variant == "llm" or live else None, "warmup": warm, "texts": rows}


# --- reports ---------------------------------------------------------------------------------------------------

def claim_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    atoms = [a for r in rows for a in r["atoms"]]
    labelled = [a for a in atoms if a["gold"] is not None]
    decided = [a for a in labelled if a["verdict"] in ("supported", "refuted")]
    correct = [a for a in decided if (a["verdict"] == "supported") == a["gold"]]
    false_claims = [a for a in labelled if a["gold"] is False]
    true_claims = [a for a in labelled if a["gold"] is True]
    pct = lambda n, d: n / d if d else None  # noqa: E731
    errors = [abs(r["fact_score"] - r["true_share"]) for r in rows if r["fact_score"] is not None]
    return {
        "claims": len(labelled),
        "coverage": pct(len(decided), len(labelled)),  # decided = supported or refuted
        "accuracy_decided": pct(len(correct), len(decided)),
        "accuracy_all": pct(len(correct), len(labelled)),  # contested/unverified/skipped count as wrong
        "false_supported": pct(sum(a["verdict"] == "supported" for a in false_claims), len(false_claims)),  # the dangerous error
        "true_refuted": pct(sum(a["verdict"] == "refuted" for a in true_claims), len(true_claims)),
        "score_error": statistics.mean(errors) if errors else None,
        "latency": statistics.median(r["latency_s"] for r in rows),
        "first": statistics.median(r["first_verdict_s"] or 0 for r in rows),
    }


def groups(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    out = {"all": rows}
    out.update({f"{k} ({lo}–{hi} sent.)": [r for r in rows if r["length"] == k] for k, (lo, hi) in LENGTHS.items()})
    out.update({f"{k} texts": [r for r in rows if r["kind"] == k] for k in KINDS})
    return out


def fmt(x: Any, kind: str = "pct") -> str:
    if x is None:
        return "–"
    return {"pct": f"{x:.0%}", "s": f"{x:.1f}s", "f": f"{x:.2f}"}[kind]


COLUMNS = [("claims", "claims", "n"), ("coverage", "coverage", "pct"), ("accuracy_decided", "accuracy (decided)", "pct"),
           ("accuracy_all", "accuracy (all)", "pct"), ("false_supported", "false → supported", "pct"),
           ("true_refuted", "true → refuted", "pct"), ("score_error", "fact-score error", "f"),
           ("latency", "latency (median)", "s"), ("first", "first verdict (median)", "s")]


def table(rows_by_name: dict[str, list[dict[str, Any]]], first_col: str) -> list[str]:
    lines = [f"| {first_col} | " + " | ".join(c[1] for c in COLUMNS) + " |", "|---" * (len(COLUMNS) + 1) + "|"]
    for name, rows in rows_by_name.items():
        m = claim_metrics(rows)
        lines.append(f"| {name} | " + " | ".join(str(m[k]) if kind == "n" else fmt(m[k], kind) for k, _, kind in COLUMNS) + " |")
    return lines + [""]


def verdict_table(rows: list[dict[str, Any]]) -> list[str]:
    counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for a in (a for r in rows for a in r["atoms"] if a["gold"] is not None):
        counts["true" if a["gold"] else "false"][a["verdict"]] += 1
    lines = ["| gold | supported | refuted | contested | unverified | skipped |", "|---|---|---|---|---|---|"]
    for gold in ("true", "false"):
        v = counts[gold]
        lines.append(f"| {gold} | {v['supported']} | {v['refuted']} | {v['contested']} | {v['unverified']} | {v['skipped']} |")
    return lines + [""]


def warmup_line(w: dict[str, Any]) -> str:
    parts = [f"load {w['load']:.1f}s"]
    for name in ("claim_filter", "judge"):
        if f"{name}_first" in w:
            parts.append(f"{name.replace('_', ' ')} {w[f'{name}_first']:.2f}s then {w[f'{name}_warm']:.2f}s")
    return ", ".join(parts) + f", full checks {', '.join(f'{x:.1f}s' for x in w['assess'])}"


def report(res: dict[str, Any]) -> str:
    rows = res["texts"]
    model = f" ({res['llm_model']})" if res.get("llm_model") else ""
    lines = [f"# FactAssessor eval: {VARIANTS[res['variant']]}{model}", "",
             f"{len(rows)} texts, {sum(r['sentences'] for r in rows)} sentences, {sum(len(r['atoms']) for r in rows)} atoms. "
             f"Evidence: {res['source']}. Per-claim timeout {res.get('timeout', 15.0):.0f}s. Warm-up (not counted): {warmup_line(res['warmup'])}.", "",
             *table(groups(rows), "group"), "## Verdicts by ground truth", "", *verdict_table(rows)]
    return "\n".join(lines)


def comparison() -> str:
    runs = {p.stem.removeprefix("eval-"): json.loads(p.read_text()) for p in sorted(RESULTS.glob("eval-*.json"))}
    recorded = {k: r for k, r in runs.items() if not r.get("live")}
    live = {k: r for k, r in runs.items() if r.get("live")}
    lines = ["# FactAssessor eval: comparison", "",
             "Same 27 synthetic texts (282 sentences, one fact each; true, false, and mixed texts; 2–25 sentences). "
             "Recorded runs share the exact same atoms, hits, and pages, so differences come from the claim filter and "
             "judge alone, and latency is their cost (atomizer, search, and crawl replay instantly). Live runs include "
             "the whole pipeline but see different web results.", ""]
    for title, rs in (("Recorded evidence", recorded), ("Live", live)):
        if not rs:
            continue
        lines.extend([f"## {title}", "", f"Evidence: {next(iter(rs.values()))['source']}.", ""])
        for group in groups(next(iter(rs.values()))["texts"]):
            lines.extend([f"### {group}", "", *table({f"{k} ({VARIANTS[r['variant']]}{', ' + r['llm_model'] if r.get('llm_model') else ''})": groups(r["texts"])[group]
                                                      for k, r in rs.items()}, "variant")])
        lines.extend(["### Warm-up (not counted above)", ""] + [f"- {k}: {warmup_line(r['warmup'])}" for k, r in rs.items()] + [""])
        for k, r in rs.items():
            lines.extend([f"### Verdicts by ground truth: {k}", "", *verdict_table(r["texts"])])
    return "\n".join(lines)


def plot() -> None:
    """data/results/eval-comparison.png (accuracy, errors, latency by length and composition) and
    eval-verdicts.png (verdict mix per ground truth), for the recorded runs."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    runs = [{**json.loads(p.read_text()), "key": p.stem.removeprefix("eval-")} for p in sorted(RESULTS.glob("eval-*.json"))]
    runs = [r for r in runs if not r.get("live")]
    if not runs:
        return
    names = [r["key"] for r in runs]
    palette = ("#4C72B0", "#DD8452", "#55A868", "#8172B3", "#937860", "#DA8BC3")
    colors = dict(zip(names, palette))
    width = 0.8 / len(runs)

    def bars(ax: Any, cats: list[str], value: Any, title: str, pct: bool = True) -> None:
        for i, r in enumerate(runs):
            vals = [value(r, c) for c in cats]
            xs = [j + (i - (len(runs) - 1) / 2) * width for j in range(len(cats))]
            b = ax.bar(xs, [v or 0 for v in vals], width, label=r["key"], color=colors[r["key"]])
            ax.bar_label(b, labels=[fmt(v, "pct" if pct else "s") for v in vals], fontsize=7, padding=1)
        ax.set_xticks(range(len(cats)), cats, fontsize=8)
        ax.set_title(title, fontsize=10)
        values = [v for r in runs for c in cats if (v := value(r, c))]
        if pct:
            ax.set_ylim(0, 1.12 if max(values, default=0) > 0.5 else max(0.1, 1.3 * max(values, default=0)))
            ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
        elif values and max(values) > 10 * min(values):
            ax.set_yscale("log")  # gliner is an order of magnitude slower
        ax.spines[["top", "right"]].set_visible(False)

    lengths, kinds = list(LENGTHS), list(KINDS)
    by = lambda r, key, c: claim_metrics([t for t in r["texts"] if t[key] == c])  # noqa: E731
    metrics = [("accuracy_decided", "accuracy\n(decided)"), ("accuracy_all", "accuracy\n(all)"), ("coverage", "coverage"),
               ("false_supported", "false →\nsupported"), ("true_refuted", "true →\nrefuted")]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5))
    bars(axes[0, 0], [m[1] for m in metrics], lambda r, c: claim_metrics(r["texts"])[dict((b, a) for a, b in metrics)[c]],
         "Overall (errors: lower is better)")
    bars(axes[0, 1], lengths, lambda r, c: by(r, "length", c)["accuracy_all"], "Accuracy (all claims) by length")
    bars(axes[0, 2], kinds, lambda r, c: by(r, "kind", c)["accuracy_all"], "Accuracy (all claims) by composition")
    bars(axes[1, 0], lengths, lambda r, c: by(r, "length", c)["false_supported"], "False claims marked supported, by length")
    bars(axes[1, 1], lengths, lambda r, c: by(r, "length", c)["latency"], "Median latency per text (models only)", pct=False)
    bars(axes[1, 2], lengths, lambda r, c: by(r, "length", c)["first"], "Median time to first verdict (models only)", pct=False)
    axes[1, 1].set_ylabel("seconds")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.suptitle(f"FactAssessor on 27 synthetic texts, same {runs[0]['source']} evidence", y=0.99, fontsize=12)
    describe = {r["key"]: VARIANTS[r["variant"]] + (f", {r['llm_model'].removeprefix('openai:')}" if r.get("llm_model") else "")
                for r in runs}
    fig.legend(handles, [f"{n}: {describe[n]}" for n in labels], loc="upper center", bbox_to_anchor=(0.5, 0.955),
               ncol=len(runs), frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    fig.savefig(RESULTS / "eval-comparison.png", dpi=130)

    verdicts = ["supported", "refuted", "contested", "unverified", "skipped"]
    vcolors = ["#55A868", "#C44E52", "#DD8452", "#8C8C8C", "#CCCCCC"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8), sharey=True)
    for ax, gold in zip(axes, (True, False)):
        left = [0.0] * len(runs)
        for v, color in zip(verdicts, vcolors):
            shares = []
            for r in runs:
                atoms = [a for t in r["texts"] for a in t["atoms"] if a["gold"] is gold]
                shares.append(sum(a["verdict"] == v for a in atoms) / len(atoms))
            b = ax.barh(names, shares, left=left, color=color, label=v)
            ax.bar_label(b, labels=[f"{s:.0%}" if s >= 0.04 else "" for s in shares], label_type="center", fontsize=8)
            left = [x + s for x, s in zip(left, shares)]
        ax.set_title(f"{'True' if gold else 'False'} claims: verdicts (want {'supported' if gold else 'refuted'})", fontsize=10)
        ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
        ax.spines[["top", "right"]].set_visible(False)
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, frameon=False)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(RESULTS / "eval-verdicts.png", dpi=130)
    plt.close("all")


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("build", help="build data/eval_texts.jsonl from data/fact_pairs.json")
    rec = sub.add_parser("record", help="record atoms, hits, and pages once")
    for p in (rec, r := sub.add_parser("run", help="evaluate variants")):
        p.add_argument("--searcher", choices=["ddg", "searxng", "serper"], default="ddg", help="for record and run --live")
        p.add_argument("--searxng-url", default="http://localhost:8888", help="your SearXNG instance (JSON enabled)")
        p.add_argument("--llm-model", default=LLM_MODEL, help="LLM for the atomizer (record, --live) and the llm judge")
    r.add_argument("variant", choices=[*VARIANTS, "all"])
    r.add_argument("--timeout", type=float, default=15.0, help="per-claim timeout (FactAssessor default 15s)")
    r.add_argument("--live", action="store_true", help="live atomizer, search (--searcher), and crawling instead of recorded evidence")
    sub.add_parser("report", help="rebuild data/results/eval-comparison.md and the plots")
    args = parser.parse_args()

    if args.cmd == "build":
        texts = build_texts(json.loads((DATA / "fact_pairs.json").read_text()))
        TEXTS.write_text("".join(json.dumps(t, ensure_ascii=False) + "\n" for t in texts))
        print(f"{len(texts)} texts, {sum(len(t['sentences']) for t in texts)} sentences -> {TEXTS}")
    elif args.cmd == "record":
        await record(args.searcher, args.searxng_url, args.llm_model)
    elif args.cmd == "run":
        RESULTS.mkdir(exist_ok=True)
        for variant in VARIANTS if args.variant == "all" else [args.variant]:
            res = await run(variant, args.live, args.timeout, args.searcher, args.searxng_url, args.llm_model)
            name = f"eval-{variant}{'-live' if args.live else ''}"
            (RESULTS / f"{name}.json").write_text(json.dumps(res, indent=1))
            (RESULTS / f"{name}.md").write_text(report(res))
            print("\n" + report(res))
    if args.cmd in ("run", "report"):
        (RESULTS / "eval-comparison.md").write_text(comparison())
        plot()
        print(f"-> {RESULTS / 'eval-comparison.md'}, eval-comparison.png, eval-verdicts.png")


if __name__ == "__main__":
    asyncio.run(main())
