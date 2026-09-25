"""Evaluate FactAssessor end to end on the synthetic texts (data/eval_texts.jsonl, built by scripts/synthetic.py).

    uv run python scripts/eval.py --variant laya                    # LayaClaimFilter + LayaJudge (default pipeline)
    uv run --extra gliner python scripts/eval.py --variant gliner   # GlinerClaimFilter + GlinerJudge
    uv run python scripts/eval.py --variant llm                     # LayaClaimFilter + LLMJudge (gpt-6-luna)

Warm-up first, timed and reported separately: load the models, run each component once (claim filter, judge),
then full checks until one is fast (warm connections, the atomizer's LLM client, crawl4ai's browser). Texts then
run one at a time so latencies don't share the machine. Every sentence is one fact with a known label; each atom
inherits the label of the sentence its span starts in. Only the claim filter and judge differ across variants.
Writes data/results/eval-<variant>.json (every atom) and eval-<variant>.md (the report).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from synthetic import DATA, KINDS, LENGTHS, label_at, load_texts  # noqa: E402

from factassessor import Atom, FactAssessor, LayaClaimFilter, LayaJudge, LLMJudge  # noqa: E402

RESULTS = DATA / "results"
WARM_TEXT = "The Moon orbits the Earth. Mount Fuji is the highest mountain in Japan."  # not in the eval set


def components(variant: str):
    """(claim_filter, judge) for a variant."""
    if variant == "gliner":
        from factassessor.gliner import GlinerClaimFilter, GlinerJudge  # one shared GLiNER model

        return GlinerClaimFilter(), GlinerJudge()
    if variant == "llm":
        return LayaClaimFilter(), LLMJudge()
    return LayaClaimFilter(), LayaJudge()


async def warm_up(fa: FactAssessor, claim_filter, judge) -> dict:
    """Load and exercise every component once; return the timings (seconds)."""
    t = {}
    start = time.perf_counter()
    await fa.aload()
    t["load"] = time.perf_counter() - start

    atom = Atom(id=0, text="The Moon orbits the Earth.", span=(0, 26))
    start = time.perf_counter()
    await claim_filter.score(atom)
    t["claim_filter_first"] = time.perf_counter() - start
    start = time.perf_counter()
    await claim_filter.score(atom)
    t["claim_filter_warm"] = time.perf_counter() - start

    doc = [{"url": "warmup", "title": "", "snippet": "The Moon is Earth's only natural satellite and orbits it."}]
    start = time.perf_counter()
    await judge.judge(atom.text, doc)
    t["judge_first"] = time.perf_counter() - start
    start = time.perf_counter()
    await judge.judge(atom.text, doc)
    t["judge_warm"] = time.perf_counter() - start

    t["assess"] = []
    for _ in range(3):  # until a full check is as fast as the one before it (connections, browser, LLM client)
        start = time.perf_counter()
        await fa.assess(WARM_TEXT)
        t["assess"].append(time.perf_counter() - start)
        if len(t["assess"]) > 1 and t["assess"][-1] <= t["assess"][-2] * 1.2:
            break
    return t


async def run(variant: str) -> tuple[dict, list[dict]]:
    claim_filter, judge = components(variant)
    fa = FactAssessor(n_atoms=30, claim_filter=claim_filter, judge=judge)  # n_atoms high enough to check every sentence
    warm = await warm_up(fa, claim_filter, judge)
    print("warm-up: " + ", ".join(f"{k} {v:.2f}s" if isinstance(v, float) else f"{k} {[round(x, 1) for x in v]}s"
                                  for k, v in warm.items()), flush=True)
    rows = []
    for ex in load_texts():
        start = time.perf_counter()
        first = None
        async for event in fa.stream(ex["text"]):
            if event.type == "claim_verified" and first is None:
                first = time.perf_counter() - start
            if event.type == "done":
                result = event.result
        total = time.perf_counter() - start
        atoms = [
            {"text": a.atom.text, "gold": label_at(ex, a.atom.span[0]), "verdict": a.verdict, "confidence": a.confidence,
             "error": a.error, "pages": sum(e.source == "page" for e in a.evidence)}
            for a in result.atoms
        ] + [{"text": s.text, "gold": label_at(ex, s.span[0]), "verdict": "skipped", "confidence": 0.0, "error": None, "pages": 0}
             for s in result.skipped]
        true_share = sum(s["true"] for s in ex["sentences"]) / len(ex["sentences"])
        rows.append({
            "id": ex["id"], "kind": ex["kind"], "length": ex["length"], "sentences": len(ex["sentences"]),
            "true_share": true_share, "fact_score": result.fact_score, "latency_s": total, "first_verdict_s": first,
            "atoms": atoms,
        })
        decided = [a for a in atoms if a["verdict"] in ("supported", "refuted")]
        right = sum((a["verdict"] == "supported") == a["gold"] for a in decided)
        score = "–" if result.fact_score is None else f"{result.fact_score:.2f}"
        print(f"{ex['id']:16s} {len(ex['sentences']):2d} sent -> {len(atoms):2d} atoms | {right}/{len(decided)} right of decided "
              f"| score {score} vs {true_share:.2f} | first {first or 0:.1f}s total {total:.1f}s", flush=True)
    await fa.aclose()
    return warm, rows


def claim_metrics(atoms: list[dict]) -> dict:
    labelled = [a for a in atoms if a["gold"] is not None]
    decided = [a for a in labelled if a["verdict"] in ("supported", "refuted")]
    correct = [a for a in decided if (a["verdict"] == "supported") == a["gold"]]
    false_claims = [a for a in labelled if a["gold"] is False]
    true_claims = [a for a in labelled if a["gold"] is True]
    pct = lambda n, d: n / d if d else None  # noqa: E731
    return {
        "claims": len(labelled),
        "coverage": pct(len(decided), len(labelled)),  # decided = supported or refuted
        "accuracy_decided": pct(len(correct), len(decided)),
        "accuracy_all": pct(len(correct), len(labelled)),  # contested/unverified/skipped count as wrong
        "false_supported": pct(sum(a["verdict"] == "supported" for a in false_claims), len(false_claims)),  # the dangerous error
        "true_refuted": pct(sum(a["verdict"] == "refuted" for a in true_claims), len(true_claims)),
        "unmatched": sum(a["gold"] is None for a in atoms),
    }


def report(warm: dict, rows: list[dict], variant: str) -> str:
    fmt = lambda x: "–" if x is None else f"{x:.0%}"  # noqa: E731
    names = {"laya": "LayaClaimFilter + LayaJudge", "gliner": "GlinerClaimFilter + GlinerJudge", "llm": "LayaClaimFilter + LLMJudge"}
    lines = [f"# FactAssessor on the synthetic set: {names[variant]}", "",
             f"{len(rows)} texts, {sum(r['sentences'] for r in rows)} sentences (one fact each), "
             f"{sum(len(r['atoms']) for r in rows)} atoms. Texts run one at a time after the warm-up.", "",
             "## Warm-up (not counted below)", "",
             f"Load {warm['load']:.1f}s. Claim filter: first call {warm['claim_filter_first']:.2f}s, then {warm['claim_filter_warm']:.2f}s. "
             f"Judge: first call {warm['judge_first']:.2f}s, then {warm['judge_warm']:.2f}s. "
             f"Full checks: {', '.join(f'{x:.1f}s' for x in warm['assess'])}.", ""]

    def section(title: str, groups: dict[str, list[dict]]) -> None:
        lines.extend([f"## {title}", "", "| group | claims | coverage | accuracy (decided) | accuracy (all) | false → supported | true → refuted | fact-score error | latency (median) | first verdict (median) |",
                      "|---|---|---|---|---|---|---|---|---|---|"])
        for name, rs in groups.items():
            m = claim_metrics([a for r in rs for a in r["atoms"]])
            errors = [abs(r["fact_score"] - r["true_share"]) for r in rs if r["fact_score"] is not None]
            lines.append(
                f"| {name} | {m['claims']} | {fmt(m['coverage'])} | {fmt(m['accuracy_decided'])} | {fmt(m['accuracy_all'])} | "
                f"{fmt(m['false_supported'])} | {fmt(m['true_refuted'])} | {statistics.mean(errors) if errors else float('nan'):.2f} | "
                f"{statistics.median(r['latency_s'] for r in rs):.1f}s | {statistics.median(r['first_verdict_s'] or 0 for r in rs):.1f}s |"
            )
        lines.append("")

    section("Overall", {"all": rows})
    by_length, by_kind = defaultdict(list), defaultdict(list)
    for r in rows:
        by_length[r["length"]].append(r)
        by_kind[r["kind"]].append(r)
    section("By length", {f"{k} ({LENGTHS[k][0]}–{LENGTHS[k][1]} sentences)": by_length[k] for k in LENGTHS})
    section("By composition", {k: by_kind[k] for k in KINDS})

    verdicts = defaultdict(lambda: defaultdict(int))
    for r in rows:
        for a in r["atoms"]:
            if a["gold"] is not None:
                verdicts["true" if a["gold"] else "false"][a["verdict"]] += 1
    lines.extend(["## Verdicts by ground truth", "", "| gold | supported | refuted | contested | unverified | skipped |", "|---|---|---|---|---|---|"])
    for gold in ("true", "false"):
        v = verdicts[gold]
        lines.append(f"| {gold} | {v['supported']} | {v['refuted']} | {v['contested']} | {v['unverified']} | {v['skipped']} |")
    lines.append("")

    per_claim = [r["latency_s"] / max(1, len(r["atoms"])) for r in rows]
    lines.extend(["## Latency", "",
                  f"Per text: median {statistics.median(r['latency_s'] for r in rows):.1f}s, max {max(r['latency_s'] for r in rows):.1f}s. "
                  f"Per claim (text latency / atoms): median {statistics.median(per_claim):.2f}s. "
                  f"First verdict: median {statistics.median(r['first_verdict_s'] or 0 for r in rows):.1f}s.", ""])
    return "\n".join(lines)


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=["laya", "gliner", "llm"], default="laya")
    args = parser.parse_args()
    warm, rows = await run(args.variant)
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / f"eval-{args.variant}.json").write_text(json.dumps({"warmup": warm, "texts": rows}, indent=1))
    text = report(warm, rows, args.variant)
    (RESULTS / f"eval-{args.variant}.md").write_text(text)
    print("\n" + text)


if __name__ == "__main__":
    asyncio.run(main())
