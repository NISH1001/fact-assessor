"""LLMAtomizer (gpt-6-luna, the default prompt) vs SpanAtomizer (atomizer-gliner-small v0.5) on the atomizer
package's test sets, and the span model's critic vs the Jev and Laya claim filters on human-labelled atoms.

The test sets and the scoring functions come from the atomizer repo and package (`atomizer.metrics`), so the span
model's numbers here match its own docs/EVALS.md. Needs the package, and a sentence model for the embedding match:

    W='--with "atomizer @ git+ssh://git@github.com/NISH1001/atomizer@v0.1.0" --with sentence-transformers'
    uv run $W python scripts/compare_atomizers.py atomize --limit 5     # both atomizers -> tmp/compare_atomizers/
    uv run $W python scripts/compare_atomizers.py critic --limit 20     # critic, Jev, Laya on the labelled atoms
    uv run $W python scripts/compare_atomizers.py report --out tmp/compare_atomizers   # a dry run's report
    uv run $W python scripts/compare_atomizers.py report                # the full run -> data/results/atomizers.{md,json,png}
    uv run $W python scripts/compare_atomizers.py figure                # the paper figure -> docs/figures/atomizers.{png,pdf}
    uv run $W python scripts/compare_atomizers.py distribution          # critic vs Jev scores -> docs/figures/atomizers-scores.{png,pdf}

Both steps resume where they stopped (cached per text). `--limit N`: the first N texts of each set (a dry run).

Sets (test splits; Decontextualization has none, its valid split is used):
    factcheckbench   30 real ChatGPT answers, human claims          propsegment       human sentence fragments (first 500)
    synthetic        323 AI-written texts, atoms + critic labels    propositionizer   GPT-4 propositions, Wikipedia (first 500)
    decontext        195 paragraphs, one sentence rewritten by a human to stand alone (atoms of that sentence only)

Metrics per set and atomizer:
    claims per text (and per sentence) next to the reference's; invented words (`invented_rate`: output words not in the
    input, glue words aside) and atoms with an invented number or name; precision / recall / F1 against the reference
    atoms by best match, with token F1 (the package's soft F1), embedding cosine (all-MiniLM-L6-v2: fair to
    paraphrase), and exact match; Decontextualization: references resolved; span model: wrong references; ms per
    text (one text at a time) and the LLM's $ per 1,000 texts.
Critic: on the synthetic and Factcheck-Bench atoms (labelled verifiable or not) and fact-assessor's claim cases:
    AUC, F1 at each scorer's default cut-off (critic 0.5, claim filters 0.4) and its best, agreement, histograms.

The span atoms come from the SpanAtomizer's own model with its defaults (every atom, the model's thresholds), read
whole so the wrong-reference count can use their substitutions.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import statistics
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("PYDANTIC_AI_NO_BANNER", "1")

from factassessor import Atom, DecisionClaimFilter, LayaRunner, LLMAtomizer, SpanAtomizer, SystemOneRunner  # noqa: E402
from factassessor.atomizers.llm import DEFAULT_MODEL as LLM_MODEL  # noqa: E402
from factassessor.utils import sentences  # noqa: E402

DATA = Path.home() / "dev/nish/projects/atomizer/data"
CACHE = Path("tmp/compare_atomizers")
RESULTS = Path("data/results")
SETS = {  # name: (file, split, at most this many texts)
    "factcheckbench": ("factcheckbench.jsonl", "test", None),
    "synthetic": ("synthetic.jsonl", "test", None),
    "decontext": ("decontext.jsonl", "valid", None),
    "propsegment": ("propsegment.jsonl", "test", 500),
    "propositionizer": ("propositionizer.jsonl", "test", 500),
}
ATOMIZERS = ["llm", "span"]
CUT = {"critic": 0.5, "jev": 0.4, "laya": 0.4}  # each scorer's default cut-off (SpanAtomizer "auto"; ClaimFilter)


def load_set(name: str, limit: int | None) -> list[dict[str, Any]]:
    file, split, cap = SETS[name]
    rows = [r for line in (DATA / file).read_text().splitlines() if line.strip()
            for r in [json.loads(line)] if r.get("split") == split]
    return rows[: min(x for x in (cap, limit) if x) if (cap or limit) else None]


def read_cache(path: Path) -> dict[str, dict[str, Any]]:
    return {r["id"]: r for line in path.read_text().splitlines() for r in [json.loads(line)]} if path.exists() else {}


def append(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


# --- atomize ------------------------------------------------------------------------------------------------------

async def atomize_llm(rows: list[dict[str, Any]], out: Path, concurrency: int) -> None:
    slots = asyncio.Semaphore(concurrency)

    async def one(r: dict[str, Any]) -> dict[str, Any]:
        async with slots:
            atomizer = LLMAtomizer(fallback=False)  # an LLM error stops the run instead of scoring sentence atoms
            run, usage = atomizer.agent.run, []

            async def counted(*args: Any, **kwargs: Any) -> Any:  # this call's token usage, for the cost
                result = await run(*args, **kwargs)
                usage.append(result.usage)  # a property in pydantic-ai 2.x
                return result

            atomizer.agent.run = counted  # type: ignore[method-assign]
            start = time.perf_counter()
            atoms = await atomizer.atomize(r["text"])
            ms = (time.perf_counter() - start) * 1e3
            return {"id": r["id"], "atoms": [{"text": a.text, "span": list(a.span)} for a in atoms], "ms": ms,
                    "input_tokens": usage[0].input_tokens if usage else 0, "output_tokens": usage[0].output_tokens if usage else 0}

    for done, result in enumerate(asyncio.as_completed([one(r) for r in rows]), 1):
        append(out, [await result])
        if done % 25 == 0:
            print(f"    llm {done}/{len(rows)}")


async def atomize_span(rows: list[dict[str, Any]], out: Path, span: SpanAtomizer) -> None:
    model = span.span  # the SpanAtomizer's model, with its defaults
    for i, r in enumerate(rows):
        start = time.perf_counter()
        [atoms] = model.atomize_many([r["text"]])  # one text at a time: the latency of a single check
        ms = (time.perf_counter() - start) * 1e3
        append(out, [{"id": r["id"], "ms": ms, "atoms": [
            {"text": a.text, "span": list(a.span), "verifiable": a.verifiable, "checkworthy": a.checkworthy,
             "antecedents": [p.text for p in a.parts if p.kind == "antecedent"]} for a in atoms]}])
        if (i + 1) % 100 == 0:
            print(f"    span {i + 1}/{len(rows)}")


async def cmd_atomize(args: argparse.Namespace) -> None:
    span = SpanAtomizer()
    if "span" in args.atomizers:
        start = time.perf_counter()
        await span.aload()
        print(f"span model loaded in {time.perf_counter() - start:.1f}s")
    for name in args.sets:
        rows = load_set(name, args.limit)
        for atomizer in args.atomizers:
            out = CACHE / f"{name}.{atomizer}.jsonl"
            todo = [r for r in rows if r["id"] not in read_cache(out)]
            print(f"{name} / {atomizer}: {len(rows) - len(todo)} cached, {len(todo)} to atomize")
            if todo and atomizer == "llm":
                await atomize_llm(todo, out, args.concurrency)
            elif todo:
                await atomize_span(todo, out, span)


# --- critic -------------------------------------------------------------------------------------------------------

def labelled(limit: int | None) -> list[dict[str, Any]]:
    """Human-labelled atoms: verifiable (1) or not (0)."""
    out = []
    for name in ("synthetic", "factcheckbench"):
        for r in load_set(name, limit):
            for i, (atom, label) in enumerate(zip(r["atoms"], r.get("atom_critic") or [])):
                if label and label.get("verifiable") is not None:
                    out.append({"id": f"{r['id']}:{i}", "set": name, "text": atom, "label": int(label["verifiable"])})
    cases = [r for line in (DATA / "critic.jsonl").read_text().splitlines() if line.strip()
             for r in [json.loads(line)] if r.get("kind") == "factassessor"]
    out += [{"id": r["id"], "set": "claim_cases", "text": r["text"], "label": int(r["critic"]["verifiable"])}
            for r in cases[:limit]]
    return out


async def cmd_critic(args: argparse.Namespace) -> None:
    rows = labelled(args.limit)
    out = CACHE / "critic.jsonl"
    todo = [r for r in rows if r["id"] not in read_cache(out)]
    print(f"{len(rows)} labelled atoms ({sum(r['label'] for r in rows)} verifiable), {len(todo)} to score")
    if not todo:
        return
    span = SpanAtomizer()
    await span.aload()
    texts = [r["text"] for r in todo]
    critic = span.span.model.critic_scores(texts)  # each atom taken whole, as one claim
    jev_runner = SystemOneRunner()
    jev = DecisionClaimFilter(jev_runner)
    filters = {"jev": jev}
    if args.laya:
        filters["laya"] = DecisionClaimFilter(LayaRunner())
    scores: dict[str, list[float]] = {}
    for name, f in filters.items():
        start = time.perf_counter()
        scores[name] = list(await asyncio.gather(*(f.score(Atom(id=0, text=t, span=(0, len(t)))) for t in texts)))
        print(f"  {name}: {time.perf_counter() - start:.1f}s")
    print(f"  Jev cost ${jev_runner.cost:.4f}")
    append(out, [{**r, "critic": c["verifiable"], **{k: v[i] for k, v in scores.items()}}
                 for i, (r, c) in enumerate(zip(todo, critic))])


# --- report -------------------------------------------------------------------------------------------------------

WORD = re.compile(r"\w+")


def invented_names_or_numbers(atom: str, text: str) -> bool:
    """A word with a digit, or a capital past the claim's first word, that the input never has: a changed value or a
    new name ("The speaker ..." is neither)."""
    ws = [w.lower() for w in WORD.findall(text)]
    known = set(ws) | {a + b for a, b in zip(ws, ws[1:])}  # "27 %" in the input, "27%" in the claim
    return any((any(c.isdigit() for c in w) or (i and w[0].isupper())) and w.lower() not in known
               for i, w in enumerate(WORD.findall(atom)))


def best_match(sim: Any) -> tuple[float, float]:
    """Sums of each prediction's best similarity to a reference atom, and each reference atom's best to a prediction."""
    if not sim.size:
        return 0.0, 0.0
    return float(sim.max(axis=1).sum()), float(sim.max(axis=0).sum())


def prf(p_sum: float, n_pred: int, r_sum: float, n_gold: int) -> dict[str, float]:
    p, r = (p_sum / n_pred if n_pred else 0.0), (r_sum / n_gold if n_gold else 0.0)
    return {"p": p, "r": r, "f1": 2 * p * r / (p + r) if p + r else 0.0}


def pct(xs: list[float], q: float) -> float:
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))] if xs else 0.0


def auc(scores: list[float], labels: list[int]) -> float:
    pos = [s for s, y in zip(scores, labels) if y]
    neg = [s for s, y in zip(scores, labels) if not y]
    if not pos or not neg:
        return float("nan")
    return sum((p > n) + 0.5 * (p == n) for p in pos for n in neg) / (len(pos) * len(neg))


def f1_at(scores: list[float], labels: list[int], cut: float) -> float:
    tp = sum(s >= cut and y for s, y in zip(scores, labels))
    fp = sum(s >= cut and not y for s, y in zip(scores, labels))
    fn = sum(s < cut and y for s, y in zip(scores, labels))
    return 2 * tp / (2 * tp + fp + fn) if tp else 0.0


def report_atoms(embed: Any) -> tuple[list[str], dict[str, Any]]:
    from atomizer import metrics
    import numpy as np
    from genai_prices import Usage, calc_price

    price = calc_price(Usage(input_tokens=1_000_000, output_tokens=1_000_000), model_ref=LLM_MODEL.split(":")[-1], provider_id="openai")
    md = ["## Atomization", "",
          "| Set | Atomizer | Texts | Claims / text (ref) | Claims / sentence | Token P / R / F1 | Embedding P / R / F1 "
          "| Exact F1 | Invented words | Atoms with a new number or name | Resolved | Wrong refs | ms / text p50 (p90) | $ / 1k texts |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    out: dict[str, Any] = {}
    for name in SETS:
        rows = {r["id"]: r for r in load_set(name, None)}
        for atomizer in ATOMIZERS:
            cached = read_cache(CACHE / f"{name}.{atomizer}.jsonl")
            done = [(rows[i], c) for i, c in cached.items() if i in rows]
            if not done:
                continue
            counts, tok_p, tok_r, emb_p, emb_r, n_pred, n_gold, new_vals, n_sent = [], 0.0, 0.0, 0.0, 0.0, 0, 0, 0, 0
            for r, c in done:
                atoms = c["atoms"]
                if name == "decontext":  # the rewritten sentence's atoms only: those starting in it
                    atoms = [a for a in atoms if a["span"][0] >= r["own_from"]]
                pred, gold = [a["text"] for a in atoms], r["atoms"]
                count = metrics.score_text(r["text"], pred, gold)
                if name == "decontext":
                    count |= metrics.resolved(r["text"], r["own_from"], pred, gold[0])
                if atomizer == "span":
                    count |= metrics.references([t for a in atoms for t in a.get("antecedents", [])], gold)
                counts.append(count)
                if pred and gold:
                    e = embed(pred + gold)
                    sim = e[: len(pred)] @ e[len(pred):].T
                    p, rr = best_match(np.clip(sim, 0, 1))
                    emb_p, emb_r = emb_p + p, emb_r + rr
                n_pred, n_gold = n_pred + len(pred), n_gold + len(gold)
                new_vals += sum(invented_names_or_numbers(t, r["text"]) for t in pred)
                n_sent += len(sentences(r["text"][r["own_from"]:] if name == "decontext" else r["text"])) or 1
            pooled = metrics.pool(counts)
            emb = prf(emb_p, n_pred, emb_r, n_gold)
            ms = [c["ms"] for _, c in done]
            cost = (sum(c.get("input_tokens", 0) for _, c in done) * float(price.input_price)
                    + sum(c.get("output_tokens", 0) for _, c in done) * float(price.output_price)) / 1e6
            res = {
                "texts": len(done), "claims_per_text": n_pred / len(done), "ref_claims_per_text": n_gold / len(done),
                "claims_per_sentence": n_pred / max(1, n_sent),
                "token": {"p": pooled["soft_p"], "r": pooled["soft_r"], "f1": pooled["soft_f1"]}, "embedding": emb,
                "exact_f1": pooled["exact_f1"], "invented_rate": pooled["invented_rate"],
                "new_number_or_name": new_vals / max(1, n_pred), "resolved": pooled.get("resolved"),
                "wrong_ref_rate": pooled.get("wrong_ref_rate"),
                "ms_p50": statistics.median(ms), "ms_p90": pct(ms, 0.9),
                "usd_per_1k_texts": cost / len(done) * 1000 if atomizer == "llm" else 0.0,
            }
            out[f"{name}/{atomizer}"] = res
            fmt = lambda x: "–" if x is None else f"{x:.3f}"  # noqa: E731
            md.append(
                f"| {name} | {atomizer} | {res['texts']} | {res['claims_per_text']:.1f} ({res['ref_claims_per_text']:.1f}) "
                f"| {res['claims_per_sentence']:.2f} | {res['token']['p']:.3f} / {res['token']['r']:.3f} / **{res['token']['f1']:.3f}** "
                f"| {emb['p']:.3f} / {emb['r']:.3f} / **{emb['f1']:.3f}** | {res['exact_f1']:.3f} | {res['invented_rate']:.2%} "
                f"| {res['new_number_or_name']:.1%} | {fmt(res['resolved'])} | {fmt(res['wrong_ref_rate'])} "
                f"| {res['ms_p50']:.0f} ({res['ms_p90']:.0f}) | {res['usd_per_1k_texts']:.2f} |")
    return md, out


def report_critic(out_dir: Path) -> tuple[list[str], dict[str, Any]]:
    rows = list(read_cache(CACHE / "critic.jsonl").values())
    if not rows:
        return [], {}
    scorers = [s for s in ("critic", "jev", "laya") if all(s in r for r in rows)]
    md = ["", "## Critic vs claim filters (human-labelled atoms: verifiable or not)", "",
          "| Set | Atoms (verifiable) | Scorer | AUC | F1 at its cut-off | Best F1 (cut-off) |", "|---|---|---|---|---|---|"]
    out: dict[str, Any] = {}
    for name in sorted({r["set"] for r in rows}) + ["all"]:
        sub = [r for r in rows if name in ("all", r["set"])]
        labels = [r["label"] for r in sub]
        for s in scorers:
            xs = [r[s] for r in sub]
            best = max((f1_at(xs, labels, c / 20), c / 20) for c in range(1, 20))
            out[f"{name}/{s}"] = {"auc": auc(xs, labels), "f1": f1_at(xs, labels, CUT[s]), "best_f1": best[0], "best_cut": best[1]}
            md.append(f"| {name} | {len(sub)} ({sum(labels)}) | {s} | {out[f'{name}/{s}']['auc']:.3f} "
                      f"| {out[f'{name}/{s}']['f1']:.3f} (at {CUT[s]}) | {best[0]:.3f} ({best[1]:.2f}) |")
    md += ["", "Agreement on keep / drop at each scorer's cut-off (all labelled atoms):", ""]
    for i, a in enumerate(scorers):
        for b in scorers[i + 1:]:
            same = sum((r[a] >= CUT[a]) == (r[b] >= CUT[b]) for r in rows) / len(rows)
            out[f"agreement/{a}-{b}"] = same
            md.append(f"- {a} vs {b}: {same:.1%}")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(scorers), figsize=(4 * len(scorers), 3), sharey=True, squeeze=False)
    for ax, s in zip(axes[0], scorers):
        for label, colour in ((1, "0.2"), (0, "0.65")):
            ax.hist([r[s] for r in rows if r["label"] == label], bins=20, range=(0, 1), color=colour, alpha=0.8,
                    label="verifiable" if label else "not verifiable")
        ax.axvline(CUT[s], color="k", linestyle="--", linewidth=1)
        ax.set_title(s)
        ax.set_xlabel("score")
    axes[0][0].set_ylabel("atoms")
    axes[0][0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "atomizers-critic.png", dpi=150)
    md += ["", "![score histograms](atomizers-critic.png)"]
    return md, out


def cmd_report(args: argparse.Namespace) -> None:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    embed = lambda texts: model.encode(texts, normalize_embeddings=True)  # noqa: E731
    atoms_md, atoms = report_atoms(embed)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    critic_md, critic = report_critic(out_dir)
    text = "\n".join(["# LLMAtomizer vs SpanAtomizer", "",
                      f"`LLMAtomizer` ({LLM_MODEL}, the default prompt) vs `SpanAtomizer` (atomizer-gliner-small v0.5, "
                      "its defaults). Written by scripts/compare_atomizers.py.", "", *atoms_md, *critic_md, ""])
    (out_dir / "atomizers.md").write_text(text)
    (out_dir / "atomizers.json").write_text(json.dumps({"atoms": atoms, "critic": critic}, indent=1))
    print(text)


SET_LABELS = {"factcheckbench": "Factcheck-\nBench", "synthetic": "Synthetic\n", "decontext": "Decontext-\nualization",
              "propsegment": "Prop-\nSegmEnt", "propositionizer": "Proposi-\ntionizer"}


def roc(scores: list[float], labels: list[int]) -> tuple[list[float], list[float]]:
    pairs = sorted(zip(scores, labels), reverse=True)
    pos, neg = sum(labels), len(labels) - sum(labels)
    fpr, tpr, tp, fp = [0.0], [0.0], 0, 0
    for i, (score, y) in enumerate(pairs):
        tp, fp = tp + y, fp + (not y)
        if i + 1 == len(pairs) or pairs[i + 1][0] != score:  # one point per distinct score
            fpr.append(fp / neg)
            tpr.append(tp / pos)
    return fpr, tpr


def cmd_figure(args: argparse.Namespace) -> None:
    """The paper figure: accuracy, speed, invented values, and the critic's ROC, in the docs/figures style (STIX Two
    Text, greyscale; PNG at 300 dpi, PDF with text as outlines via Ghostscript)."""
    import shutil
    import subprocess

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    plt.rcParams.update({"font.family": "STIX Two Text", "mathtext.fontset": "stix", "font.size": 9,
                         "axes.titlesize": 9.5, "axes.labelsize": 9, "legend.fontsize": 8, "xtick.labelsize": 8,
                         "ytick.labelsize": 8, "hatch.linewidth": 0.5, "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42})
    res = json.loads((RESULTS / "atomizers.json").read_text())
    atoms, critic = res["atoms"], list(read_cache(CACHE / "critic.jsonl").values())
    sets = [s for s in SETS if f"{s}/llm" in atoms and f"{s}/span" in atoms]
    colour = {"llm": "0.25", "span": "0.82"}
    hatch = {"llm": "", "span": "////"}  # tells the two apart in print and greyscale too
    name = {"llm": "LLMAtomizer (gpt-6-luna)", "span": "SpanAtomizer (163M, local)"}
    x, w = np.arange(len(sets)), 0.38
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.2))

    ax = axes[0][0]  # (a) accuracy
    for i, a in enumerate(("llm", "span")):
        xs = x + (i - 0.5) * w
        ax.bar(xs, [atoms[f"{s}/{a}"]["embedding"]["f1"] for s in sets], w, color=colour[a], hatch=hatch[a],
               edgecolor="black", linewidth=0.5, label=name[a])
        ax.scatter(xs, [atoms[f"{s}/{a}"]["token"]["f1"] for s in sets], marker="D", s=14, facecolor="white",
                   edgecolor="black", linewidth=0.7, zorder=3, label="token F1" if i else None)
    ax.set_xticks(x, [SET_LABELS[s] for s in sets])
    ax.set_ylim(0.4, 1.0)
    ax.set_ylabel("F1 vs reference claims")
    ax.set_title("(a) Accuracy: embedding F1 (bars), token F1 (diamonds)", loc="left")

    ax = axes[0][1]  # (b) speed
    for i, a in enumerate(("llm", "span")):
        xs = x + (i - 0.5) * w
        p50 = np.array([atoms[f"{s}/{a}"]["ms_p50"] for s in sets])
        p90 = np.array([atoms[f"{s}/{a}"]["ms_p90"] for s in sets])
        ax.bar(xs, p50, w, color=colour[a], hatch=hatch[a], edgecolor="black", linewidth=0.5)
        ax.errorbar(xs, p50, yerr=[np.zeros_like(p50), p90 - p50], fmt="none", ecolor="black", elinewidth=0.7, capsize=2)
    ax.set_yscale("log")
    ax.set_xticks(x, [SET_LABELS[s] for s in sets])
    ax.set_ylabel("ms per text (p50, whisker p90)")
    ax.set_title("(b) Latency per text", loc="left")

    ax = axes[1][0]  # (c) invented values
    for i, a in enumerate(("llm", "span")):
        ax.bar(x + (i - 0.5) * w, [100 * atoms[f"{s}/{a}"]["new_number_or_name"] for s in sets], w, color=colour[a],
               hatch=hatch[a], edgecolor="black", linewidth=0.5)
    ax.set_xticks(x, [SET_LABELS[s] for s in sets])
    ax.set_ylabel("% of claims")
    ax.set_title("(c) Claims with a number or name not in the text", loc="left")

    ax = axes[1][1]  # (d) critic vs claim filters
    labels = [r["label"] for r in critic]
    styles = {"critic": ("SpanAtomizer critic", "black", "-"), "jev": ("Jev claim filter", "0.35", "--"),
              "laya": ("Laya claim filter", "0.6", ":")}
    for s, (label, c, ls) in styles.items():
        if critic and all(s in r for r in critic):
            scores = [r[s] for r in critic]
            fpr, tpr = roc(scores, labels)
            ax.plot(fpr, tpr, color=c, linestyle=ls, linewidth=1.4, label=f"{label} (AUC {auc(scores, labels):.3f})")
    ax.plot([0, 1], [0, 1], color="0.8", linewidth=0.6)
    ax.set_xlabel("false positive rate")
    ax.set_ylabel("true positive rate")
    ax.set_title(f"(d) Verifiable or not: {len(critic):,} labelled claims", loc="left")
    ax.legend(loc="lower right", frameon=False)

    handles, labels = axes[0][0].get_legend_handles_labels()
    order = sorted(range(len(labels)), key=lambda i: labels[i] == "token F1")  # the two atomizers first
    fig.legend([handles[i] for i in order], [labels[i] for i in order], loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "atomizers.png", dpi=300)
    fig.savefig(out / "atomizers.raw.pdf")
    if shutil.which("gs"):  # text as outlines, like the other figures: no fonts needed to open it
        subprocess.run(["gs", "-q", "-dNOPAUSE", "-dBATCH", "-dSAFER", "-sDEVICE=pdfwrite", "-dNoOutputFonts",
                        f"-sOutputFile={out / 'atomizers.pdf'}", str(out / "atomizers.raw.pdf")], check=True)
        (out / "atomizers.raw.pdf").unlink()
    else:
        (out / "atomizers.raw.pdf").rename(out / "atomizers.pdf")
    print(f"wrote {out / 'atomizers.png'} and {out / 'atomizers.pdf'}")


def cmd_distribution(args: argparse.Namespace) -> None:
    """The critic's and Jev's scores on the same human-labelled claims, in one figure: verifiable claims (left) and
    the rest (right), side by side per score bin; log scale (most scores sit at 0 or 1)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    plt.rcParams.update({"font.family": "STIX Two Text", "font.size": 9, "axes.titlesize": 9.5, "legend.fontsize": 8,
                         "xtick.labelsize": 8, "ytick.labelsize": 8, "hatch.linewidth": 0.5,
                         "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42})
    rows = list(read_cache(CACHE / "critic.jsonl").values())
    bins = np.linspace(0, 1, 11)
    w = (bins[1] - bins[0]) * 0.4
    style = {"critic": ("SpanAtomizer critic (cut-off 0.5)", "0.25", ""),
             "jev": ("Jev claim filter (cut-off 0.4)", "0.82", "////")}
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8), sharey=True)
    for ax, label, title in ((axes[0], 1, "verifiable"), (axes[1], 0, "not verifiable")):
        sub = [r for r in rows if r["label"] == label]
        for i, (s_, (name, colour, hatch)) in enumerate(style.items()):
            counts, _ = np.histogram([r[s_] for r in sub], bins=bins)
            ax.bar(bins[:-1] + (bins[1] - bins[0]) / 2 + (i - 0.5) * w, counts, w, color=colour, hatch=hatch,
                   edgecolor="black", linewidth=0.5, label=name)
            ax.axvline(CUT[s_], color=colour if s_ == "critic" else "0.5", linestyle="--" if s_ == "critic" else ":",
                       linewidth=0.9)
        ax.set_yscale("log")
        ax.set_xlim(0, 1)
        ax.set_xticks(bins)
        ax.set_xlabel("score: P(verifiable claim)")
        ax.set_title(f"({'ab'[1 - label]}) {len(sub):,} claims labelled {title}", loc="left")
    axes[0].set_ylabel("claims (log scale)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "atomizers-scores.png", dpi=300)
    _pdf(fig, out / "atomizers-scores.pdf")
    print(f"wrote {out / 'atomizers-scores.png'} and {out / 'atomizers-scores.pdf'}")


def _pdf(fig: Any, path: Path) -> None:
    """A PDF with its text as outlines (Ghostscript), like the other figures: no fonts needed to open it."""
    import shutil
    import subprocess

    raw = path.with_suffix(".raw.pdf")
    fig.savefig(raw)
    if shutil.which("gs"):
        subprocess.run(["gs", "-q", "-dNOPAUSE", "-dBATCH", "-dSAFER", "-sDEVICE=pdfwrite", "-dNoOutputFonts",
                        f"-sOutputFile={path}", str(raw)], check=True)
        raw.unlink()
    else:
        raw.rename(path)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("atomize")
    a.add_argument("--sets", nargs="+", default=list(SETS), choices=list(SETS))
    a.add_argument("--atomizers", nargs="+", default=ATOMIZERS, choices=ATOMIZERS)
    a.add_argument("--limit", type=int, help="first N texts of each set (a dry run)")
    a.add_argument("--concurrency", type=int, default=16, help="LLM calls in flight")
    c = sub.add_parser("critic")
    c.add_argument("--limit", type=int, help="first N texts of each set")
    c.add_argument("--laya", action="store_true", help="also Laya (local, slower)")
    r = sub.add_parser("report")
    r.add_argument("--out", default=str(RESULTS), help="where to write atomizers.{md,json,png}")
    g = sub.add_parser("figure")
    g.add_argument("--out", default="docs/figures", help="where to write atomizers.{png,pdf}")
    d = sub.add_parser("distribution")
    d.add_argument("--out", default="docs/figures", help="where to write atomizers-scores.{png,pdf}")
    args = ap.parse_args()
    if args.cmd == "report":
        cmd_report(args)
    elif args.cmd == "figure":
        cmd_figure(args)
    elif args.cmd == "distribution":
        cmd_distribution(args)
    else:
        asyncio.run(cmd_atomize(args) if args.cmd == "atomize" else cmd_critic(args))


if __name__ == "__main__":
    main()
