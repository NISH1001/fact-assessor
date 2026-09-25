"""Build the synthetic evaluation texts (data/eval_texts.jsonl) from data/fact_pairs.json.

    uv run python scripts/synthetic.py            # rewrites data/eval_texts.jsonl (seed 7, 27 texts)

Each pair is a true sentence and a false variant that changes one detail (a date, number, place, or person), the
realistic failure mode: mostly-right text with one wrong detail. A text samples pairs without repeats and joins one
sentence from each, so every sentence has a ground-truth label and a char span. The file is frozen so every run and
variant sees the same texts.

27 texts = {true, false, mixed} x {short 2-4, medium 5-10, long 15-25 sentences} x 3.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

DATA = Path(__file__).parent.parent / "data"
LENGTHS = {"short": (2, 4), "medium": (5, 10), "long": (15, 25)}  # sentences per text
KINDS = ("true", "false", "mixed")


def build_texts(pairs: list[dict], seed: int = 7, per_cell: int = 3) -> list[dict]:
    rng = random.Random(seed)
    texts = []
    for length, (lo, hi) in LENGTHS.items():
        for kind in KINDS:
            for i in range(per_cell):
                n = rng.randint(lo, hi)
                picked = rng.sample(pairs, n)
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
                texts.append({
                    "id": f"{length}-{kind}-{i}", "kind": kind, "length": length,
                    "text": " ".join(s["text"] for s in sentences), "sentences": sentences,
                })
    return texts


def load_texts() -> list[dict]:
    return [json.loads(line) for line in (DATA / "eval_texts.jsonl").read_text().splitlines()]


def label_at(text: dict, offset: int) -> bool | None:
    """Ground truth of the sentence containing char `offset` (an atom's span start)."""
    for s in text["sentences"]:
        if s["span"][0] <= offset < s["span"][1]:
            return s["true"]
    return None


if __name__ == "__main__":
    texts = build_texts(json.loads((DATA / "fact_pairs.json").read_text()))
    (DATA / "eval_texts.jsonl").write_text("".join(json.dumps(t, ensure_ascii=False) + "\n" for t in texts))
    print(f"{len(texts)} texts, {sum(len(t['sentences']) for t in texts)} sentences -> data/eval_texts.jsonl")
