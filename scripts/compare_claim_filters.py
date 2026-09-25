"""Accuracy and speed of claim filters on data/claim_cases.json.

    uv run --extra gliner python scripts/compare_claim_filters.py
"""

import asyncio
import json
import time
from pathlib import Path

from factassessor import Atom, GlinerClaimFilter, LayaClaimFilter

CASES = [(c["text"], c["is_claim"]) for c in json.loads((Path(__file__).parent.parent / "data" / "claim_cases.json").read_text())]


async def score(name, claim_filter):
    atoms = [Atom(id=i, text=text, span=(0, len(text))) for i, (text, _) in enumerate(CASES)]
    await claim_filter.start()
    await asyncio.gather(*(claim_filter.score(a) for a in atoms))  # warm up
    start = time.perf_counter()
    scores = await asyncio.gather(*(claim_filter.score(a) for a in atoms))
    elapsed = time.perf_counter() - start
    correct = sum((s >= claim_filter.threshold) == bool(k) for s, (_, k) in zip(scores, CASES))
    print(f"{name:20s} {correct:2d}/{len(CASES)} at threshold {claim_filter.threshold}  {elapsed * 1000:5.0f} ms for {len(CASES)} atoms")
    for s, (text, k) in zip(scores, CASES):
        if (s >= claim_filter.threshold) != bool(k):
            print(f"    miss: {'claim' if k else 'not a claim':11s} scored {s:.2f}  {text}")


async def main():
    await score("LayaClaimFilter", LayaClaimFilter())
    await score("GlinerClaimFilter", GlinerClaimFilter())


if __name__ == "__main__":
    asyncio.run(main())
