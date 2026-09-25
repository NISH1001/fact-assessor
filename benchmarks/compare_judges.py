"""Accuracy and speed of evidence judges on benchmarks/judge_cases.py.

    uv run --extra gliner python benchmarks/compare_judges.py        # LLM judges need OPENAI_API_KEY

Each case is one (claim, evidence snippet) pair with the expected label; every judge sees the same pairs.
"""

import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from judge_cases import CASES  # noqa: E402

from factassessor import LayaJudge, LLMJudge  # noqa: E402


async def score(name, judge):
    await judge.aload()
    docs = [[{"url": f"case{i}", "title": "", "snippet": e}] for i, (_, e, _) in enumerate(CASES)]
    await asyncio.gather(*(judge.judge(c, d) for (c, _, _), d in zip(CASES, docs)))  # warm up
    start = time.perf_counter()
    results = await asyncio.gather(*(judge.judge(c, d) for (c, _, _), d in zip(CASES, docs)))
    elapsed = time.perf_counter() - start
    correct = sum(r[0].label == expected for r, (_, _, expected) in zip(results, CASES))
    misses = [(c[:45], expected, r[0].label) for r, (c, _, expected) in zip(results, CASES) if r[0].label != expected]
    print(f"{name:34s} {correct:2d}/{len(CASES)}  {elapsed * 1000:6.0f} ms for {len(CASES)} pairs", flush=True)
    for claim, expected, got in misses:
        print(f"    miss: expected {expected:15s} got {got:15s} {claim}")


async def main():
    await score("LayaJudge", LayaJudge())
    for model in ("openai:gpt-5.6-luna", "openai:gpt-6-luna", "openai:gpt-5.4-mini", "openai:gpt-5.4-nano"):
        name = model.split(":")[1]
        await score(f"LLMJudge {name} (1 call)", LLMJudge(model, window_ms=20))
        await score(f"LLMJudge {name} (15 calls)", LLMJudge(model, window_ms=0))
    try:
        from factassessor.gliner import GlinerJudge
    except ImportError:
        print("GlinerJudge: install the extra first (uv sync --extra gliner)")
        return
    await score("GlinerJudge fp32", GlinerJudge(variant="fp32"))


if __name__ == "__main__":
    asyncio.run(main())
