# fact-assessor

Fast, async-first fact assessment for any piece of text: a sentence, a paragraph, a model's answer.
It splits the text into atomic claims, finds web evidence for each one, and returns a verdict per claim,
an overall fact score, and a knowledge graph linking claims to their sources.

```python
from factassessor import FactAssessor

async with FactAssessor() as fa:
    result = await fa.acheck("Nepal's earthquake in 2017 of 7.8 magnitude caused massive damage. Total lives lost were 1 million people.")

for atom in result.atoms:
    print(atom.verdict, atom.atom.text)
# contested  Nepal's earthquake occurred in 2017.        (it was 2015; see Roadmap: per-detail checks)
# supported  Nepal's earthquake had a magnitude of 7.8.
# supported  Nepal's earthquake caused massive damage.
# refuted    Nepal's earthquake killed 1 million people.

result.fact_score   # 0.5
result.graph        # {"nodes": [...], "edges": [...]}: sources -> claims, supports / refutes
```

It's built for interactive use (select text in a UI, see verdicts in seconds): every step is async, all claims
are checked concurrently, and most of the work is I/O that overlaps.

> Status: alpha. The API will change as the pipeline becomes composable (see [Roadmap](#roadmap)).

## How it works

```
text
 └─ Atomizer ─────────── one LLM call: atomic, self-contained claims ("Total lives lost…" → "The Nepal earthquake killed…")
     └─ AtomFilter ───── Laya, local: drop opinions, greetings, questions
         └─ search ───── Serper (Google), every claim in parallel; slow requests are hedged
             └─ judge ── Laya, local: does each snippet support / refute the claim?
                 ├─ settled → done (no crawling)
                 └─ not yet → crawl pages (crawl4ai) in parallel, judge each page as it lands,
                              stop as soon as the evidence settles the claim
 └─ aggregate ─────────── verdict per claim → fact score + knowledge graph
```

| Step | What does it | Where it runs |
|---|---|---|
| Atomize + decontextualize | `Atomizer`: [pydantic-ai](https://ai.pydantic.dev) → `openai:gpt-5.6-luna` (reasoning off) | API, ~2s |
| Check-worthiness filter | `AtomFilter`: [Laya](https://github.com/NandhaKishorM/laya) `choice` decision | local (MPS / CUDA / CPU) |
| Search | [Serper](https://serper.dev), social media and video sites filtered out | API, ~1s |
| Crawl | [crawl4ai](https://github.com/unclecode/crawl4ai), one shared headless browser, cleaned plain text | network, ~1s/page |
| Evidence judge | `LayaJudge`: pages chunked with Laya's own tokenizer, best BM25 passage per page | local |
| Verdicts, score, graph | strong evidence weighed per side: `supported` / `refuted` / `contested` / `unverified` | local |

Laya is a non-autoregressive decision model (a Jev-style encoder that classifies instead of generating), so the
filter and the judge are single forward passes. Every Laya request that arrives within a few milliseconds, from
any claim or page, is merged into one batch by `LayaRunner`.

**Latency** (M-series Mac, MPS, warm): ~4–6s for a 2–5 claim paragraph, most of it network. The atomizer call,
search, and crawling dominate; Laya passes take 40–150ms each. The first call in a fresh process also loads Laya
and starts the browser (several seconds, ~30s the very first time); call `await fa.aload()` up front to pay that
before the user is waiting. Evidence comes from the live web, so verdicts on borderline claims can vary between runs.

## Install

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/NISH1001/fact-assessor && cd fact-assessor
uv sync
uv run crawl4ai-setup          # installs the headless browser crawl4ai uses
cp .env.example .env           # then fill in the keys below
```

| Variable | Used by |
|---|---|
| `SERPER_API_KEY` | web search ([serper.dev](https://serper.dev)) |
| `OPENAI_API_KEY` | the atomizer (any [pydantic-ai model](https://ai.pydantic.dev/models/) works; see below) |

`.env` is loaded from the working directory (or a parent) when the package is imported. Laya downloads its
weights from Hugging Face on first use.

## Usage

```python
import asyncio
from factassessor import FactAssessor

async def main():
    async with FactAssessor(n_atoms=8, top_k=5) as fa:
        await fa.aload()                    # optional: warm Laya and the browser up front (~3s)
        result = await fa.acheck("Marie Curie won the Nobel Prize in Physics in 1903. The Eiffel Tower is 500 meters tall.")
        print(result.fact_score, [(a.atom.text, a.verdict, round(a.confidence, 2)) for a in result.atoms])

asyncio.run(main())
```

Each `AtomResult` carries the claim, its `span` in the input (for highlighting), the verdict and confidence, and
every piece of `Evidence` (URL, passage, label, probability). Atoms the filter skipped are in `result.skipped`.

Useful knobs (all keyword arguments to `FactAssessor`):

| Argument | Default | Meaning |
|---|---|---|
| `n_atoms` | 5 | max claims checked per text |
| `top_k` | 5 | search results per claim |
| `atomizer_model` | `openai:gpt-5.6-luna` | any pydantic-ai model string |
| `device` | `auto` | Laya device: cuda → mps → cpu |
| `checkworthy_threshold` | 0.4 | min P(factual claim) to keep an atom |
| `early_exit_conf` | 0.9 | 2+ passages this sure (and none against) settle a claim |
| `crawl_timeout` | 2.5 | seconds per page |
| `search_hedge_after` | 1.2 | seconds before racing a duplicate search |
| `blocked_domains` | social + video | hosts never used as evidence |

Steps are swappable: pass `atomizer=`, `atom_filter=`, or `judge=` any object with the same async method
(`aatomize(text)`, `afilter(atoms)`, `ajudge(claim, docs)`).

## Notebooks

[marimo](https://marimo.io) notebooks:

```bash
uv run marimo edit notebooks/fact_check.py      # one box, one button: score, knowledge graph, evidence
uv run marimo edit notebooks/step_by_step.py    # every step's intermediate output, then run-everything
```

## Development

```bash
uv run pytest            # fast, offline: every network/model call is faked
```

Layout:

```
factassessor/
  assessor.py        FactAssessor: orchestration, search, crawl, verdicts, score, graph
  atomizer.py        Atomizer (LLM)
  atom_filter.py     AtomFilter (Laya check-worthiness)
  evidence_judge.py  LayaJudge
  laya.py            LayaRunner: shared model + micro-batcher
  passages.py        page cleaning, token-exact chunking, BM25
  schema.py          Atom, Evidence, AtomResult, CheckResult
```

## Roadmap

- **Streaming, composable pipeline**: `Atomizer() >> Filter(...) >> Search(...) >> Verify(...) >> Aggregate()`,
  every step behind a small interface, claims flowing downstream the moment they exist, and `astream()` for
  per-claim results in the UI. Design: [docs/design/streaming-pipeline.md](docs/design/streaming-pipeline.md). Why things are the way they are (benchmarks, trade-offs): [docs/design/decisions.md](docs/design/decisions.md).
- Per-detail checks in the judge (ask Laya about each date/number in a claim in the same forward pass), so a
  claim that is right except for one detail comes out refuted rather than contested.
- Component-level wrappers: caching, retries, timeouts.

## Acknowledgements

Inspired by factuality pipelines such as [FActScore](https://github.com/shmsw25/FActScore), SAFE, and
[IBM's FactReasoner](https://github.com/IBM/FactReasoner); this project is an independent, latency-focused
implementation. Built on [Laya](https://github.com/NandhaKishorM/laya), [crawl4ai](https://github.com/unclecode/crawl4ai),
[pydantic-ai](https://ai.pydantic.dev), and [Serper](https://serper.dev).

## License

MIT
