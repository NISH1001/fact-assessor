# fact-assessor

Fast, async-first fact assessment for any piece of text: a sentence, a paragraph, a model's answer.
It splits the text into atomic claims, finds web evidence for each one, and returns a verdict per claim,
an overall fact score, and a knowledge graph linking claims to their sources.

```python
from factassessor import FactAssessor

async with FactAssessor() as fa:
    result = await fa.assess("Nepal's earthquake in 2017 of 7.8 magnitude caused massive damage. Total lives lost were 1 million people.")

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

## Quick start: the notebook, one line

Needs [uv](https://docs.astral.sh/uv/), a [Serper](https://serper.dev) key, and an OpenAI key. No clone needed:

```bash
export SERPER_API_KEY=... OPENAI_API_KEY=...        # or put them in a .env in the current folder
uvx --from crawl4ai crawl4ai-setup                  # one time: installs the headless browser used for crawling
uvx marimo edit --sandbox https://raw.githubusercontent.com/NISH1001/fact-assessor/main/notebooks/fact_check.py
```

Type or paste text, press **Fact-check**, and get the fact score, a knowledge graph of claims and sources, and
every piece of evidence. The first run downloads Laya's weights (~0.8 GB) and warms up; after that a check takes
a few seconds.

## Install

As a dependency of your project:

```bash
uv add git+https://github.com/NISH1001/fact-assessor
uv run crawl4ai-setup          # one time: headless browser for crawling
```

For development:

```bash
git clone https://github.com/NISH1001/fact-assessor && cd fact-assessor
uv sync
uv run crawl4ai-setup
cp .env.example .env           # then fill in the keys
uv run marimo edit notebooks/fact_check.py      # one box, one button
uv run marimo edit notebooks/step_by_step.py    # every step's intermediate output, then run-everything
```

| Variable | Used by |
|---|---|
| `SERPER_API_KEY` | web search ([serper.dev](https://serper.dev)) |
| `OPENAI_API_KEY` | the atomizer (swap in any [pydantic-ai model](https://ai.pydantic.dev/models/), see below) |

A `.env` in the working directory (or a parent) is loaded when `factassessor` is imported; variables already set
in the environment win.

## Usage

### Check a text

```python
import asyncio
from factassessor import FactAssessor

async def main():
    async with FactAssessor() as fa:
        result = await fa.assess("Marie Curie won the Nobel Prize in Physics in 1903. The Eiffel Tower is 500 meters tall.")
    print(f"fact score {result.fact_score:.0%} in {result.latency_ms / 1000:.1f}s")
    for a in result.atoms:
        print(f"{a.verdict:10s} {a.confidence:.2f}  {a.atom.text}")

asyncio.run(main())
# example output (evidence comes from the live web, so borderline verdicts can vary between runs):
# fact score 50% in 4.4s
# supported  0.97  Marie Curie won the Nobel Prize in Physics in 1903.
# refuted    0.96  The Eiffel Tower is 500 meters tall.
```

In Jupyter or marimo, `await` works at the top level: `result = await fa.assess(text)`.

Not in async code? `assess_sync` blocks and returns the same result:

```python
from factassessor import FactAssessor

with FactAssessor() as fa:                     # closes the browser and HTTP pool on exit
    result = fa.assess_sync("NASA was founded in 1958.")
    result = fa.assess_sync("Python was created by James Gosling.")   # reuses the warm assessor
```

It runs on a background event loop owned by the assessor, so it also works where a loop is already running
(Jupyter), and repeated calls stay fast. Pick one style per instance: `assess` or `assess_sync`, not both.
(`acheck` is an alias for `assess`.)

### Keep one assessor alive

Creating a `FactAssessor` is cheap, but the first check loads Laya and starts a browser. In a notebook, app, or
service, create one, warm it up once, and reuse it for every check:

```python
fa = FactAssessor(n_atoms=8)
await fa.aload()              # load Laya + start the browser now (~3s), not on the first user request
...
result = await fa.assess(text)   # reuse for every check
...
await fa.aclose()             # on shutdown: closes the browser and HTTP pool (or use `async with`)
```

### Read the result

```
CheckResult
├── fact_score      supported / (supported + refuted + contested); None if nothing was checkable
├── latency_ms
├── atoms           list[AtomResult], in text order
│   ├── atom        Atom: text (self-contained claim), span (char offsets into your input), checkworthiness
│   ├── verdict     "supported" | "refuted" | "contested" | "unverified"
│   ├── confidence  0..1
│   ├── evidence    list[Evidence]: url, title, text (passage), source ("snippet" | "page"), label, prob
│   └── error       set if this claim failed (e.g. search down); the rest of the check still completes
├── skipped         list[Atom]: opinions, greetings, questions the filter dropped
└── graph           {"nodes": [...], "edges": [...]}: sources -> claims, labelled supports / refutes with weights
```

Everything is a Pydantic model, so `result.model_dump()` / `model_dump_json()` gives JSON for a UI. To highlight
claims in the original text, use each atom's `span`:

```python
for a in result.atoms:
    start, end = a.atom.span
    print(f"[{a.verdict}] {text[start:end]!r}")
```

### Configure

All keyword arguments to `FactAssessor`:

| Argument | Default | Meaning |
|---|---|---|
| `n_atoms` | 5 | max claims checked per text |
| `top_k` | 5 | search results per claim |
| `atomizer_model` | `openai:gpt-5.6-luna` | any pydantic-ai model string |
| `device` | `auto` | Laya device: cuda → mps → cpu |
| `checkworthy_threshold` | 0.4 | min P(factual claim) to keep an atom |
| `early_exit_conf` | 0.9 | 2+ passages this sure (and none against) settle a claim without more crawling |
| `strong_evidence` | 0.7 | min probability for a passage to count toward a verdict |
| `crawl_timeout` | 2.5 | seconds per page |
| `search_hedge_after` | 1.2 | seconds before racing a duplicate search |
| `blocked_domains` | social + video | hosts never used as evidence (subdomains included); `()` to allow all |
| `timeout` | 15 | overall deadline; claims still running come back `unverified` |

### Swap a step

Each step is an object with one async method; pass your own to replace it.

```python
from factassessor import Atomizer, FactAssessor

# another LLM for atomization (install its extra first, e.g. `uv add "pydantic-ai-slim[anthropic]"`)
fa = FactAssessor(atomizer=Atomizer("anthropic:claude-haiku-4-5", model_settings={}))

# your own filter: afilter(atoms) -> (kept, skipped)
class KeepEverything:
    async def afilter(self, atoms):
        return atoms, []

fa = FactAssessor(atom_filter=KeepEverything())
```

| Step | Argument | Method |
|---|---|---|
| Atomizer | `atomizer=` | `async aatomize(text) -> list[Atom]` |
| Filter | `atom_filter=` | `async afilter(atoms) -> (kept, skipped)` |
| Evidence judge | `judge=` | `async ajudge(claim, docs) -> list[Evidence]` |

Search and crawling become swappable the same way in the [streaming pipeline](docs/design/streaming-pipeline.md).

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
- Atomizer: keep opinions marked as opinions. Unwrapping hedges currently also strips "I think", so
  "I think pizza is the best food" becomes a plain claim; the default filter catches it, a custom one may not.

## Acknowledgements

Inspired by factuality pipelines such as [FActScore](https://github.com/shmsw25/FActScore), SAFE, and
[IBM's FactReasoner](https://github.com/IBM/FactReasoner); this project is an independent, latency-focused
implementation. Built on [Laya](https://github.com/NandhaKishorM/laya), [crawl4ai](https://github.com/unclecode/crawl4ai),
[pydantic-ai](https://ai.pydantic.dev), and [Serper](https://serper.dev).

## License

MIT
