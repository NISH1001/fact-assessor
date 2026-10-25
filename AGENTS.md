# fact-assessor

Async fact assessment: text → LLM atomizer → claim filter → search → crawl → judge → verdicts, score, graph.

## Where to read

- `README.md`: the pipeline, every component and parameter, the recommended setups (Serper + Jev; AKD Scholar
  Search + full text under "Scholarly search over full-text papers").
- `docs/ARCHITECTURE.md`: how the pieces fit, `Verify` step by step, concurrency and deadlines.
- `docs/WALKTHROUGH.md`: a guided tour (mirrors `notebooks/fa_walk.py`; keep them in sync).
- `docs/EVALS.md`: the benchmarks. "End-to-end harness" there is the manual for `scripts/eval_atoms.py` (modes,
  flags, output files, batching, costs, the measured setups' exact commands).
- `docs/design/decisions.md`: the measured trade-off behind every default, dated. Add a dated section for every
  eval that decides something.
- `docs/design/streaming-pipeline.md` (composable design), `docs/design/decision-runners.md` (the model layer:
  Laya, Jev / System One, LLM, GLiNER behind one `DecisionJudge` and one `DecisionClaimFilter`),
  `docs/design/composable-reader.md` (FirstOf / resolvers-as-steps, deferred).
- Eval runs land in `tmp/eval_atoms/<tag>/` (gitignored); their numbers and findings are logged by date in
  `tmp/eval_atoms/RESULTS-<date>.md` and, once settled, in `docs/design/decisions.md` and issue #1.

## Conventions

- Always use uv: `uv add <pkg>`, `uv sync`, `uv run <cmd>`. Never pip or `.venv/bin/python`.
- Tests: `uv run pytest` (offline; fakes for Serper, crawl4ai, Laya, the LLM and the AKD endpoints). TDD: write the
  failing test first.
- Speed matters most. Measure before optimizing (trace the timeline of a real `assess`; when a live run stalls,
  dump the main thread's stack before theorizing), and check accuracy on frozen evidence (`eval_atoms.py replay`),
  not live runs: live web results change between runs.
- Every step stays swappable behind a small async interface. Component-level concerns (caching, retries, hedging,
  connection pools) belong in wrappers around a component, not in the orchestrator.
- Laya is the local model for classification (filter, judge). It can't generate text. Question wording and state
  field order matter a lot for its accuracy: benchmark changes.
- Keys live in `.env` (gitignored): `SERPER_API_KEY`, `OPENAI_API_KEY`, `OPENROUTER_API_KEY`; AKD Scholar Search
  (`AKDScholarSearcher`, `AKDDoiCrawler`): `AKD_SCHOLAR_SEARCH_URL`, `AKD_SCHOLAR_SEARCH_API_KEY`. Never write that
  server's address in code, tests or docs: refer to the variable.
- Paid runs (OpenRouter, OpenAI, Serper credits) are the user's call: state the cost and check the key's remaining
  daily limit before a live run or a replay, and run small batches first (`--limit`, resumable by tag).
- Notebooks are marimo (`notebooks/`). Run them with `uv run marimo edit --no-sandbox notebooks/<name>.py`
  (`fa.py` has a PEP 723 header for the no-clone one-liner; in the repo, `--no-sandbox` uses the local code).
- Commit messages: plain subject and body, no attribution trailers.
