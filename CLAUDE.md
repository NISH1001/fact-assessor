# fact-assessor

Async fact assessment: text → LLM atomizer → Laya filter → Serper → crawl4ai → Laya judge → verdicts, score, graph.
See README.md for the pipeline, docs/design/decisions.md for measured trade-offs behind every choice, and
docs/design/streaming-pipeline.md for the planned composable design (draft, awaiting review).

## Conventions

- Always use uv: `uv add <pkg>`, `uv sync`, `uv run <cmd>`. Never pip or `.venv/bin/python`.
- Tests: `uv run pytest` (offline; fakes for Serper, crawl4ai, Laya, and the LLM). TDD: write the failing test first.
- Speed matters most. Measure before optimizing (trace the timeline of a real `acheck`), and check accuracy on
  frozen evidence, not live runs: live web results change between runs.
- Every step stays swappable behind a small async interface. Component-level concerns (caching, retries,
  hedging) belong in wrappers around a component, not in the orchestrator.
- Laya is the local model for classification (filter, judge). It can't generate text. Question wording and state
  field order matter a lot for its accuracy: benchmark changes.
- Keys live in `.env` (gitignored): `SERPER_API_KEY`, `OPENAI_API_KEY`.
- Notebooks are marimo (`notebooks/`). Run them with `uv run marimo edit notebooks/<name>.py`.
