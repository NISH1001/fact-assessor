"""The judge role: (claim, snippets or pages) -> Evidence (does each passage support, refute, or not settle it?)."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from factassessor.schema import Evidence


@runtime_checkable
class Judge(Protocol):
    """Role: does each passage support, refute, or not settle a claim? One method; anything with it is a judge.

    `docs` are search hits {"url", "title", "snippet"} and/or crawled pages {"url", "title", "text"}: judge snippets
    as-is and cut pages down to what fits your model.

    Optional, read when present: `aload()` / `aclose()` warm up and release the model (the pipeline's lifecycle
    calls them); `concurrency`, how many claims this judge takes at once, `Verify`'s default (absent or None: no
    limit, for judges that batch or scale; a slow judge sets a number so claims wait for a free slot instead of
    all sharing it and timing out together).
    """

    async def judge(self, claim: str, docs: list[dict[str, Any]]) -> list[Evidence]: ...
