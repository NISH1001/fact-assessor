"""The judge role: (claim, snippets or pages) -> Evidence (does each passage support, refute, or not settle it?)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from factassessor.schema import Evidence


class Judge(ABC):
    """Role: does each passage support, refute, or not settle a claim? Implement `judge`.

    `docs` are search hits {"url", "title", "snippet"} and/or crawled pages {"url", "title", "text"}: judge snippets
    as-is and cut pages down to what fits your model. Optional `aload`/`aclose` warm up / release the model.
    """

    @abstractmethod
    async def judge(self, claim: str, docs: list[dict[str, Any]]) -> list[Evidence]: ...

    async def aload(self) -> None:
        pass

    async def aclose(self) -> None:
        pass
