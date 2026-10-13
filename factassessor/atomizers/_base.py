"""The atomizer role: text -> self-contained atomic claims. A step: texts -> atoms.

Implement `atomize(text)`. Compose: `LLMAtomizer() >> DecisionClaimFilter() >> Take(8)`.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

from factassessor.pipeline import FlatMap, Step
from factassessor.schema import Atom


class Atomizer(Step):
    """Role: text -> atoms. Implement `atomize`; streaming, concurrency, and chaining come from here."""

    async def atomize(self, text: str) -> list[Atom]:
        raise NotImplementedError(f"{type(self).__name__}.atomize")

    def __call__(self, texts: AsyncIterator[str]) -> AsyncIterator[Atom]:
        async def atoms(text: str) -> AsyncIterator[Atom]:
            for atom in await self.atomize(text):
                yield atom

        return FlatMap(atoms)(texts)
