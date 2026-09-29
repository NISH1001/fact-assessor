"""Atomizers: text -> self-contained atomic claims. A step: texts -> atoms.

`Atomizer` is the role: implement `atomize(text)`. `LLMAtomizer` does it in one fast-LLM call.
Compose: `LLMAtomizer() >> LayaCheckworthy() >> Take(8)`.
"""

from __future__ import annotations

import logging
import re
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from typing import Any

from pydantic import BaseModel
from pydantic_ai import Agent

from factassessor.pipeline import FlatMap, Step
from factassessor._llm import reasoning_off
from factassessor.schema import Atom

logger = logging.getLogger(__name__)

INSTRUCTIONS = """\
Split the text into atomic claims for fact-checking.

- One fact per claim: a single subject with a single detail (a date, number, place, person, cause, or outcome). \
A sentence stating several details becomes several claims.
- Self-contained, for a reader who has not seen the text: no pronouns, and no claim that starts with or relies \
on a bare reference such as the study, the authors, the researchers, the model, the data, or it. Name what is \
meant as specifically as the text allows: a study by its venue, authors, place, and topic (if the text never \
names it, describe it by its topic and place); a place, event, object, instrument, dataset, or method by its \
name. Use only what the text says; never invent authors, years, or names.
- Repeat that identifying context in every claim, even when it is also stated as a claim of its own. What is \
never repeated is another claim's checked value: a date, measurement, or figure that has its own claim stays out \
of the other claims, so that if it is wrong only its own claim fails.
- Unwrap hedges and attributions (it was believed that, reports say) and state the claim directly.
- Keep every value exactly as written, even if you think it is wrong: we are checking the text, not correcting it.
- Also return opinions, greetings, and questions as claims; a later step filters them.
- source: copy the exact words from the text that the claim comes from."""

DEFAULT_MODEL = "openai:gpt-5.6-luna"

_SENTENCE = re.compile(r"\S.*?(?:[.!?]+(?=\s|$)|$)", re.S)  # ends at .!? + space, so "7.8" stays whole


class Claim(BaseModel):
    text: str
    source: str


class Claims(BaseModel):
    atoms: list[Claim]


class Atomizer(Step, ABC):
    """Role: text -> atoms. Implement `atomize`; streaming, concurrency, and chaining come from here."""

    @abstractmethod
    async def atomize(self, text: str) -> list[Atom]: ...

    def __call__(self, texts: AsyncIterator[str]) -> AsyncIterator[Atom]:
        async def atoms(text: str) -> AsyncIterator[Atom]:
            for atom in await self.atomize(text):
                yield atom

        return FlatMap(atoms)(texts)


class LLMAtomizer(Atomizer):
    """Atomize + decontextualize in one LLM call. Falls back to plain sentences if the LLM is unavailable."""

    def __init__(self, model: str = DEFAULT_MODEL, model_settings: dict[str, Any] | None = None) -> None:
        self.agent = Agent(
            model,
            output_type=Claims,
            instructions=INSTRUCTIONS,
            model_settings=reasoning_off(model) if model_settings is None else model_settings,
            defer_model_check=True,  # don't require an API key until the first call
        )

    async def atomize(self, text: str) -> list[Atom]:
        if not text.strip():
            return []
        try:
            claims = (await self.agent.run(text)).output.atoms
        except Exception as exc:  # best effort: an LLM outage degrades to sentence atoms, not a failed check
            logger.warning("atomizer LLM failed, falling back to sentences: %r", exc)
            return [Atom(id=i, text=text[s:e], span=(s, e)) for i, (s, e) in enumerate(_sentences(text))]
        return [
            Atom(id=i, text=c.text.strip(), span=_locate(c.source, text, hint=c.text))
            for i, c in enumerate(claims)
            if c.text.strip()
        ]


def _sentences(text: str) -> list[tuple[int, int]]:
    return [(m.start(), m.start() + len(m.group().rstrip())) for m in _SENTENCE.finditer(text)]


def _locate(quote: str, text: str, hint: str = "") -> tuple[int, int]:
    """Char span of `quote` in `text` (case/whitespace-insensitive); if the model paraphrased instead of
    quoting, the sentence sharing the most words with the claim itself (`hint`), then with the quote."""
    words = quote.split()
    if words:
        pattern = r"\s+".join(re.escape(w) for w in words)
        if m := re.search(pattern, text, re.IGNORECASE):
            return m.span()

    def overlap(sentence: tuple[int, int], other: str) -> int:
        return len(_words(text[sentence[0] : sentence[1]]) & _words(other))

    sentences = _sentences(text) or [(0, len(text))]
    return max(sentences, key=lambda se: (overlap(se, hint), overlap(se, quote)))


def _words(text: str) -> set[str]:
    return {w.lower() for w in re.findall(r"\w+", text)}
