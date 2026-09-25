"""Text -> decontextualized atomic claims, in one fast-LLM call. A step: texts -> atoms.

Compose: `Atomizer() >> LayaCheckworthy() >> Take(8)`. Swap in any step that turns text into atoms.
"""

from __future__ import annotations

import logging
import re
from collections.abc import AsyncIterator
from typing import Any

from pydantic import BaseModel
from pydantic_ai import Agent

from factassessor.pipeline import FlatMap, Step
from factassessor.schema import Atom

logger = logging.getLogger(__name__)

INSTRUCTIONS = """\
Split the text into atomic claims for fact-checking.

- One fact per claim: a single subject with a single detail (a date, number, place, person, cause, or outcome). \
"Nepal's 2017 earthquake of magnitude 7.8 caused massive damage" is three claims.
- Self-contained: replace pronouns and implicit references with the specific names from the text, and include \
the context needed to check the claim on its own ("Total lives lost were 1 million" -> "The Nepal earthquake killed 1 million people").
- Name the subject so it is identifiable on its own (keep names and places: "the Nepal earthquake", never just \
"the earthquake"), but never repeat a detail that has its own claim: "The Nepal earthquake had a magnitude of 7.8", \
not "Nepal's 2017 earthquake had a magnitude of 7.8" (the year is its own claim).
- Unwrap hedges and attributions: "It was believed that X" / "Reports say X" -> X.
- Keep every value exactly as written, even if you think it is wrong: we are checking the text, not correcting it.
- Also return opinions, greetings, and questions as claims; a later step filters them.
- source: copy the exact words from the text that the claim comes from."""

DEFAULT_MODEL = "openai:gpt-5.6-luna"
DEFAULT_SETTINGS: dict[str, Any] = {"openai_reasoning_effort": "none"}

_SENTENCE = re.compile(r"\S.*?(?:[.!?]+(?=\s|$)|$)", re.S)  # ends at .!? + space, so "7.8" stays whole


class Claim(BaseModel):
    text: str
    source: str


class Claims(BaseModel):
    atoms: list[Claim]


class Atomizer(Step):
    """Atomize + decontextualize in one call. Falls back to plain sentences if the LLM is unavailable."""

    def __init__(self, model: str = DEFAULT_MODEL, model_settings: dict[str, Any] | None = None) -> None:
        self.agent = Agent(
            model,
            output_type=Claims,
            instructions=INSTRUCTIONS,
            model_settings=DEFAULT_SETTINGS if model_settings is None else model_settings,
            defer_model_check=True,  # don't require an API key until the first call
        )

    def __call__(self, texts: AsyncIterator[str]) -> AsyncIterator[Atom]:
        async def atoms(text: str) -> AsyncIterator[Atom]:
            for atom in await self.aatomize(text):
                yield atom

        return FlatMap(atoms)(texts)

    async def aatomize(self, text: str) -> list[Atom]:
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
