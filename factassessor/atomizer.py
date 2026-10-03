"""Atomizers: text -> self-contained atomic claims. A step: texts -> atoms.

`Atomizer` is the role: implement `atomize(text)`. `LLMAtomizer` does it in one fast-LLM call.
Compose: `LLMAtomizer() >> DecisionClaimFilter() >> Take(8)`.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from typing import Any

from pydantic import BaseModel
from pydantic_ai import Agent

from factassessor.pipeline import FlatMap, Step
from factassessor._llm import reasoning_off
from factassessor.schema import Atom
from factassessor.utils import locate, sentences

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
- Also return opinions, greetings, and questions as claims; a later step filters them."""

SOURCE_QUERY_RULE = """
- source_query: one web search query to find the document the whole text was taken from (a paper, report, or \
article), written the way its title and keywords would read: the topic, the method, the place or object, the \
instruments or datasets. No quotation marks and no numbers or results: they rarely appear in a title."""

DEFAULT_MODEL = "openai:gpt-6-luna"  # the same atoms as gpt-5.6-luna at half the price (reasoning off)


class Claim(BaseModel):
    text: str  # only the claim: a quote of its source words doubled the output tokens, and output is what the call's time goes on


class Claims(BaseModel):
    atoms: list[Claim]
    source_query: str = ""


class Atomizer(Step):
    """Role: text -> atoms. Implement `atomize`; streaming, concurrency, and chaining come from here."""

    async def atomize(self, text: str) -> list[Atom]:
        raise NotImplementedError(f"{type(self).__name__}.atomize")

    def __call__(self, texts: AsyncIterator[str]) -> AsyncIterator[Atom]:
        async def atoms(text: str) -> AsyncIterator[Atom]:
            for atom in await self.atomize(text):
                yield atom

        return FlatMap(atoms)(texts)


class LLMAtomizer(Atomizer):
    """Atomize + decontextualize in one LLM call. Falls back to plain sentences if the LLM is unavailable.

    `source_query=True`: the same call also writes one search query for the document the text was taken from
    (title-like: topic, method, place, instruments), carried by every atom as `Atom.source_query`. A claim about a
    detail inside a paper rarely finds that paper by itself; the whole text usually does (on 8 scientific passages,
    1 of 8 by their claims' own searches vs 5 of 8 by such a query, SearXNG top 10)."""

    def __init__(
        self, model: str = DEFAULT_MODEL, model_settings: dict[str, Any] | None = None, source_query: bool = False
    ) -> None:
        self.source_query = source_query
        self.instructions = INSTRUCTIONS + (SOURCE_QUERY_RULE if source_query else "")
        self.agent = Agent(
            model,
            output_type=Claims,
            instructions=self.instructions,
            model_settings=reasoning_off(model) if model_settings is None else model_settings,
            defer_model_check=True,  # don't require an API key until the first call
        )

    async def atomize(self, text: str) -> list[Atom]:
        if not text.strip():
            return []
        try:
            out = (await self.agent.run(text)).output
        except Exception as exc:  # best effort: an LLM outage degrades to sentence atoms, not a failed check
            logger.warning("atomizer LLM failed, falling back to sentences: %r", exc)
            return [Atom(id=i, text=text[s:e], span=(s, e)) for i, (s, e) in enumerate(sentences(text))]
        source_query = out.source_query.strip() or None if self.source_query else None
        return [  # span: the sentence the claim was made from, found here rather than quoted by the model
            Atom(id=i, text=c.text.strip(), span=locate(c.text, text), source_query=source_query)
            for i, c in enumerate(out.atoms)
            if c.text.strip()
        ]
