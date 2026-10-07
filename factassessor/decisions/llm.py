"""LLMRunner: any chat LLM (pydantic-ai) asked to pick an option, as a decision model."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field
from pydantic_ai import Agent

from factassessor._llm import reasoning_off
from factassessor.decisions.types import Answer, DecisionRequest, DecisionResponse, DecisionRunner, Question
from factassessor.decisions.utils import RUN_LOCALLY, Batcher
from factassessor.keys import require_model_key


DEFAULT_LLM = "openai:gpt-6-luna"  # the most accurate judge we measured (15/15 on the judge cases); reasoning off

LLM_INSTRUCTIONS = """\
Each item shows named fields, a question about them (field names in backticks), and the options with what each
means. Using the fields alone, pick exactly one option per item and give your confidence from 0 to 1. Return
exactly one item per id."""


class Pick(BaseModel):
    id: int
    choice: str
    confidence: float = Field(ge=0, le=1)


class Picks(BaseModel):
    items: list[Pick]


class LLMRunner(DecisionRunner):
    """Any chat model as a decision model (pydantic-ai: OpenAI, OpenRouter, Ollama, an OpenAI-compatible endpoint).

    Every question becomes one numbered item of a prompt (its state fields, the question, the options); the model
    returns one pick with a confidence per item, as structured output, and that becomes the answer: the pick gets
    its confidence, the other options share the rest. Self-reported, not calibrated like Laya's or Jev's, so set
    `strong` from a replay before trusting a threshold. `batch_size` items per call; concurrent callers' requests
    are merged into shared calls (instructions sent once) with up to `max_concurrent` in flight. A call that fails
    raises: the claim comes back unverified with the error, not silently unsupported. An item the model skips or
    misnames gets even probabilities and confidence 0.
    """

    def __init__(
        self,
        model: str = DEFAULT_LLM,
        model_settings: dict[str, Any] | None = None,  # default: reasoning as low as the model allows
        batch_size: int = 40,
        max_concurrent: int = 8,
    ) -> None:
        require_model_key(model, needed_by="LLMRunner", instead=RUN_LOCALLY)
        self.model = model
        self.batch_size = batch_size
        self.agent = Agent(
            model,
            output_type=Picks,
            instructions=LLM_INSTRUCTIONS,
            model_settings=reasoning_off(model) if model_settings is None else model_settings,
            defer_model_check=True,  # don't require an API key until the first call
        )
        self._batch = Batcher(self._run, max_wait_ms=5.0, max_concurrent=max_concurrent, take=batch_size)

    async def predict(self, requests: list[DecisionRequest]) -> list[DecisionResponse]:
        if not requests:
            return []
        return await self._batch.submit(requests)

    async def _run(self, requests: list[DecisionRequest]) -> list[DecisionResponse]:
        items = [(i, key, r, q) for i, r in enumerate(requests) for key, q in r.questions.items()]
        prompt = "\n\n".join(_item(n, r.state, q) for n, (_, _, r, q) in enumerate(items, 1))
        picks = {p.id: p for p in (await self.agent.run(prompt)).output.items}
        answers: list[dict[str, Answer]] = [{} for _ in requests]
        for n, (i, key, _, q) in enumerate(items, 1):
            answers[i][key] = _answer(picks.get(n), list(q.criteria))
        return [DecisionResponse(answers=a) for a in answers]


def _item(n: int, state: str | dict[str, Any], q: Question) -> str:
    fields = f"state: {state}" if isinstance(state, str) else "\n".join(f"{k}: {v}" for k, v in state.items())
    options = "\n".join(f"- {label}: {meaning}" for label, meaning in q.criteria.items())
    return f"[{n}]\n{fields}\nquestion: {q.instructions}\noptions:\n{options}"


def _answer(pick: Pick | None, options: list[str]) -> Answer:
    chosen = next((o for o in options if pick and o.lower() == pick.choice.strip().lower()), None)
    if chosen is None or pick is None:  # skipped or an option that doesn't exist: no lean either way
        return Answer(probabilities={o: 1 / len(options) for o in options}, confidence=0.0)
    rest = (1 - pick.confidence) / (len(options) - 1) if len(options) > 1 else 0.0
    return Answer(choice=chosen, probabilities={o: pick.confidence if o == chosen else rest for o in options}, confidence=pick.confidence)
