"""Shared LLM settings for the steps that call an LLM (atomizer, LLMJudge)."""

from __future__ import annotations

import re
from typing import Any

_GPT5 = re.compile(r"gpt-5(?:-|$)")  # gpt-5, gpt-5-mini, gpt-5-nano (and dated snapshots); not gpt-5.1+
_O_SERIES = re.compile(r"o\d")  # o1, o3, o4-mini
_NO_REASONING = re.compile(r"gpt-(?:3|4)")  # gpt-4o, gpt-4.1: no reasoning, so the setting is an error


def reasoning_off(model: str) -> dict[str, Any]:
    """Model settings with reasoning as low as `model` allows (reasoning is billed as output tokens and slow).

    OpenAI models disagree on the lowest value: newer ones (gpt-5.1+) take "none", the gpt-5 family rejects it and
    takes "minimal", the o-series goes no lower than "low", and non-reasoning models reject the setting."""
    provider, _, name = model.partition(":")
    if provider != "openai":
        return {}
    if _NO_REASONING.match(name):
        return {}
    if _O_SERIES.match(name):
        return {"openai_reasoning_effort": "low"}
    return {"openai_reasoning_effort": "minimal" if _GPT5.match(name) else "none"}
