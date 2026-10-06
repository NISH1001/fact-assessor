"""API keys, checked when a component is created: a missing key is an error right away, with what to do instead,
not a failure at the first call (or a silent fallback, as the atomizer's sentence atoms once were)."""

from __future__ import annotations

import logging
import os

logger = logging.getLogger("factassessor")

# pydantic-ai model prefix -> the environment variable its provider reads; models not listed (ollama, local
# endpoints, test models) need no key from us
PROVIDER_KEYS = {
    "openai": "OPENAI_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "google-gla": "GEMINI_API_KEY",
    "groq": "GROQ_API_KEY",
    "mistral": "MISTRAL_API_KEY",
}


class MissingAPIKeyError(RuntimeError):
    """A component needs an API key that is neither passed in nor in the environment (or `.env`)."""


def require_key(env: str, value: str | None, *, needed_by: str, instead: str) -> str:
    """`value`, else the `env` variable; logs and raises MissingAPIKeyError if neither is set."""
    key = (value or os.environ.get(env, "")).strip()
    if not key:
        message = f"{needed_by} needs {env}: set it in the environment or .env (or pass api_key=). {instead}"
        logger.error(message)
        raise MissingAPIKeyError(message)
    return key


def require_model_key(model: object, *, needed_by: str, instead: str) -> None:
    """For a pydantic-ai model string (`openai:gpt-6-luna`): its provider's key must be set."""
    if isinstance(model, str) and (env := PROVIDER_KEYS.get(model.split(":", 1)[0])):
        require_key(env, None, needed_by=f"{needed_by} ({model})", instead=instead)
