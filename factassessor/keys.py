"""API keys, checked when a component is created: a missing key is an error right away, with what to do instead,
not a failure at the first call (or a silent fallback, as the atomizer's sentence atoms once were)."""

from __future__ import annotations

import os
from typing import Any

from loguru import logger


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


# the hosts whose 401/403 means our key was rejected -> its environment variable. A crawled website's 401/403 (a
# paywall, a bot block) is not about our keys, so it fails only its own claim.
PROVIDER_HOSTS = {"openrouter.ai": "OPENROUTER_API_KEY", "api.openai.com": "OPENAI_API_KEY", "google.serper.dev": "SERPER_API_KEY"}


class MissingAPIKeyError(RuntimeError):
    """A component needs an API key that is neither passed in nor in the environment (or `.env`)."""


class InvalidAPIKeyError(RuntimeError):
    """A provider rejected an API key (expired, revoked, wrong): the whole check stops, since every call with that key
    would fail the same way. `env` is the variable to renew (None if unknown), `reason` the provider's message."""

    def __init__(self, env: str | None, reason: str) -> None:
        self.env, self.reason = env, reason
        super().__init__(f"{env or 'an API key'} was rejected: {reason} (renew it in the environment or .env)")


def auth_error(exc: BaseException, env: str | None = None) -> InvalidAPIKeyError | None:
    """`exc` as an InvalidAPIKeyError if it is a provider's 401/403 (`httpx.HTTPStatusError` from a provider host,
    or pydantic-ai's `ModelHTTPError`, whose key the caller names with `env`); None for anything else."""
    from pydantic_ai.exceptions import ModelHTTPError
    import httpx

    if isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code in (401, 403):
        host = exc.request.url.host
        known = next((v for h, v in PROVIDER_HOSTS.items() if host == h or host.endswith("." + h)), None)
        return InvalidAPIKeyError(known, _reason(_json(exc.response), exc.response.text)) if known else None
    if isinstance(exc, ModelHTTPError) and exc.status_code in (401, 403):
        return InvalidAPIKeyError(env, _reason(exc.body, str(exc.body)))
    return None


def _json(response: Any) -> Any:
    try:
        return response.json()
    except ValueError:
        return None


def _reason(body: Any, fallback: str) -> str:
    """The provider's own message: `{"error": {"message": ...}}` (OpenRouter, OpenAI) or `{"message": ...}`."""
    if isinstance(body, dict):
        inner = body.get("error", body)
        if isinstance(inner, dict) and inner.get("message"):
            return str(inner["message"])
        if isinstance(inner, str):
            return inner
    return fallback.strip()[:300]


def require_key(env: str, value: str | None, *, needed_by: str, instead: str) -> str:
    """`value`, else the `env` variable; logs and raises MissingAPIKeyError if neither is set."""
    key = (value or os.environ.get(env, "")).strip()
    if not key:
        message = f"{needed_by} needs {env}: set it in the environment or .env (or pass api_key=). {instead}"
        logger.error(message)
        raise MissingAPIKeyError(message)
    return key


def model_key(model: object) -> str | None:
    """The environment variable a pydantic-ai model string's provider reads (`openai:gpt-6-luna` -> OPENAI_API_KEY)."""
    return PROVIDER_KEYS.get(model.split(":", 1)[0]) if isinstance(model, str) else None


def require_model_key(model: object, *, needed_by: str, instead: str) -> None:
    """For a pydantic-ai model string (`openai:gpt-6-luna`): its provider's key must be set."""
    if env := model_key(model):
        require_key(env, None, needed_by=f"{needed_by} ({model})", instead=instead)
