import pytest


@pytest.fixture(autouse=True)
def api_keys(monkeypatch):
    """Dummy keys: components check for their key when created, and tests never call the real APIs."""
    for name in ("OPENAI_API_KEY", "OPENROUTER_API_KEY", "SERPER_API_KEY"):
        monkeypatch.setenv(name, "test-key")
