import pytest


@pytest.fixture(autouse=True)
def api_keys(monkeypatch):
    """Dummy keys: components check for their key when created, and tests never call the real APIs."""
    for name in ("OPENAI_API_KEY", "OPENROUTER_API_KEY", "SERPER_API_KEY"):
        monkeypatch.setenv(name, "test-key")


@pytest.fixture
def logs():
    """What factassessor logged through loguru during the test, one "LEVEL message" line each."""
    from loguru import logger

    lines: list[str] = []
    sink = logger.add(lambda m: lines.append(f"{m.record['level'].name} {m.record['message']}"), level="DEBUG")
    yield lines
    logger.remove(sink)
