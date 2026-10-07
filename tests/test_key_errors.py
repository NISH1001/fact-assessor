"""A rejected API key (expired, revoked, wrong) stops the whole check with one clear error, naming the env var, instead
of every claim coming back unverified while the rest of the pipeline keeps searching and crawling."""

import asyncio

import httpx
import pytest
from pydantic_ai.exceptions import ModelHTTPError
from pydantic_ai.models.function import FunctionModel

from factassessor import Atom, Evidence, FactAssessor, Filter, FlatMap, InvalidAPIKeyError, Map, Step
from factassessor.atomizer import LLMAtomizer
from factassessor.keys import auth_error


def rejected(url, status=401, message="API key expired."):
    request = httpx.Request("POST", url)
    response = httpx.Response(status, json={"error": {"message": message}}, request=request)
    return httpx.HTTPStatusError(f"{status} from {url}", request=request, response=response)


def test_auth_error_names_the_key_from_the_provider_and_keeps_the_servers_reason():
    err = auth_error(rejected("https://openrouter.ai/api/v1/systemone"))
    assert isinstance(err, InvalidAPIKeyError) and err.env == "OPENROUTER_API_KEY" and err.reason == "API key expired."
    assert "OPENROUTER_API_KEY" in str(err) and "API key expired." in str(err)
    assert auth_error(rejected("https://api.openai.com/v1/decisions", 401, "Incorrect API key provided")).env == "OPENAI_API_KEY"
    assert auth_error(rejected("https://google.serper.dev/search", 403, "Unauthorized")).env == "SERPER_API_KEY"


def test_only_a_providers_rejection_counts_not_a_websites_or_another_failure():
    assert auth_error(rejected("https://www.nature.com/articles/x", 403, "Forbidden")) is None  # a paywall, not our key
    assert auth_error(rejected("https://openrouter.ai/api/v1/systemone", 500, "busy")) is None
    assert auth_error(rejected("https://openrouter.ai/api/v1/systemone", 429, "slow down")) is None
    assert auth_error(ValueError("nope")) is None


def test_an_llm_providers_rejection_names_the_key_given_by_the_caller():
    err = auth_error(ModelHTTPError(401, "gpt-6-luna", {"message": "Incorrect API key provided"}), env="OPENAI_API_KEY")
    assert isinstance(err, InvalidAPIKeyError) and err.env == "OPENAI_API_KEY" and "Incorrect API key" in err.reason


# --- the whole check stops ----------------------------------------------------------------------------------------

CLAIMS = [f"Claim number {i}." for i in range(8)]


class Atoms(Step):
    def __call__(self, texts):
        async def atoms(text):
            for i, claim in enumerate(CLAIMS):
                yield Atom(id=i, text=claim, span=(0, 1))

        return FlatMap(atoms)(texts)


class Searcher(Step):
    def __init__(self, fail=None):
        self.calls, self.fail = 0, fail

    def __call__(self, queries):
        async def hits(query):
            self.calls += 1
            await asyncio.sleep(0.01)
            if self.fail:
                raise self.fail
            yield {"url": f"https://example.org/{self.calls}", "title": "t", "snippet": query}

        return FlatMap(hits)(queries)


class Judge:
    def __init__(self, fail=None):
        self.calls, self.fail = 0, fail

    async def judge(self, claim, docs):
        self.calls += 1
        await asyncio.sleep(0.01)
        if self.fail:
            raise self.fail
        return [Evidence(url=d["url"], title="t", text="x", source="snippet", label="supports", prob=0.95) for d in docs]


class NoCrawl(Step):
    def __call__(self, urls):
        return Map(lambda url: None)(urls)


def assessor(searcher, judge, claim_filter=None, max_concurrent_claims=2):
    return FactAssessor(atomizer=Atoms(), claim_filter=claim_filter, searcher=searcher, crawler=NoCrawl(), resolver=None,
                        judge=judge, max_concurrent_claims=max_concurrent_claims)


async def raises_and_then_nothing_more(fa, *counters):
    with pytest.raises(InvalidAPIKeyError) as err:
        await fa.assess("text")
    at_error = [c.calls for c in counters]
    await asyncio.sleep(0.1)  # nothing still running in the background
    assert [c.calls for c in counters] == at_error
    return err.value


async def test_a_rejected_judge_key_stops_the_check_instead_of_failing_every_claim():
    searcher, judge = Searcher(), Judge(fail=rejected("https://openrouter.ai/api/v1/systemone"))
    err = await raises_and_then_nothing_more(assessor(searcher, judge), searcher, judge)
    assert err.env == "OPENROUTER_API_KEY"
    assert judge.calls < len(CLAIMS)  # 2 claims at a time: the rest never started


async def test_a_rejected_search_key_stops_the_check():
    searcher = Searcher(fail=rejected("https://google.serper.dev/search", 403, "Unauthorized"))
    err = await raises_and_then_nothing_more(assessor(searcher, Judge()), searcher)
    assert err.env == "SERPER_API_KEY" and searcher.calls < len(CLAIMS)


async def test_a_rejected_claim_filter_key_comes_out_as_the_same_error():
    async def dead_key(atom):
        raise rejected("https://openrouter.ai/api/v1/systemone")

    searcher = Searcher()
    err = await raises_and_then_nothing_more(assessor(searcher, Judge(), claim_filter=Filter(dead_key)), searcher)
    assert err.env == "OPENROUTER_API_KEY" and searcher.calls == 0


async def test_a_websites_403_still_fails_only_its_own_claim():
    searcher, judge = Searcher(), Judge(fail=rejected("https://www.nature.com/articles/x", 403, "Forbidden"))
    result = await assessor(searcher, judge).assess("text")
    assert len(result.atoms) == len(CLAIMS) and all(a.verdict == "unverified" and a.error for a in result.atoms)


async def test_a_rejected_atomizer_key_raises_instead_of_falling_back_to_sentences():
    def dead_key(messages, info):
        raise ModelHTTPError(401, "gpt-6-luna", {"message": "Incorrect API key provided"})

    atomizer = LLMAtomizer(source_queries=2)  # fallback on: an outage still degrades to sentences, a dead key doesn't
    with atomizer.agent.override(model=FunctionModel(dead_key)), atomizer.source_agent.override(model=FunctionModel(dead_key)):
        with pytest.raises(InvalidAPIKeyError, match="OPENAI_API_KEY"):
            await atomizer.atomize("NASA was founded in 1958.")
        with pytest.raises(InvalidAPIKeyError, match="OPENAI_API_KEY"):
            await atomizer.source_queries_for("NASA was founded in 1958.")


async def test_assess_many_stops_every_text_when_a_key_is_rejected():
    searcher, judge = Searcher(), Judge(fail=rejected("https://openrouter.ai/api/v1/systemone"))
    fa = assessor(searcher, judge)
    with pytest.raises(InvalidAPIKeyError):
        await fa.assess_many(["one", "two", "three", "four", "five"], concurrency=2)
    at_error = (searcher.calls, judge.calls)
    await asyncio.sleep(0.2)  # the other texts were cancelled, not left running (or queued to start)
    assert (searcher.calls, judge.calls) == at_error


async def test_a_halt_is_logged_once_with_the_key_and_what_was_cancelled(logs):
    searcher, judge = Searcher(), Judge(fail=rejected("https://openrouter.ai/api/v1/systemone"))
    with pytest.raises(InvalidAPIKeyError):
        await assessor(searcher, judge).assess_many(["one", "two", "three"], concurrency=1)  # two queued
    errors = [line for line in logs if line.startswith("ERROR")]
    warnings = [line for line in logs if line.startswith("WARNING")]
    assert len(errors) == 1 and "OPENROUTER_API_KEY" in errors[0] and "API key expired." in errors[0]  # once, not per claim
    assert len(warnings) == 1 and "cancelling 2 other text" in warnings[0]
