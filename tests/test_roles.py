import pytest

from factassessor import (
    Atomizer, ClaimFilter, Crawler, DecisionJudge, DecisionRunner, Evidence, Judge, Policy, Ranker, Resolver, Searcher,
    Step, WeightedPolicy, collect, once,
)


class BareJudge:
    async def judge(self, claim, docs):
        return [Evidence(url=d["url"], title="", text="x", source="snippet", label="supports", prob=0.9) for d in docs]


class BarePolicy:
    def settled(self, evidence):
        return False

    def verdict(self, evidence):
        return "unverified", 0.0


def test_judge_policy_resolver_ranker_and_runner_are_structural_protocols():
    # a plain object with the one method is a judge; the built-in judge is one too; nothing else is
    assert isinstance(BareJudge(), Judge) and isinstance(DecisionJudge(runner=object()), Judge)
    assert isinstance(BarePolicy(), Policy) and isinstance(WeightedPolicy(), Policy)
    assert not any(isinstance(object(), role) for role in (Judge, Policy, Resolver, Ranker, DecisionRunner))


async def test_step_roles_are_base_classes_that_turn_one_method_into_a_step():
    class Sentences(Atomizer):
        async def atomize(self, text):
            from factassessor import Atom

            return [Atom(id=i, text=s, span=(0, 1)) for i, s in enumerate(text.split(". "))]

    class TwoHits(Searcher):
        async def search(self, query):
            return [{"url": f"https://{i}.org", "title": "", "snippet": query} for i in range(2)]

    class Nothing(Crawler):
        async def crawl(self, url):
            return None

    atoms = await collect((Sentences() >> Step.__rshift__(TwoHits(), Nothing()))(once("A. B")))  # chains with >>
    assert atoms == []  # two atoms -> four hits -> nothing crawled
    assert [h["url"] for h in await collect(TwoHits()(once("q")))] == ["https://0.org", "https://1.org"]


async def test_a_role_left_unimplemented_says_which_method():
    with pytest.raises(NotImplementedError, match="Atomizer.atomize"):
        await Atomizer().atomize("x")
    with pytest.raises(NotImplementedError, match="ClaimFilter.score"):
        await ClaimFilter().score(None)
