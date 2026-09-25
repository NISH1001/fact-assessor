from factassessor import Atom, ClaimFilter, LayaClaimFilter, Take, collect
from factassessor.pipeline import dropped

ATOMS = [
    Atom(id=0, text="Hi, i am paradox.", span=(0, 16)),
    Atom(id=1, text="Paradox lives on Mars.", span=(18, 32)),
    Atom(id=2, text="Marie Curie won the Nobel Prize in 1903.", span=(34, 70)),
    Atom(id=3, text="NASA was founded in 1958.", span=(72, 95)),
]
P_FACTUAL = {a.text: p for a, p in zip(ATOMS, [0.08, 0.83, 0.89, 0.45])}


class FakeLaya:
    def __init__(self):
        self.requests = []

    async def predict_batch(self, requests):
        self.requests.extend(requests)
        return [{"answers": {"kind": {"probabilities": {"factual_claim": P_FACTUAL[r["state"]["claim"]]}}}} for r in requests]


async def atoms():
    for a in ATOMS:
        yield a


async def test_drops_non_factual_and_scores_the_rest():
    kept = await collect(LayaClaimFilter(threshold=0.4, runner=FakeLaya())(atoms()))
    assert sorted((a.id, a.claim_score) for a in kept) == [(1, 0.83), (2, 0.89), (3, 0.45)]


async def test_dropped_atoms_are_reported_as_skipped():
    skipped = []
    token = dropped.set(skipped)
    try:
        await collect(LayaClaimFilter(threshold=0.4, runner=FakeLaya())(atoms()))
    finally:
        dropped.reset(token)
    assert [(a.id, a.claim_score) for a in skipped] == [(0, 0.08)]


async def test_one_laya_decision_per_atom_with_the_claim_as_state():
    laya = FakeLaya()
    await collect(LayaClaimFilter(runner=laya)(atoms()))
    assert sorted(r["state"]["claim"] for r in laya.requests) == sorted(a.text for a in ATOMS)


async def test_chains_with_take():
    kept = await collect((LayaClaimFilter(threshold=0.4, runner=FakeLaya()) >> Take(2))(atoms()))
    assert len(kept) == 2


async def test_any_scoring_function_makes_a_claim_filter():
    class LongIsFactual(ClaimFilter):
        async def score(self, atom):
            return 0.9 if len(atom.text) > 20 else 0.1

    kept = await collect(LongIsFactual(threshold=0.5)(atoms()))
    assert sorted(a.id for a in kept) == [1, 2, 3] and all(a.claim_score == 0.9 for a in kept)
