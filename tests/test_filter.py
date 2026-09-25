from factassessor import Atom, AtomFilter

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


async def test_drops_non_factual_and_keeps_input_order():
    kept, skipped = await AtomFilter(FakeLaya(), threshold=0.4).afilter(ATOMS)
    assert [a.id for a in kept] == [1, 2, 3]
    assert [a.id for a in skipped] == [0]
    assert [a.checkworthiness for a in kept] == [0.83, 0.89, 0.45]


async def test_caps_at_n_atoms_by_score():
    kept, skipped = await AtomFilter(FakeLaya(), n_atoms=2, threshold=0.4).afilter(ATOMS)
    assert [a.id for a in kept] == [1, 2]  # the two highest, still in input order
    assert [a.id for a in skipped] == [0, 3]


async def test_sends_one_batch_with_the_claim_as_state():
    laya = FakeLaya()
    await AtomFilter(laya).afilter(ATOMS)
    assert [r["state"] for r in laya.requests] == [{"claim": a.text} for a in ATOMS]


async def test_empty():
    assert await AtomFilter(FakeLaya()).afilter([]) == ([], [])
