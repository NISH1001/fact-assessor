from factassessor.passages import top_passages
from factassessor.rankers import BM25Ranker, EmbeddingRanker, HybridRanker, Ranker

CLAIM = "WorldView-3 has a saturation point at 247 Mg ha-1 for forest aboveground biomass."
CHUNKS = [
    "Field plots were established across the park in 2018 and 2019 to calibrate the LiDAR model.",
    "Sensitivity to large AGB values was higher for WV3, which saturates at 247 Mg ha-1, than for S2 and L8.",
    "The saturation point of Landsat-8 was 192 Mg ha-1 and that of Sentinel-2B 204 Mg ha-1.",
    "Random Forest models were fitted with three blocks of training data and one block for validation.",
]


class Fixed:
    """A fake embedder: a hand-set vector per text, so the ranking is deterministic and offline."""

    def __init__(self, vectors):
        self.vectors = vectors

    def encode(self, texts):
        import numpy as np

        return np.array([self.vectors.get(t, [0.0, 0.0, 1.0]) for t in texts], dtype=float)


def test_rankers_are_runtime_checkable_and_bm25_matches_top_passages():
    for r in (BM25Ranker(), EmbeddingRanker(model=Fixed({})), HybridRanker(BM25Ranker(), EmbeddingRanker(model=Fixed({})))):
        assert isinstance(r, Ranker)
    assert not isinstance(object(), Ranker)
    for k in (1, 2, 4):
        assert BM25Ranker().top(CLAIM, CHUNKS, k) == top_passages(CLAIM, CHUNKS, k)


def test_bm25_prefers_the_chunk_sharing_the_words_even_when_the_meaning_is_elsewhere():
    # "saturation point" + "Mg ha-1" twice: BM25 likes the Landsat/Sentinel sentence; the fact about WorldView-3
    # is phrased "saturates at" in chunk 1
    assert BM25Ranker().top(CLAIM, CHUNKS, 1) == [CHUNKS[2]]


def test_embedding_ranker_finds_the_paraphrase():
    vectors = {CLAIM: [1.0, 0.0, 0.0], CHUNKS[1]: [0.9, 0.1, 0.0], CHUNKS[2]: [0.5, 0.5, 0.0]}
    assert EmbeddingRanker(model=Fixed(vectors)).top(CLAIM, CHUNKS, 2) == [CHUNKS[1], CHUNKS[2]]


def test_hybrid_combines_both_scores_and_alpha_moves_the_balance():
    vectors = {CLAIM: [1.0, 0.0, 0.0], CHUNKS[1]: [0.95, 0.05, 0.0], CHUNKS[2]: [0.3, 0.7, 0.0]}
    embed = EmbeddingRanker(model=Fixed(vectors))
    assert HybridRanker(BM25Ranker(), embed, alpha=0.5).top(CLAIM, CHUNKS, 1) == [CHUNKS[1]]  # the paraphrase wins
    assert HybridRanker(BM25Ranker(), embed, alpha=1.0).top(CLAIM, CHUNKS, 1) == [CHUNKS[2]]  # pure BM25
    assert HybridRanker(BM25Ranker(), embed, alpha=0.0).top(CLAIM, CHUNKS, 1) == [CHUNKS[1]]  # pure embeddings


def test_k_none_returns_every_chunk_ranked():
    ranked = BM25Ranker().top(CLAIM, CHUNKS)
    assert sorted(ranked) == sorted(CHUNKS) and ranked[0] == CHUNKS[2]  # all of them, best first
    assert BM25Ranker().top(CLAIM, CHUNKS[:1]) == CHUNKS[:1]


def test_few_chunks_come_back_as_they_are_and_embeddings_are_cached_per_chunk():
    calls = []

    class Counting(Fixed):
        def encode(self, texts):
            calls.append(list(texts))
            return super().encode(texts)

    r = EmbeddingRanker(model=Counting({}))
    assert r.top(CLAIM, CHUNKS[:1], 3) == CHUNKS[:1] and calls == []  # nothing to rank
    r.top(CLAIM, CHUNKS, 1)
    r.top("another claim about the same page", CHUNKS, 1)
    encoded = [t for batch in calls for t in batch]
    assert encoded.count(CHUNKS[0]) == 1  # the page's chunks were embedded once, the claims each time


async def test_judges_take_a_ranker():
    from factassessor import DecisionJudge

    r = BM25Ranker(k1=1.2)
    assert DecisionJudge(ranker=r).ranker is r
