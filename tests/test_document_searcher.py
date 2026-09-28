from factassessor import DocumentSearcher, Take, collect, once
from factassessor.passages import BM25Index, top_passages

FILLER = " ".join(f"background{i} sentence about forests and carbon." for i in range(300))
DOCS = [
    {"url": "paper-a", "title": "Biomass resilience of Neotropical secondary forests",
     "text": f"{FILLER} Poorter and colleagues analyzed 1,468 plots across 45 sites in the Neotropics. {FILLER}"},
    {"url": "paper-b", "title": "Lunar pits", "text": "Lunar pits are steep-walled depressions on the Moon. " * 40},
]


async def test_the_passage_with_the_fact_comes_first():
    s = DocumentSearcher(DOCS, passage_words=40)
    hits = await s.search("Poorter analyzed 1,468 plots across 45 sites in the Neotropics.")
    assert hits[0]["url"].startswith("paper-a#") and hits[0]["title"] == DOCS[0]["title"]
    assert "1,468 plots" in hits[0]["snippet"] and len(hits[0]["snippet"].split()) <= 40
    assert set(hits[0]) == {"url", "title", "snippet"}  # the same shape as web search hits


async def test_num_passages_and_distinct_urls():
    hits = await DocumentSearcher(DOCS, passage_words=40, num=3).search("lunar pits steep walls moon")
    assert len(hits) == 3 and len({h["url"] for h in hits}) == 3 and all(h["url"].startswith("paper-b#") for h in hits)


async def test_it_is_a_searcher_step_that_chains():
    hits = await collect((DocumentSearcher(DOCS, passage_words=40) >> Take(1))(once("45 sites in the Neotropics")))
    assert len(hits) == 1 and "45 sites" in hits[0]["snippet"]


async def test_no_documents_or_no_match_is_empty():
    assert await DocumentSearcher([]).search("anything") == []
    assert await DocumentSearcher(DOCS).search("zzzz qqqq") == []  # no shared words: nothing is relevant


def test_bm25_index_matches_top_passages():
    chunks = [f"plot {i} forest" for i in range(20)] + ["Poorter analyzed 1,468 plots", "lunar pits"]
    index = BM25Index(chunks)
    assert [chunks[i] for i in index.top("1,468 plots Poorter", 3)] == top_passages("1,468 plots Poorter", chunks, 3)
