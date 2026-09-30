"""Rankers: which of a page's chunks to show the judge for a claim. `top(claim, chunks, k)` -> the best `k`, best first
(`k=None`: every chunk, ranked).

`Ranker` is the role (a Protocol: anything with that method). The judge cuts a page into chunks and asks its ranker
for `passages_per_page` of them, so the ranker decides what the judge ever sees: a fact that is on the page but not
in the chosen chunk is lost.

- `BM25Ranker`: word overlap (the default). Exact terms and numbers, no model, milliseconds.
- `EmbeddingRanker`: cosine similarity of small static embeddings (model2vec, `fact-assessor[embed]`, CPU, a few ms
  per page); finds paraphrases BM25 misses ("saturates at" vs "saturation point of").
- `HybridRanker(bm25, embed, alpha)`: both, min-max normalized and mixed; `alpha` is BM25's weight.
"""

from __future__ import annotations

import threading
from typing import Any, Protocol, runtime_checkable

import numpy as np

from factassessor.passages import BM25Index


@runtime_checkable
class Ranker(Protocol):
    def top(self, claim: str, chunks: list[str], k: int | None = None) -> list[str]: ...


class BM25Ranker:
    """BM25 over the page's chunks; ties keep page order. Passages sharing no word with the claim rank last."""

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1, self.b = k1, b

    def scores(self, claim: str, chunks: list[str]) -> list[float]:
        found = BM25Index(chunks, self.k1, self.b).scores(claim)
        return [found.get(i, 0.0) for i in range(len(chunks))]

    def top(self, claim: str, chunks: list[str], k: int | None = None) -> list[str]:
        if k is not None and len(chunks) <= k:
            return chunks  # nothing to choose between
        return _best(chunks, self.scores(claim, chunks), k)


_MODELS: dict[str, Any] = {}  # model name -> loaded model2vec StaticModel, once per process
_MODELS_LOCK = threading.Lock()


def _static_model(name: str) -> Any:
    with _MODELS_LOCK:
        if name not in _MODELS:
            from model2vec import StaticModel

            _MODELS[name] = StaticModel.from_pretrained(name)
        return _MODELS[name]


class EmbeddingRanker:
    """Cosine similarity between the claim and each chunk, with a small static-embedding model (model2vec:
    `potion-base-8M` is ~30 MB, a token lookup and a mean per text, thousands of chunks per second on CPU).
    `model`: a model2vec model name (loaded once per process) or any object with `encode(list[str]) -> array`.
    Chunk vectors are cached (`cache_size` chunks): a text's claims are judged against the same pages, so a page is
    embedded once. Encoding is serialized: the judge ranks from worker threads and the tokenizer isn't thread-safe."""

    def __init__(self, model: Any = "minishlab/potion-base-8M", cache_size: int = 20_000) -> None:
        self._model = model
        self._cache: dict[str, Any] = {}
        self._lock = threading.Lock()
        self.cache_size = cache_size

    @property
    def model(self) -> Any:
        return _static_model(self._model) if isinstance(self._model, str) else self._model

    def _encode(self, texts: list[str]) -> Any:
        with self._lock:
            return np.asarray(self.model.encode(texts), dtype=float)

    def _vectors(self, texts: list[str]) -> Any:
        with self._lock:
            missing = [t for t in dict.fromkeys(texts) if t not in self._cache]
        if missing:
            vectors = self._encode(missing)
            with self._lock:
                for t, v in zip(missing, vectors):
                    self._cache[t] = v
                while len(self._cache) > self.cache_size:
                    del self._cache[next(iter(self._cache))]
        with self._lock:
            return np.stack([self._cache[t] for t in texts])

    def scores(self, claim: str, chunks: list[str]) -> list[float]:
        q = self._encode([claim])[0]  # the claim is new each time: not cached
        m = self._vectors(chunks)
        norms = np.linalg.norm(m, axis=1) * (np.linalg.norm(q) or 1.0)
        return [float(x) for x in m @ q / np.where(norms == 0, 1.0, norms)]

    def top(self, claim: str, chunks: list[str], k: int | None = None) -> list[str]:
        if k is not None and len(chunks) <= k:
            return chunks
        return _best(chunks, self.scores(claim, chunks), k)


class HybridRanker:
    """`alpha * bm25 + (1 - alpha) * embedding`, each min-max normalized over the page's chunks."""

    def __init__(self, bm25: BM25Ranker | None = None, embed: EmbeddingRanker | None = None, alpha: float = 0.5) -> None:
        self.bm25 = bm25 or BM25Ranker()
        self.embed = embed or EmbeddingRanker()
        self.alpha = alpha

    def top(self, claim: str, chunks: list[str], k: int | None = None) -> list[str]:
        if k is not None and len(chunks) <= k:
            return chunks
        a, b = _normalized(self.bm25.scores(claim, chunks)), _normalized(self.embed.scores(claim, chunks))
        return _best(chunks, [self.alpha * x + (1 - self.alpha) * y for x, y in zip(a, b)], k)


def _normalized(scores: list[float]) -> list[float]:
    lo, hi = min(scores), max(scores)
    return [(s - lo) / (hi - lo) if hi > lo else 0.0 for s in scores]


def _best(chunks: list[str], scores: list[float], k: int | None) -> list[str]:
    order = sorted(range(len(chunks)), key=lambda i: -scores[i])  # stable: ties keep page order
    return [chunks[i] for i in (order if k is None else order[:k])]
