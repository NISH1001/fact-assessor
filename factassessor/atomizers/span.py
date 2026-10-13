"""SpanAtomizer: text -> atomic claims with a local span model (the `atomizer` package), no LLM call.

The model (GLiNER-based, 163M params for small) builds each claim from the text's own words, swapping a reference
("She") for what it names, and scores it with a critic: `verifiable`, P(a factual claim that can be checked). Tens
of ms per text; texts over 1,024 tokens are atomized in sentence windows. It can't rephrase, so an event reference
("the decision") stays as written. Weights: https://huggingface.co/nishparadox/atomizer-gliner-small (v0.5).

The package is private for now, so it isn't a dependency:
`uv add "atomizer @ git+ssh://git@github.com/NISH1001/atomizer@v0.1.0"`.
"""

from __future__ import annotations

import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from factassessor.atomizers._base import Atomizer
from factassessor.decisions.utils import Batcher
from factassessor.pipeline import report_dropped
from factassessor.schema import Atom

DEFAULT_MODEL = "nishparadox/atomizer-gliner-small"
DEFAULT_REVISION = "v0.5"
INSTALL = 'uv add "atomizer @ git+ssh://git@github.com/NISH1001/atomizer@v0.1.0"'


class SpanModel:
    """One loaded span model, run on its own thread (torch: one call at a time). Get it with `span_model()`."""

    def __init__(self, repo: str, revision: str | None, device: str) -> None:
        self.repo, self.revision, self.device = repo, revision, device
        self.model: Any = None
        self.thread = ThreadPoolExecutor(max_workers=1, thread_name_prefix="atomizer")

    async def aload(self) -> None:
        await asyncio.get_running_loop().run_in_executor(self.thread, self.load)

    def load(self) -> None:
        if self.model is not None:
            return
        try:
            from atomizer import Atomizer as Model
        except ImportError as exc:
            raise ImportError(f"SpanAtomizer needs the atomizer package: {INSTALL}") from exc
        self.model = Model.load(self.repo, device=self.device, revision=self.revision)

    def atomize_many(self, texts: list[str], batch_size: int = 16, anchor_threshold: float | None = None,
                     ante_threshold: float | None = None) -> list[list[Any]]:
        self.load()
        return self.model.atomize_many(texts, batch_size=batch_size, anchor_threshold=anchor_threshold,
                                       ante_threshold=ante_threshold)

    def cut_off(self, name: str) -> float:
        """The model's own critic cut-off for `name` ("auto"): the package's, 0.5 when the model stores none."""
        return (getattr(getattr(self.model, "config", None), "critic_threshold", None) or {}).get(name, 0.5)


_models: dict[tuple[str, str | None, str], SpanModel] = {}
_models_lock = threading.Lock()


def span_model(model: str = DEFAULT_MODEL, revision: str | None = DEFAULT_REVISION, device: str = "auto") -> SpanModel:
    """The process-wide span model for (model, revision, device); loaded on first use."""
    with _models_lock:
        return _models.setdefault((model, revision, device), SpanModel(model, revision, device))


class SpanAtomizer(Atomizer):
    """Atomize with the local span model: no LLM call, no API key. The parameters are the package's own
    (`Atomizer.load`, `atomize_many`), with the same defaults: every atom, the model's own decoding thresholds.

    Every atom's `claim_score` is the model's `verifiable` score. `min_verifiable` / `min_checkworthy` drop atoms
    below a cut-off ("auto": the model's, 0.5 for small v0.5), reported as `skipped`; with one set, the critic is the
    claim filter: `FactAssessor(atomizer=SpanAtomizer(min_verifiable=0.5), claim_filter=None, source_queries=0)`
    (`source_queries=0`: this atomizer writes none, so claims get the page budget of their own searches only).
    Texts atomized at the same time (`assess_many`) share one model call.
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,  # a Hugging Face repo or a local directory written by atomizer's `save`
        revision: str | None = DEFAULT_REVISION,
        device: str = "auto",  # cuda > mps > cpu
        batch_size: int = 16,
        anchor_threshold: float | None = None,  # None: the model's own (0.35); lower: more atoms
        ante_threshold: float | None = None,  # None: the model's own (0); higher: fewer substituted references
        min_verifiable: float | str | None = None,  # None: keep every atom; a cut-off or "auto"
        min_checkworthy: float | str | None = None,
    ) -> None:
        self.model, self.revision, self.device = model, revision, device
        self.batch_size = batch_size
        self.anchor_threshold, self.ante_threshold = anchor_threshold, ante_threshold
        self.min_verifiable, self.min_checkworthy = min_verifiable, min_checkworthy
        self._model: SpanModel | None = None  # tests inject a fake
        self._batcher = Batcher(self._run)

    @property
    def span(self) -> SpanModel:
        if self._model is None:
            self._model = span_model(self.model, self.revision, self.device)
        return self._model

    async def aload(self) -> None:
        """Download (first time) and load the model."""
        await self.span.aload()

    async def _run(self, texts: list[str]) -> list[list[Any]]:
        thread = getattr(self.span, "thread", None)  # a fake has none: the default executor
        return await asyncio.get_running_loop().run_in_executor(
            thread, lambda: self.span.atomize_many(texts, batch_size=self.batch_size,
                                                   anchor_threshold=self.anchor_threshold, ante_threshold=self.ante_threshold))

    def _cut(self, name: str, value: float | str | None) -> float | None:
        if value != "auto":
            return value  # type: ignore[return-value]
        cut_off = getattr(self.span, "cut_off", None)
        return cut_off(name) if cut_off else 0.5

    async def atomize(self, text: str) -> list[Atom]:
        if not text.strip():
            return []
        (found,) = await self._batcher.submit([text])
        verifiable, checkworthy = self._cut("verifiable", self.min_verifiable), self._cut("checkworthy", self.min_checkworthy)
        atoms = []
        for i, a in enumerate(found):
            atom = Atom(id=i, text=a.text, span=tuple(a.span), claim_score=a.verifiable)
            if (verifiable is not None and a.verifiable < verifiable) or (checkworthy is not None and a.checkworthy < checkworthy):
                report_dropped(atom)  # in a check: result.skipped, like the claim filter's
            else:
                atoms.append(atom)
        return atoms
