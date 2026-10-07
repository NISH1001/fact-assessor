"""DecisionJudge: every (claim, passage) pair is one decision of a `DecisionRunner` (Laya by default)."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from factassessor.decisions import DecisionRequest, DecisionRunner, Question, SystemOneRunner
from factassessor.judges._base import Judge
from factassessor.passages import normalize_text, word_windows
from factassessor.rankers import BM25Ranker, Ranker
from factassessor.schema import Evidence

PASSAGES_PER_PAGE = 3  # windows of each page the judge sees: 3 vs 1 gave +0.10 F1 on the paper eval

# Evidence-first state + this wording: 13/15 on our judge benchmark with Laya; claim-first variants topped out at 9/15.
QUESTION = {
    "stance": Question(
        type="choice",
        instructions="According to `evidence`, is `claim` true or false?",
        criteria={
            "supports": "true: the evidence says the same thing as the claim",
            "refutes": "false: the evidence says something different that makes the claim wrong",
            "not_enough_info": "unknown: the evidence does not mention what the claim is about",
        },
    )
}


class DecisionJudge(Judge):
    """Every (claim, passage) pair is one decision; the runner batches them across all claims and pages.

    Pages are cut into `passage_words`-word windows (90 ~ Laya's 128 tokens, the size that measured best; the
    text's own words, so any runner sees the same passages) and the ranker picks `passages_per_page` of them (3:
    +0.10 F1 on the paper eval, ~3s more per text). A snippet is judged whole unless it is really a page (some
    DuckDuckGo ones run to 3,000 tokens; the claim came after the evidence and got truncated), then cut the same
    way. `chunk` replaces the window cutter (text -> passages), e.g. a token-exact one for a specific model.
    `passage_chars`: a window longer than this is cut into pieces (words are counted by whitespace, so a page with
    almost none, like an 800,000-character gist, made one "90-word" window over Jev's input limit). The default,
    22 x `passage_words`, is about 3.4x a typical 90-word window (580 characters); 0.06% of 1M real windows are longer.
    `concurrency` (claims at once) comes from the runner when it sets one (GLiNER: 3).
    """

    def __init__(
        self,
        runner: DecisionRunner | None = None,  # default: Jev (SystemOneRunner, OPENROUTER_API_KEY); LayaRunner() runs locally
        passages_per_page: int = PASSAGES_PER_PAGE,
        passage_words: int = 90,
        ranker: Ranker | None = None,  # which passages of a page the judge sees; default BM25 (word overlap)
        chunk: Callable[[str], list[str]] | None = None,
        passage_chars: int | None = None,  # a passage's most characters; default 22 x passage_words (~2,000 for 90)
    ) -> None:
        self.runner = runner or SystemOneRunner()
        self.passages_per_page = passages_per_page
        self.passage_words = passage_words
        self.passage_chars = passage_chars or 22 * passage_words
        self.ranker = ranker or BM25Ranker()
        self.chunk = chunk or (lambda text: word_windows(text, passage_words, passage_words // 4, max_chars=self.passage_chars))
        self.concurrency = getattr(runner, "concurrency", None)

    async def aload(self) -> None:
        if hasattr(self.runner, "aload"):
            await self.runner.aload()

    async def aclose(self) -> None:
        if hasattr(self.runner, "aclose"):
            await self.runner.aclose()

    async def judge(self, claim: str, docs: list[dict[str, Any]]) -> list[Evidence]:
        """docs: search hits {"url", "title", "snippet"} or crawled pages {"url", "title", "text"}."""
        # one notation for units and symbols on both sides: "ha⁻¹" vs "ha−1" must not look like different numbers
        claim = normalize_text(claim)
        passages = self._passages(claim, docs)
        if not passages:
            return []
        responses = await self.runner.predict(
            [DecisionRequest(state={"evidence": p["text"], "claim": claim}, questions=QUESTION) for p in passages]
        )
        evidence = []
        for p, r in zip(passages, responses):
            answer = r.answers["stance"]
            evidence.append(Evidence(**p, label=answer.label, prob=answer.probabilities[answer.label]))
        return evidence

    def _passages(self, claim: str, docs: list[dict[str, Any]]) -> list[dict[str, Any]]:
        passages: list[dict[str, Any]] = []
        for d in docs:
            base = {"url": d["url"], "title": d.get("title") or ""}
            if "text" in d:  # a crawled page: the ranker's best windows
                chunks = self.chunk(normalize_text(d["text"]))
                passages += [{**base, "text": t, "source": "page"} for t in self.ranker.top(claim, chunks, self.passages_per_page)]
            elif d.get("snippet"):
                text = normalize_text(d["snippet"])
                if len(text.split()) > 2 * self.passage_words or len(text) > self.passage_chars:  # a snippet that is really a page
                    text = self.ranker.top(claim, self.chunk(text), 1)[0]
                passages.append({**base, "text": text, "source": "snippet"})
        return passages
