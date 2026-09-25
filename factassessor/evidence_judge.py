"""(claim, snippets or pages) -> Evidence: does each passage support, refute, or not settle the claim?

`Judge` is the role: implement `judge(claim, docs)`. `LayaJudge` uses the local Laya model; any other `Judge`
(e.g. a GLiNER2.5-decide judge) drops in as `FactAssessor(judge=...)`.
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
import copy
import threading
from typing import Any

from factassessor.laya import LayaRunner
from factassessor.passages import chunk, top_passages
from factassessor.schema import Evidence

# Evidence-first state + this wording: 13/15 on our judge benchmark; claim-first variants topped out at 9/15.
QUESTION = {
    "stance": {
        "type": "choice",
        "instructions": "According to `evidence`, is `claim` true or false?",
        "criteria": {
            "supports": "true: the evidence says the same thing as the claim",
            "refutes": "false: the evidence says something different that makes the claim wrong",
            "not_enough_info": "unknown: the evidence does not mention what the claim is about",
        },
    }
}


class Judge(ABC):
    """Role: does each passage support, refute, or not settle a claim? Implement `judge`.

    `docs` are search hits {"url", "title", "snippet"} and/or crawled pages {"url", "title", "text"}: judge snippets
    as-is and cut pages down to what fits your model. Optional `aload`/`aclose` warm up / release the model.
    """

    @abstractmethod
    async def judge(self, claim: str, docs: list[dict[str, Any]]) -> list[Evidence]: ...

    async def aload(self) -> None:
        pass

    async def aclose(self) -> None:
        pass


class LayaJudge(Judge):
    """Every (claim, passage) pair is one Laya decision; the shared runner batches them across all atoms."""

    def __init__(
        self,
        laya: LayaRunner | None = None,
        model: str = "english",
        # Frozen-evidence benchmark (12 gold claims): 1 x 128 tokens matched 3 x ~290 on accuracy (10/12) at 1/3
        # the judge time. On MPS cost ~ pairs x tokens, and extra passages mostly added stray strong verdicts.
        passages_per_page: int = 1,
        passage_tokens: int | None = 128,
    ) -> None:
        self.laya = laya or LayaRunner()
        self.model = model
        self.passages_per_page = passages_per_page
        self.passage_tokens = passage_tokens
        # HF fast tokenizers aren't thread-safe ("Already borrowed"): chunking gets its own copy, used under a lock,
        # so it never touches the tokenizer Laya is using on its own thread.
        self._tok: Any = None
        self._room = 0  # evidence tokens available before subtracting the claim
        self._tok_lock = threading.Lock()

    async def aload(self) -> None:
        await self.laya.agent(self.model)

    async def judge(self, claim: str, docs: list[dict[str, Any]]) -> list[Evidence]:
        """docs: Serper hits {"url", "title", "snippet"} or crawled pages {"url", "title", "text"}.
        Snippets are judged as-is; pages are chunked with Laya's tokenizer to its exact budget first."""
        passages = await self._passages(claim, docs)
        results = await self.laya.predict_batch(
            [{"state": {"evidence": p["text"], "claim": claim}, "questions": QUESTION, "model": self.model} for p in passages]
        )
        evidence = []
        for p, r in zip(passages, results):
            probs = r["answers"]["stance"]["probabilities"]
            label = max(probs, key=probs.get)
            evidence.append(Evidence(**p, label=label, prob=probs[label]))
        return evidence

    async def _passages(self, claim: str, docs: list[dict[str, Any]]) -> list[dict[str, Any]]:
        passages = [
            {"url": d["url"], "title": d.get("title", ""), "text": d["snippet"], "source": "snippet"}
            for d in docs
            if "text" not in d and d.get("snippet")
        ]
        pages = [d for d in docs if "text" in d]
        if pages:
            if self._tok is None:
                agent = await self.laya.agent(self.model)
                self._tok = copy.deepcopy(agent.tok)
                # Laya packs [CLS] question+options (<= head_max_len) [SEP] state [SEP] into max_len; margin covers
                # the "evidence:"/"claim:" keys and re-tokenization drift.
                self._room = agent.cfg.get("max_len", 512) - agent.cfg.get("head_max_len", 192) - 16
            # One worker thread per page: pages finish independently; BM25 runs outside the tokenizer lock.
            chosen = await asyncio.gather(*(asyncio.to_thread(self._top_chunks, claim, p["text"]) for p in pages))
            for page, texts in zip(pages, chosen):
                passages += [{"url": page["url"], "title": page.get("title", ""), "text": t, "source": "page"} for t in texts]
        return passages

    def _top_chunks(self, claim: str, text: str) -> list[str]:
        with self._tok_lock:
            budget = self._room - len(self._tok(claim, add_special_tokens=False)["input_ids"])
            if self.passage_tokens:
                budget = min(budget, self.passage_tokens)
            chunks = chunk(text, self._tok, max_tokens=budget, overlap=budget // 4)
        return top_passages(claim, chunks, k=self.passages_per_page)
