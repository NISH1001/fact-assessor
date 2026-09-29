"""LayaJudge: every (claim, passage) pair is one decision of the local Laya model (the default judge)."""

from __future__ import annotations

import asyncio
import copy
import threading
from typing import Any

from factassessor.judges._base import Judge
from factassessor.laya import laya_runner
from factassessor.passages import chunk, normalize_text, top_passages
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


class LayaJudge(Judge):
    """Every (claim, passage) pair is one Laya decision; the shared runner batches them across all atoms."""

    def __init__(
        self,
        model: str = "english",  # Laya checkpoint: english | multilingual | typed-decisions
        device: str = "auto",
        # Frozen-evidence benchmark (12 gold claims): 1 x 128 tokens matched 3 x ~290 on accuracy (10/12) at 1/3
        # the judge time. On MPS cost ~ pairs x tokens, and extra passages mostly added stray strong verdicts.
        passages_per_page: int = 1,
        passage_tokens: int | None = 128,
        runner: Any = None,  # tests inject a fake; normally the process's shared Laya runner for this device
    ) -> None:
        self.model = model
        self.device = device
        self._runner = runner
        self.passages_per_page = passages_per_page
        self.passage_tokens = passage_tokens
        # HF fast tokenizers aren't thread-safe ("Already borrowed"): chunking gets its own copy, used under a lock,
        # so it never touches the tokenizer Laya is using on its own thread.
        self._tok: Any = None
        self._room = 0  # evidence tokens available before subtracting the claim
        self._tok_lock = threading.Lock()

    async def aload(self) -> None:
        await self._laya().agent(self.model)

    def _laya(self) -> Any:
        return self._runner or laya_runner(self.device)

    async def judge(self, claim: str, docs: list[dict[str, Any]]) -> list[Evidence]:
        """docs: Serper hits {"url", "title", "snippet"} or crawled pages {"url", "title", "text"}.
        Snippets are judged as-is; pages are chunked with Laya's tokenizer to its exact budget first."""
        # one notation for units and symbols on both sides: "ha⁻¹" vs "ha−1" must not look like different numbers
        claim = normalize_text(claim)
        docs = [{**d, **{k: normalize_text(d[k]) for k in ("snippet", "text") if d.get(k)}} for d in docs]
        passages = await self._passages(claim, docs)
        results = await self._laya().predict_batch(
            [{"state": {"evidence": p["text"], "claim": claim}, "questions": QUESTION, "model": self.model} for p in passages]
        )
        evidence = []
        for p, r in zip(passages, results):
            probs = r["answers"]["stance"]["probabilities"]
            label = max(probs, key=probs.get)
            evidence.append(Evidence(**p, label=label, prob=probs[label]))
        return evidence

    async def _passages(self, claim: str, docs: list[dict[str, Any]]) -> list[dict[str, Any]]:
        snippets = [d for d in docs if "text" not in d and d.get("snippet")]
        pages = [d for d in docs if "text" in d]
        if not (snippets or pages):
            return []
        await self._load_tokenizer()
        # Snippets are judged whole when they fit next to the claim. Some (DuckDuckGo) run to 3,000+ tokens, and
        # since the evidence comes first, Laya's truncation then cut the claim itself: a true and a false claim got
        # identical answers. Those are cut to their best passage, like pages.
        fitted = await asyncio.gather(*(asyncio.to_thread(self._fit, claim, d["snippet"]) for d in snippets))
        passages = [
            {"url": d["url"], "title": d.get("title") or "", "text": t, "source": "snippet"} for d, t in zip(snippets, fitted)
        ]
        # One worker thread per page: pages finish independently; BM25 runs outside the tokenizer lock.
        chosen = await asyncio.gather(*(asyncio.to_thread(self._top_chunks, claim, p["text"]) for p in pages))
        for page, texts in zip(pages, chosen):
            passages += [{"url": page["url"], "title": page.get("title") or "", "text": t, "source": "page"} for t in texts]
        return passages

    async def _load_tokenizer(self) -> None:
        if self._tok is None:
            agent = await self._laya().agent(self.model)
            self._tok = copy.deepcopy(agent.tok)
            # Laya packs [CLS] question+options (<= head_max_len) [SEP] state [SEP] into max_len; margin covers
            # the "evidence:"/"claim:" keys and re-tokenization drift.
            self._room = agent.cfg.get("max_len", 512) - agent.cfg.get("head_max_len", 192) - 16

    def _fit(self, claim: str, snippet: str) -> str:
        """The snippet itself if it fits next to the claim, else its most relevant passage."""
        with self._tok_lock:
            room = self._room - len(self._tok(claim, add_special_tokens=False)["input_ids"])
            fits = len(self._tok(snippet, add_special_tokens=False)["input_ids"]) <= room
        if fits:
            return snippet
        best = self._top_chunks(claim, snippet)
        return best[0] if best else snippet

    def _top_chunks(self, claim: str, text: str) -> list[str]:
        with self._tok_lock:
            budget = self._room - len(self._tok(claim, add_special_tokens=False)["input_ids"])
            if self.passage_tokens:
                budget = min(budget, self.passage_tokens)
            chunks = chunk(text, self._tok, max_tokens=budget, overlap=budget // 4)
        return top_passages(claim, chunks, k=self.passages_per_page)
