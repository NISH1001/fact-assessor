"""End-to-end evaluation harness for FactAssessor on synthetic texts with known truth.

    uv run python scripts/eval.py build                          # data/fact_pairs.json -> data/eval_texts.jsonl
    uv run --extra ddg python scripts/eval.py record             # atomize, search, crawl once -> data/evidence/
    uv run --extra gliner python scripts/eval.py run all         # laya, gliner, llm on the recorded evidence + comparison
    uv run --extra ddg python scripts/eval.py run laya --live    # live atomizer/search/crawl (web results drift)
    uv run python scripts/eval.py report                         # rebuild the comparison + plots from data/results/

    # paired original/corrupted passages from a workbook (external data: kept in tmp/paired/, gitignored)
    uv run --with openpyxl --with pandas python scripts/eval.py build --dataset paired --xlsx "tmp/<workbook>.xlsx" --sheet <name>
    uv run python scripts/eval.py record --dataset paired --searcher searxng
    uv run python scripts/eval.py run laya --dataset paired

Texts: each pair in data/fact_pairs.json is a true sentence and a false variant with one detail changed (date,
number, place, person). `build` samples them into 27 texts = {true, false, mixed ~50/50} x {short 2-4, medium 5-10,
long 15-25 sentences} x 3 (seed 7), so every sentence has a label; an atom takes the label of the sentence its span
starts in.

Evidence: live web results change between runs, so `record` fixes the atoms (LLMAtomizer), the top 5 unblocked hits
per atom (DuckDuckGo by default; `--searcher searxng --searxng-url ...` or `--searcher serper`), and every hit's page (crawl4ai, live timeout, so pages that
time out live are missing here too). All atoms are searched, since claim filters differ per variant. Stored in
data/evidence/evidence.json.gz (gitignored: third-party page text). Resumes where it stopped.

Variants (only the models differ):
    laya    DecisionClaimFilter + DecisionJudge on Laya (the default pipeline)
    gliner  the same filter and judge on GlinerRunner
    llm     no claim filter + DecisionJudge on LLMRunner (gpt-6-luna): everything after the atomizer is the LLM

Paired dataset (`--dataset paired`): pairs of an original passage (every sentence true) and a corrupted copy, read from
a workbook sheet (`--sheet`). Corrupted sentences are found by diffing the pair: sentences changed from the original
are false, unchanged ones true. A reference system's per-row F1s from the sheet (checked against the source paper, and
on the open web) are kept with each text, and the report scores FactAssessor the same way: supported vs not supported
per claim, accuracy / precision / recall / NPV / F1 per passage, averaged over passages. The reference's labels are
human annotations of its own atoms; ours come from the sentence diff, so the numbers are comparable, not identical.

Warm-up before every run, timed and reported separately. Texts run one at a time. On recorded evidence the atomizer,
search, and crawl return instantly, so latency is the claim filter + judge + policy (the models' cost); `--live`
measures the whole pipeline. Results: data/results/eval-<variant>[-live].{json,md}, eval-comparison.md, and plots
(eval-comparison.png, eval-verdicts.png) of the recorded runs.
"""

from __future__ import annotations

import argparse
import asyncio
import gzip
import json
import random
import statistics
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import httpx

from factassessor import Atom, Crawl4AICrawler, DecisionClaimFilter, DecisionJudge, FactAssessor, LayaRunner, LLMRunner, SerperSearcher
from factassessor._llm import reasoning_off
from factassessor.atomizer import Atomizer, LLMAtomizer
from factassessor.crawlers import Crawler, Fetch, HTTPXCrawler
from factassessor.resolvers import ArxivResolver, CompositeResolver, OpenAlexResolver, locations
from factassessor.verify import read_first
from factassessor.pipeline import Take
from factassessor.search import DuckDuckGoSearcher, Searcher, SearxngSearcher, is_blocked, not_blocked

ROOT = Path(__file__).parent.parent
DATA = ROOT / "data"
TEXTS = DATA / "eval_texts.jsonl"
EVIDENCE = DATA / "evidence" / "evidence.json.gz"
RESULTS = DATA / "results"
LENGTHS = {"short": (2, 4), "medium": (5, 10), "long": (15, 25)}  # sentences per text
KINDS = ("true", "false", "mixed")
DATASET = "synthetic"


def use_evidence(tag: str | None) -> None:
    """A named evidence file next to the default one (evidence-<tag>.json.gz), e.g. from `requery`."""
    global EVIDENCE
    if tag:
        EVIDENCE = EVIDENCE.with_name(f"evidence-{tag}.json.gz")


def use_dataset(name: str) -> None:
    """Point the harness at a dataset's texts, evidence, and results."""
    global TEXTS, EVIDENCE, RESULTS, KINDS, DATASET
    DATASET = name
    if name == "paired":  # external data: never committed (tmp/ is gitignored)
        base = ROOT / "tmp" / "paired"
        TEXTS, EVIDENCE, RESULTS, KINDS = base / "texts.jsonl", base / "evidence.json.gz", base / "results", ("original", "corrupted")
VARIANTS = {"laya": "Laya filter + Laya judge", "laya-nofilter": "Laya judge, no claim filter",
            "gliner": "GLiNER filter + GLiNER judge", "llm": "LLM judge, no claim filter"}
LLM_MODEL = "openai:gpt-5-nano"  # the cheapest OpenAI model ($0.05 in / $0.40 out per 1M tokens, Sept 2026): queries, LLM judge
ATOMIZER_MODEL = "openai:gpt-6-luna"  # $0.10 / $0.50, reasoning off; gpt-5-nano returns whole sentences instead of atoms
TOP_K = 5
WARM_TEXT = "The Moon orbits the Earth. Mount Fuji is the highest mountain in Japan."  # not in the eval set


# --- texts -----------------------------------------------------------------------------------------------------

def build_texts(pairs: list[dict[str, str]], seed: int = 7, per_cell: int = 3) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    texts = []
    for length, (lo, hi) in LENGTHS.items():
        for kind in KINDS:
            for i in range(per_cell):
                n = rng.randint(lo, hi)
                picked = rng.sample(pairs, n)  # no fact twice in one text
                if kind == "mixed":  # about half and half, at least one of each
                    k = max(1, min(n - 1, round(n / 2)))
                    truths = [True] * k + [False] * (n - k)
                    rng.shuffle(truths)
                else:
                    truths = [kind == "true"] * n
                sentences, offset = [], 0
                for pair, is_true in zip(picked, truths):
                    s = pair["true"] if is_true else pair["false"]
                    sentences.append({"text": s, "true": is_true, "span": [offset, offset + len(s)]})
                    offset += len(s) + 1  # the joining space
                texts.append({"id": f"{length}-{kind}-{i}", "kind": kind, "length": length,
                              "text": " ".join(s["text"] for s in sentences), "sentences": sentences})
    return texts


_ABBREVIATIONS = ("et al.", "e.g.", "i.e.", "cf.", "vs.", "approx.", "ca.", "fig.", "figs.", "eq.", "eqs.", "no.", "ref.",
                  "refs.", "sect.", "tab.", "dr.", "mr.", "ms.", "st.")


def _sentences(text: str) -> list[str]:
    """Sentence split for scientific text: a break after "et al.", "e.g.", "Fig." or an initial ("J.") is undone."""
    import re

    pieces = [p.strip() for p in re.split(r"(?<=[.!?])\s+(?=[A-Z(\[])", " ".join(str(text).split())) if p.strip()]
    merged: list[str] = []
    for piece in pieces:
        if merged and (merged[-1].lower().endswith(_ABBREVIATIONS) or re.search(r"(?:^|[^A-Za-z])[A-Z]\.$", merged[-1])):
            merged[-1] += " " + piece
        else:
            merged.append(piece)
    return merged


def _length(n: int) -> str:
    return "short" if n <= LENGTHS["short"][1] else "medium" if n <= LENGTHS["medium"][1] else "long"


def build_paired(xlsx: str, sheet_name: str) -> list[dict[str, Any]]:
    """Original + corrupted text per workbook pair, sentence labels from the diff, the reference system's F1s."""
    import difflib

    import pandas as pd

    sheet = pd.read_excel(xlsx, sheet_name=sheet_name)
    sheet = sheet[pd.to_numeric(sheet.iloc[:, 0], errors="coerce").notna()]
    num = lambda v: None if pd.isna(pd.to_numeric(v, errors="coerce")) else float(pd.to_numeric(v, errors="coerce"))  # noqa: E731
    texts = []
    for pair, (_, row) in enumerate(sheet.iterrows(), start=1):  # not S.No: it restarts at 1 for each SME
        original, corrupted = _sentences(row.iloc[5]), _sentences(row.iloc[7])
        unchanged = set()
        for op, _, _, j1, j2 in difflib.SequenceMatcher(None, original, corrupted, autojunk=False).get_opcodes():
            if op == "equal":
                unchanged.update(range(j1, j2))
        for kind, sents, truths, cols in (
            ("original", original, [True] * len(original), (13, 15)),
            ("corrupted", corrupted, [i in unchanged for i in range(len(corrupted))], (14, 16)),
        ):
            spans, offset = [], 0
            for sent, true in zip(sents, truths):
                spans.append({"text": sent, "true": true, "span": [offset, offset + len(sent)]})
                offset += len(sent) + 1
            texts.append({
                "id": f"fr{pair:02d}-{kind}", "kind": kind, "length": _length(len(sents)), "pair": pair,
                "text": " ".join(sents), "sentences": spans, "source": str(row.iloc[2]),
                "reference": {"in_domain_f1": num(row.iloc[cols[0]]), "web_f1": num(row.iloc[cols[1]])},
            })
    return texts


def load_texts() -> list[dict[str, Any]]:
    return [json.loads(line) for line in TEXTS.read_text().splitlines()]


def label_at(text: dict[str, Any], offset: int) -> bool | None:
    """Ground truth of the sentence containing char `offset` (an atom's span start)."""
    for s in text["sentences"]:
        if s["span"][0] <= offset < s["span"][1]:
            return s["true"]
    return None


# --- evidence: record once, replay per variant -----------------------------------------------------------------

def load_evidence() -> dict[str, Any]:
    if not EVIDENCE.exists():
        return {"searcher": None, "texts": {}, "pages": {}}
    return json.loads(gzip.decompress(EVIDENCE.read_bytes()))


def save_evidence(evidence: dict[str, Any]) -> None:
    EVIDENCE.parent.mkdir(parents=True, exist_ok=True)
    EVIDENCE.write_bytes(gzip.compress(json.dumps(evidence, ensure_ascii=False).encode()))


def make_searcher(name: str, searxng_url: str, search_type: str = "general") -> Searcher:
    """`name`: ddg (no key), searxng (self-hosted, no key), or serper (API key and credits).
    `search_type`: general (the web) or science (Serper -> Google Scholar; SearXNG -> its scholarly engines);
    DuckDuckGo only does general."""
    science = search_type == "science"
    if name == "searxng":
        return SearxngSearcher(searxng_url, num=2 * TOP_K, search_type=search_type,
                               timeout=20.0 if science else 5.0, hedge_after=None if science else 1.2)
    if name == "serper":  # no hedging: a duplicate request would cost a second credit
        return SerperSearcher(num=2 * TOP_K, hedge_after=None, search_type=search_type, timeout=15.0 if science else 5.0)
    if science:
        raise SystemExit("--search-type science needs --searcher serper or searxng")
    return DuckDuckGoSearcher(num=2 * TOP_K)


async def record(searcher_name: str, searxng_url: str, llm_model: str = LLM_MODEL, search_type: str = "general") -> None:
    searcher = make_searcher(searcher_name, searxng_url, search_type)
    atomizer, crawler = LLMAtomizer(ATOMIZER_MODEL), Crawl4AICrawler()
    await crawler.start()
    evidence = load_evidence()
    evidence["searcher"] = searcher_name if search_type == "general" else f"{searcher_name} {search_type}"
    evidence["atomizer"] = llm_model
    slots = asyncio.Semaphore(3)  # DuckDuckGo rate-limits bursts

    async def search(query: str) -> list[dict[str, Any]]:
        for attempt in range(4):  # errors and empty results both retry: DuckDuckGo throttles by returning nothing
            try:
                async with slots:
                    hits = await searcher.search(query)
                if hits:
                    return [h for h in hits if not is_blocked(h["url"])][:TOP_K]
                reason = "no results"
            except Exception as exc:
                reason = f"{exc!r:.60}"
            print(f"  search retry {attempt + 1} ({reason})", flush=True)
            await asyncio.sleep(2 * (attempt + 1))
        return []

    for ex in load_texts():
        if ex["id"] in evidence["texts"]:
            continue
        start = time.perf_counter()
        atoms = await atomizer.atomize(ex["text"])
        hits = await asyncio.gather(*(search(a.text) for a in atoms))
        urls = list({h["url"] for hs in hits for h in hs} - evidence["pages"].keys())
        pages = [f.page if f else None for f in await asyncio.gather(*(crawler.crawl(u) for u in urls))]
        evidence["pages"].update(zip(urls, pages))
        evidence["texts"][ex["id"]] = {"atoms": [{"text": a.text, "span": list(a.span)} for a in atoms],
                                       "hits": {a.text: hs for a, hs in zip(atoms, hits)}}
        save_evidence(evidence)
        print(f"{ex['id']:16s} {len(atoms):2d} atoms, {sum(map(len, hits)):3d} hits, "
              f"{sum(p is not None for p in pages)}/{len(urls)} new pages, {time.perf_counter() - start:.1f}s", flush=True)
    await crawler.stop()


async def fetch_papers(xlsx: str, links_sheet: str, min_words: int = 500) -> None:
    """paired: each pair's source paper (full text), for in-domain checks against the paper itself.
    Tries the workbook's open-access link (plain download: PDF or HTML, then the browser), then OpenAlex's
    open-access copies via the DOI. Writes papers.json next to the texts (gitignored)."""
    import re

    import pandas as pd

    from factassessor.extract import extract

    norm = lambda v: " ".join(str(v).split())  # noqa: E731
    new = pd.read_excel(xlsx, sheet_name=links_sheet)  # a sheet with the source's open-access link per row
    links = {norm(r.iloc[2]): str(r.iloc[3]).strip() for _, r in new.iterrows() if pd.notna(r.iloc[3])}
    sources = list(dict.fromkeys(t["source"] for t in load_texts()))
    out_path = TEXTS.parent / "papers.json"
    papers = json.loads(out_path.read_text()) if out_path.exists() else {}
    browser = Crawl4AICrawler(timeout=20)
    open_access = paper_crawler(min_words=min_words, timeout=30, pdf_timeout=30)
    http = httpx.AsyncClient(timeout=30, follow_redirects=True, headers={"User-Agent": "Mozilla/5.0 (fact-assessor eval)"})

    async def download(url: str) -> str:
        try:
            r = await http.get(url)
            if not 200 <= r.status_code < 300:
                return ""
            got = await asyncio.to_thread(extract, r.content, r.headers.get("content-type", ""), r.charset_encoding)
            return got[1] if got else ""
        except Exception:
            return ""

    for source in sources:
        if len(papers.get(source, {}).get("text", "").split()) >= min_words:
            continue
        link = links.get(norm(source), "")
        doi = re.search(r"10\.\d{4,9}/[^\s)\]]+", source + " " + link)
        tried = []
        text, via = "", ""
        if link.startswith("http"):
            async def render(url: str) -> str:
                f = await browser.crawl(url)
                return f.page["text"] if f else ""

            for name, get in (("download", download), ("browser", render)):
                got = await get(link)
                tried.append(f"{name}:{len(got)}")
                if len(got.split()) >= min_words:
                    text, via = got, f"{name} {link}"
                    break
        if not text and doi:
            f = await open_access.crawl(f"https://doi.org/{doi.group(0).rstrip('.')}")
            page = f.page if f else None
            tried.append(f"open access:{len(page['text']) if page else 0}")
            if page:
                text, via = page["text"], f"open access {doi.group(0)}"
        papers[source] = {"text": text, "via": via}
        out_path.write_text(json.dumps(papers, ensure_ascii=False))
        print(f"{'OK  ' if text else 'none'} {len(text):7d} chars  {source[:70]}  [{', '.join(tried)}]", flush=True)
    await browser.stop(); await open_access.stop(); await http.aclose()
    have = sum(len(p["text"].split()) >= min_words for p in papers.values())
    pairs = {t["pair"] for t in load_texts() if len(papers.get(t["source"], {}).get("text", "").split()) >= min_words}
    print(f"\nfull text for {have} of {len(sources)} papers -> {len(pairs)} of 50 pairs usable in-domain")


async def recrawl_open_access() -> None:
    """Pages that failed to crawl: read them over plain HTTP, papers from their free copies first (arXiv, DOIs via
    OpenAlex; PDFs too), as `CascadedCrawler(Crawl4AICrawler(), paper_crawler())` would have while recording.
    Keeps a backup."""
    import shutil

    evidence = load_evidence()
    backup = EVIDENCE.with_name(EVIDENCE.name.replace(".json.gz", ".before-open-access.json.gz"))
    if not backup.exists():
        shutil.copy(EVIDENCE, backup)
    todo = [u for u, page in evidence["pages"].items() if page is None]  # the crawler skips what it can't read
    crawler = paper_crawler()
    start = time.perf_counter()
    pages = [f.page if f else None for f in await asyncio.gather(*(crawler.crawl(u) for u in todo))]
    await crawler.stop()
    recovered = {u: p for u, p in zip(todo, pages) if p}
    evidence["pages"].update(recovered)
    evidence["crawler"] = "crawl4ai, then plain HTTP with resolvers (arXiv, OpenAlex; PDFs) for failed pages"
    save_evidence(evidence)
    print(f"{len(todo)} failed pages -> {len(recovered)} recovered (arXiv, PDFs, open access) in "
          f"{time.perf_counter() - start:.1f}s (backup: {backup.name})")


# LLM-written search queries (requery --queries llm)
QUERY_INSTRUCTIONS = """Write one web search query for checking whether the CLAIM is true. Keep its distinctive terms:
names, numbers, dates, places, and technical terms; drop filler words. Put an exact phrase in quotes only when the exact
wording matters. Return only the query.
Examples:
CLAIM: The Hubble Space Telescope was launched in 1990.
QUERY: Hubble Space Telescope launch year 1990
CLAIM: Secondary forests in the Neotropics regain 122 Mg/ha of aboveground biomass within 20 years.
QUERY: Neotropical secondary forest aboveground biomass recovery 20 years 122 Mg ha"""


async def requery(tag: str, searcher_name: str, searxng_url: str, llm_model: str = LLM_MODEL,
                  queries: str = "llm", pairs: int | None = None, search_type: str = "general") -> None:
    """Search the recorded atoms again: with another engine and/or LLM-written queries (`queries="llm"`) instead of the claim text (`queries="claim"`). Results are cached by query in the evidence
    file, so a query is only ever paid for once (reruns and bigger runs reuse it). New pages are crawled (browser,
    then open access); known pages are reused. Writes evidence-<tag>.json.gz; the original evidence is untouched."""
    from pydantic import BaseModel
    from pydantic_ai import Agent

    from factassessor import CascadedCrawler

    class Query(BaseModel):
        query: str

    base = load_evidence()
    use_evidence(tag)
    evidence = load_evidence() if EVIDENCE.exists() else {"texts": {}, "pages": dict(base["pages"])}
    evidence.setdefault("search_cache", {})
    evidence.update(searcher=searcher_name if search_type == "general" else f"{searcher_name} {search_type}", atomizer=base.get("atomizer"), queries_by=llm_model if queries == "llm" else "claim text",
                    crawler="crawl4ai, then plain HTTP with resolvers (arXiv, OpenAlex)")
    paid = 0
    agent = Agent(llm_model, output_type=Query, instructions=QUERY_INSTRUCTIONS, model_settings=llm_settings(llm_model))
    searcher = make_searcher(searcher_name, searxng_url, search_type)
    crawler = CascadedCrawler(Crawl4AICrawler(), paper_crawler())  # browser first, like the recorded evidence
    slots = asyncio.Semaphore(3)

    async def query_for(claim: str) -> str:
        if queries == "claim":
            return claim
        try:
            return (await agent.run(f"CLAIM: {claim}")).output.query.strip() or claim
        except Exception as exc:
            print(f"  query builder failed ({exc!r:.60}); using the claim", flush=True)
            return claim

    async def search(query: str) -> list[dict[str, Any]]:
        nonlocal paid
        if query in evidence["search_cache"]:  # already paid for
            return evidence["search_cache"][query]
        attempts = 2 if searcher_name == "serper" else 4  # Serper bills every request
        hits: list[dict[str, Any]] = []
        for attempt in range(attempts):
            try:
                async with slots:
                    paid += 1
                    found = await searcher.search(query)
                if found:
                    hits = [h for h in found if not is_blocked(h["url"])][:TOP_K]
                    break
            except Exception as exc:
                print(f"  search failed ({exc!r:.60})", flush=True)
            await asyncio.sleep(2 * (attempt + 1))
        evidence["search_cache"][query] = hits
        return hits

    order = [t["id"] for t in load_texts()]
    wanted = {t["id"] for t in load_texts() if pairs is None or t.get("pair", 0) <= pairs}
    for tid in order:
        rec = base["texts"].get(tid)
        if rec is None or tid not in wanted or tid in evidence["texts"]:
            continue
        start = time.perf_counter()
        claims = [a["text"] for a in rec["atoms"]]
        qs = await asyncio.gather(*(query_for(c) for c in claims))
        hits = [await search(q) for q in qs]  # one at a time: identical queries in a text hit the cache
        urls = list({h["url"] for hs in hits for h in hs} - evidence["pages"].keys())
        pages = [f.page if f else None for f in await asyncio.gather(*(crawler.crawl(u) for u in urls))]
        evidence["pages"].update(zip(urls, pages))
        evidence["texts"][tid] = {"atoms": rec["atoms"], "hits": dict(zip(claims, hits)), "queries": dict(zip(claims, qs))}
        save_evidence(evidence)
        print(f"{tid:16s} {len(claims):2d} queries, {sum(map(len, hits)):3d} hits, {sum(p is not None for p in pages)}/{len(urls)} new pages, "
              f"{time.perf_counter() - start:.1f}s | e.g. {qs[0][:70]!r}", flush=True)
    print(f"searches paid this run: {paid} (cached queries: {len(evidence['search_cache'])})")
    await crawler.crawlers[0].stop()
    await crawler.crawlers[1].stop()


class RecordedAtomizer(Atomizer):
    def __init__(self, evidence: dict[str, Any]) -> None:
        by_id = {ex["id"]: ex["text"] for ex in load_texts()}
        self.atoms = {by_id[i]: t["atoms"] for i, t in evidence["texts"].items()}

    async def atomize(self, text: str) -> list[Atom]:
        return [Atom(id=i, text=a["text"], span=tuple(a["span"])) for i, a in enumerate(self.atoms[text])]


class RecordedSearcher(Searcher):
    def __init__(self, evidence: dict[str, Any]) -> None:
        self.hits = {q: hs for t in evidence["texts"].values() for q, hs in t["hits"].items()}

    async def search(self, query: str) -> list[dict[str, Any]]:
        return self.hits.get(query, [])


class RecordedCrawler(Crawler):
    def __init__(self, evidence: dict[str, Any]) -> None:
        self.pages = evidence["pages"]

    async def crawl(self, url: str) -> Fetch:
        return Fetch(url=url, crawler="recorded", page=self.pages.get(url))


# --- runs ------------------------------------------------------------------------------------------------------

class PaperReader(Crawler):
    """crawl(url) as the pipeline reads a hit with resolvers: its free copies (arXiv HTML or PDF, OpenAlex) first,
    then the page itself, over plain HTTP."""

    def __init__(self, min_words: int = 300, **http: Any) -> None:
        self.resolver = CompositeResolver(ArxivResolver(), OpenAlexResolver())
        self.http = HTTPXCrawler(**http)
        self.min_words = min_words

    async def crawl(self, url: str) -> Fetch:
        page = await read_first(self.http, url, locations(url, await self.resolver.resolve(url)), self.min_words)
        return Fetch(url=url, crawler=type(self).__name__, page=page)

    async def stop(self) -> None:
        await self.resolver.aclose()
        await self.http.stop()


def paper_crawler(**kwargs: Any) -> PaperReader:
    return PaperReader(**kwargs)


def llm_settings(model: str) -> dict[str, Any]:
    """Reasoning as low as the model allows (the library's own default for LLM steps)."""
    return reasoning_off(model)


def components(variant: str, llm_model: str = LLM_MODEL) -> tuple[Any, Any]:
    """(claim_filter, judge) for a variant."""
    if variant == "gliner":
        from factassessor import GlinerRunner  # one shared GLiNER model

        gliner = GlinerRunner()
        return DecisionClaimFilter(gliner), DecisionJudge(gliner)
    if variant == "llm":
        return None, DecisionJudge(LLMRunner(llm_model, model_settings=llm_settings(llm_model)))
    laya = LayaRunner()  # one model, one batch queue for both
    if variant == "laya-nofilter":  # check every atom
        return None, DecisionJudge(laya)
    return DecisionClaimFilter(laya), DecisionJudge(laya)


async def warm_up(fa: FactAssessor, claim_filter: Any, judge: Any, text: str) -> dict[str, Any]:
    """Load and exercise every component once; return the timings (seconds)."""
    t: dict[str, Any] = {}
    start = time.perf_counter()
    await fa.aload()
    t["load"] = time.perf_counter() - start
    atom = Atom(id=0, text="The Moon orbits the Earth.", span=(0, 26))
    for name, call in [
        ("claim_filter", (lambda: claim_filter.score(atom)) if claim_filter else None),
        ("judge", lambda: judge.judge(atom.text, [{"url": "warmup", "title": "", "snippet": "The Moon orbits the Earth."}])),
    ]:
        if call is None:
            continue
        for when in ("first", "warm"):
            start = time.perf_counter()
            await call()
            t[f"{name}_{when}"] = time.perf_counter() - start
    t["assess"] = []
    for _ in range(3):  # until a full check is no slower than the one before (connections, browser, LLM client)
        start = time.perf_counter()
        await fa.assess(text)
        t["assess"].append(time.perf_counter() - start)
        if len(t["assess"]) > 1 and t["assess"][-1] <= t["assess"][-2] * 1.2:
            break
    return t


async def run(variant: str, live: bool, timeout: float = 15.0, searcher: str = "searxng", searxng_url: str = "",
              llm_model: str = LLM_MODEL, limit: int | None = None, strong: float = 0.7,
              in_domain: bool = False, search_type: str = "general") -> dict[str, Any]:
    claim_filter, judge = components(variant, llm_model)
    texts = load_texts()[:limit]
    if in_domain:  # search the fetched source papers instead of the web
        from factassessor import DocumentSearcher, NoCrawler

        papers = json.loads((TEXTS.parent / "papers.json").read_text())
        docs = [{"url": src, "title": src[:100], "text": p["text"]} for src, p in papers.items() if len(p["text"]) >= 3000]
        have = {d["url"] for d in docs}
        texts = [t for t in texts if t["source"] in have]
        evidence = load_evidence()
        fa = FactAssessor(n_atoms=30, claim_filter=claim_filter, judge=judge, timeout=timeout, strong_evidence=strong,
                          atomizer=RecordedAtomizer(evidence), searcher=DocumentSearcher(docs), crawler=NoCrawler())
        warm_text, source = texts[0]["text"], f"in-domain ({len(docs)} source papers, DocumentSearcher top 5 passages)"
        live = False
    elif live:
        search = make_searcher(searcher, searxng_url, search_type) >> not_blocked() >> Take(TOP_K)  # as FactAssessor wires Serper
        fa = FactAssessor(n_atoms=30, claim_filter=claim_filter, judge=judge, timeout=timeout, searcher=search,
                          atomizer=LLMAtomizer(ATOMIZER_MODEL))
        warm_text, source = WARM_TEXT, f"live (LLMAtomizer, {searcher} search, crawl4ai)"
    else:
        evidence = load_evidence()
        missing = [ex["id"] for ex in texts if ex["id"] not in evidence["texts"]]
        if missing:
            raise SystemExit(f"no recorded evidence for {len(missing)} texts (run `eval.py record` first): {missing[:3]}...")
        fa = FactAssessor(n_atoms=30, claim_filter=claim_filter, judge=judge, timeout=timeout, strong_evidence=strong,
                          atomizer=RecordedAtomizer(evidence), searcher=RecordedSearcher(evidence), crawler=RecordedCrawler(evidence))
        warm_text, source = texts[0]["text"], f"recorded ({evidence['searcher']} search, {evidence.get('crawler', 'crawl4ai')})"
    warm = await warm_up(fa, claim_filter, judge, warm_text)
    print(f"[{variant}] warm-up: " + ", ".join(f"{k} {v:.2f}s" if isinstance(v, float) else f"{k} {[round(x, 1) for x in v]}s"
                                             for k, v in warm.items()), flush=True)
    rows = []
    for ex in texts:
        start, first, result = time.perf_counter(), None, None
        async for event in fa.stream(ex["text"]):
            if event.type == "claim_verified" and first is None:
                first = time.perf_counter() - start
            if event.type == "done":
                result = event.result
        total = time.perf_counter() - start
        assert result is not None
        atoms = [
            {"text": a.atom.text, "gold": label_at(ex, a.atom.span[0]), "verdict": a.verdict, "confidence": a.confidence,
             "error": a.error, "pages": sum(e.source == "page" for e in a.evidence)}
            for a in result.atoms
        ] + [{"text": s.text, "gold": label_at(ex, s.span[0]), "verdict": "skipped", "confidence": 0.0, "error": None, "pages": 0}
             for s in result.skipped]
        true_share = sum(s["true"] for s in ex["sentences"]) / len(ex["sentences"])
        rows.append({"id": ex["id"], "kind": ex["kind"], "length": ex["length"], "sentences": len(ex["sentences"]),
                     "true_share": true_share, "fact_score": result.fact_score, "latency_s": total,
                     "first_verdict_s": first, "atoms": atoms, "pair": ex.get("pair"), "reference": ex.get("reference")})
        decided = [a for a in atoms if a["verdict"] in ("supported", "refuted")]
        right = sum((a["verdict"] == "supported") == a["gold"] for a in decided)
        score = "–" if result.fact_score is None else f"{result.fact_score:.2f}"
        print(f"[{variant}] {ex['id']:16s} {len(ex['sentences']):2d} sent -> {len(atoms):2d} atoms | {right}/{len(decided)} right "
              f"of decided | score {score} vs {true_share:.2f} | first {first or 0:.1f}s total {total:.1f}s", flush=True)
    await fa.aclose()
    atomizer = llm_model if live else evidence.get("atomizer")
    return {"variant": variant, "live": live, "source": source, "timeout": timeout, "atomizer": atomizer,
            "llm_model": llm_model if variant == "llm" or live else None, "warmup": warm, "texts": rows}


# --- reports ---------------------------------------------------------------------------------------------------

def claim_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    atoms = [a for r in rows for a in r["atoms"]]
    labelled = [a for a in atoms if a["gold"] is not None]
    decided = [a for a in labelled if a["verdict"] in ("supported", "refuted")]
    correct = [a for a in decided if (a["verdict"] == "supported") == a["gold"]]
    false_claims = [a for a in labelled if a["gold"] is False]
    true_claims = [a for a in labelled if a["gold"] is True]
    pct = lambda n, d: n / d if d else None  # noqa: E731
    errors = [abs(r["fact_score"] - r["true_share"]) for r in rows if r["fact_score"] is not None]
    return {
        "claims": len(labelled),
        "coverage": pct(len(decided), len(labelled)),  # decided = supported or refuted
        "accuracy_decided": pct(len(correct), len(decided)),
        "accuracy_all": pct(len(correct), len(labelled)),  # contested/unverified/skipped count as wrong
        "false_supported": pct(sum(a["verdict"] == "supported" for a in false_claims), len(false_claims)),  # the dangerous error
        "true_refuted": pct(sum(a["verdict"] == "refuted" for a in true_claims), len(true_claims)),
        "score_error": statistics.mean(errors) if errors else None,
        "latency": statistics.median(r["latency_s"] for r in rows),
        "first": statistics.median(r["first_verdict_s"] or 0 for r in rows),
    }


def passage_metrics(atoms: list[dict[str, Any]]) -> dict[str, float | None]:
    """One passage, scored like the reference sheet: positive = supported. Zero when undefined, as in the sheet ("no true
    statements in the corrupted passage, so P, R, F1 = 0"); NPV None when nothing was marked not supported."""
    labelled = [a for a in atoms if a["gold"] is not None]
    tp = sum(a["gold"] and a["verdict"] == "supported" for a in labelled)
    fp = sum(not a["gold"] and a["verdict"] == "supported" for a in labelled)
    fn = sum(a["gold"] and a["verdict"] != "supported" for a in labelled)
    tn = sum(not a["gold"] and a["verdict"] != "supported" for a in labelled)
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return {"accuracy": (tp + tn) / len(labelled) if labelled else None, "precision": p, "recall": r,
            # NPV only means something when the passage has false claims (the sheet leaves it blank for originals)
            "npv": tn / (tn + fn) if tn + fn and tn + fp else None, "f1": 2 * p * r / (p + r) if p + r else 0.0,
            "atoms": len(labelled), "false_atoms": sum(not a["gold"] for a in labelled)}


def paired_report(res: dict[str, Any]) -> str:
    rows = res["texts"]
    mean = lambda vals: statistics.mean(v for v in vals if v is not None) if any(v is not None for v in vals) else None  # noqa: E731
    lines = [f"# FactAssessor vs the reference system: {len(rows) // 2} pairs", "",
             f"FactAssessor: {VARIANTS[res['variant']]}; evidence {res['source']}; atomizer {res.get('atomizer') or '?'}. "
             "Claim-level scoring as in the reference sheet (positive = supported; per passage, then averaged). "
             "The reference's labels are human annotations of its atoms; ours come from diffing each pair.", ""]
    lines += ["| | F1 original | F1 corrupted |", "|---|---|---|"]
    by_kind = {k: [r for r in rows if r["kind"] == k] for k in ("original", "corrupted")}
    fa = {k: [passage_metrics(r["atoms"]) for r in rs] for k, rs in by_kind.items()}
    for name, key in (("reference system, open web", "web_f1"), ("reference system, in-domain (source paper)", "in_domain_f1")):
        lines.append(f"| {name} | {fmt(mean([r['reference'][key] for r in by_kind['original']]), 'f')} | "
                     f"{fmt(mean([r['reference'][key] for r in by_kind['corrupted']]), 'f')} |")
    mode = "in-domain" if res["source"].startswith("in-domain") else "open web"
    ref_key = "in_domain_f1" if mode == "in-domain" else "web_f1"
    lines.append(f"| **FactAssessor, {mode}** | **{fmt(mean([m['f1'] for m in fa['original']]), 'f')}** | "
                 f"**{fmt(mean([m['f1'] for m in fa['corrupted']]), 'f')}** |")
    lines += ["", "## FactAssessor in detail", "", "| | accuracy | precision | recall | NPV | F1 | atoms / passage | false atoms | latency (median) |",
              "|---|---|---|---|---|---|---|---|---|"]
    for k, ms in fa.items():
        lat = statistics.median(r["latency_s"] for r in by_kind[k])
        lines.append(f"| {k} | {fmt(mean([m['accuracy'] for m in ms]), 'f')} | {fmt(mean([m['precision'] for m in ms]), 'f')} | "
                     f"{fmt(mean([m['recall'] for m in ms]), 'f')} | {fmt(mean([m['npv'] for m in ms]), 'f')} | "
                     f"{fmt(mean([m['f1'] for m in ms]), 'f')} | {statistics.mean(m['atoms'] for m in ms):.1f} | "
                     f"{sum(m['false_atoms'] for m in ms)} | {lat:.1f}s |")
    wins = {"better": 0, "tied": 0, "worse": 0}
    for k in fa:
        for r, m in zip(by_kind[k], fa[k]):
            ref = r["reference"][ref_key]
            if ref is not None:
                wins["better" if m["f1"] > ref + 1e-9 else "worse" if m["f1"] < ref - 1e-9 else "tied"] += 1
    pairs = {r["pair"]: r for r in by_kind["original"]}
    drops = [(pairs[r["pair"]]["fact_score"], r["fact_score"]) for r in by_kind["corrupted"] if r["pair"] in pairs]
    drops = [(o, c) for o, c in drops if o is not None and c is not None]
    lines += ["", f"Per passage vs the reference system ({mode}): FactAssessor F1 higher on {wins['better']}, tied on {wins['tied']}, "
              f"lower on {wins['worse']} (of {sum(wins.values())}). Corrupted copy scored below its original: "
              f"{sum(c < o for o, c in drops)} of {len(drops)} pairs.", ""]
    return "\n".join(lines)


def groups(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    out = {"all": rows}
    out.update({f"{k} ({lo}–{hi} sent.)": [r for r in rows if r["length"] == k] for k, (lo, hi) in LENGTHS.items()})
    out.update({f"{k} texts": [r for r in rows if r["kind"] == k] for k in KINDS})
    return out


def fmt(x: Any, kind: str = "pct") -> str:
    if x is None:
        return "–"
    return {"pct": f"{x:.0%}", "s": f"{x:.1f}s", "f": f"{x:.2f}"}[kind]


COLUMNS = [("claims", "claims", "n"), ("coverage", "coverage", "pct"), ("accuracy_decided", "accuracy (decided)", "pct"),
           ("accuracy_all", "accuracy (all)", "pct"), ("false_supported", "false → supported", "pct"),
           ("true_refuted", "true → refuted", "pct"), ("score_error", "fact-score error", "f"),
           ("latency", "latency (median)", "s"), ("first", "first verdict (median)", "s")]


def table(rows_by_name: dict[str, list[dict[str, Any]]], first_col: str) -> list[str]:
    lines = [f"| {first_col} | " + " | ".join(c[1] for c in COLUMNS) + " |", "|---" * (len(COLUMNS) + 1) + "|"]
    for name, rows in rows_by_name.items():
        m = claim_metrics(rows)
        lines.append(f"| {name} | " + " | ".join(str(m[k]) if kind == "n" else fmt(m[k], kind) for k, _, kind in COLUMNS) + " |")
    return lines + [""]


def verdict_table(rows: list[dict[str, Any]]) -> list[str]:
    counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for a in (a for r in rows for a in r["atoms"] if a["gold"] is not None):
        counts["true" if a["gold"] else "false"][a["verdict"]] += 1
    lines = ["| gold | supported | refuted | contested | unverified | skipped |", "|---|---|---|---|---|---|"]
    for gold in ("true", "false"):
        v = counts[gold]
        lines.append(f"| {gold} | {v['supported']} | {v['refuted']} | {v['contested']} | {v['unverified']} | {v['skipped']} |")
    return lines + [""]


def warmup_line(w: dict[str, Any]) -> str:
    parts = [f"load {w['load']:.1f}s"]
    for name in ("claim_filter", "judge"):
        if f"{name}_first" in w:
            parts.append(f"{name.replace('_', ' ')} {w[f'{name}_first']:.2f}s then {w[f'{name}_warm']:.2f}s")
    return ", ".join(parts) + f", full checks {', '.join(f'{x:.1f}s' for x in w['assess'])}"


def report(res: dict[str, Any]) -> str:
    rows = res["texts"]
    model = f" ({res['llm_model']})" if res.get("llm_model") else ""
    lines = [f"# FactAssessor eval: {VARIANTS[res['variant']]}{model}", "",
             f"{len(rows)} texts, {sum(r['sentences'] for r in rows)} sentences, {sum(len(r['atoms']) for r in rows)} atoms. "
             f"Evidence: {res['source']}. Per-claim timeout {res.get('timeout', 15.0):.0f}s. Warm-up (not counted): {warmup_line(res['warmup'])}.", "",
             *table(groups(rows), "group"), "## Verdicts by ground truth", "", *verdict_table(rows)]
    return "\n".join(lines)


def comparison() -> str:
    runs = {p.stem.removeprefix("eval-"): json.loads(p.read_text()) for p in sorted(RESULTS.glob("eval-*.json"))}
    recorded = {k: r for k, r in runs.items() if not r.get("live")}
    live = {k: r for k, r in runs.items() if r.get("live")}
    lines = ["# FactAssessor eval: comparison", "",
             "Same 27 synthetic texts (282 sentences, one fact each; true, false, and mixed texts; 2–25 sentences). "
             "Recorded runs share the exact same atoms, hits, and pages, so differences come from the claim filter and "
             "judge alone, and latency is their cost (atomizer, search, and crawl replay instantly). Live runs include "
             "the whole pipeline but see different web results.", ""]
    for title, rs in (("Recorded evidence", recorded), ("Live", live)):
        if not rs:
            continue
        lines.extend([f"## {title}", "", f"Evidence: {next(iter(rs.values()))['source']}.", ""])
        for group in groups(next(iter(rs.values()))["texts"]):
            lines.extend([f"### {group}", "", *table({f"{k} ({VARIANTS[r['variant']]}{', ' + r['llm_model'] if r.get('llm_model') else ''})": groups(r["texts"])[group]
                                                      for k, r in rs.items()}, "variant")])
        lines.extend(["### Warm-up (not counted above)", ""] + [f"- {k}: {warmup_line(r['warmup'])}" for k, r in rs.items()] + [""])
        for k, r in rs.items():
            lines.extend([f"### Verdicts by ground truth: {k}", "", *verdict_table(r["texts"])])
    return "\n".join(lines)


def plot() -> None:
    """data/results/eval-comparison.png (accuracy, errors, latency by length and composition) and
    eval-verdicts.png (verdict mix per ground truth), for the recorded runs."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    runs = [{**json.loads(p.read_text()), "key": p.stem.removeprefix("eval-")} for p in sorted(RESULTS.glob("eval-*.json"))]
    runs = [r for r in runs if not r.get("live")]
    if not runs:
        return
    names = [r["key"] for r in runs]
    palette = ("#4C72B0", "#DD8452", "#55A868", "#8172B3", "#937860", "#DA8BC3")
    colors = dict(zip(names, palette))
    width = 0.8 / len(runs)

    def bars(ax: Any, cats: list[str], value: Any, title: str, pct: bool = True) -> None:
        for i, r in enumerate(runs):
            vals = [value(r, c) for c in cats]
            xs = [j + (i - (len(runs) - 1) / 2) * width for j in range(len(cats))]
            b = ax.bar(xs, [v or 0 for v in vals], width, label=r["key"], color=colors[r["key"]])
            ax.bar_label(b, labels=[fmt(v, "pct" if pct else "s") for v in vals], fontsize=7, padding=1)
        ax.set_xticks(range(len(cats)), cats, fontsize=8)
        ax.set_title(title, fontsize=10)
        values = [v for r in runs for c in cats if (v := value(r, c))]
        if pct:
            ax.set_ylim(0, 1.12 if max(values, default=0) > 0.5 else max(0.1, 1.3 * max(values, default=0)))
            ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
        elif values and max(values) > 10 * min(values):
            ax.set_yscale("log")  # gliner is an order of magnitude slower
        ax.spines[["top", "right"]].set_visible(False)

    lengths, kinds = list(LENGTHS), list(KINDS)
    by = lambda r, key, c: claim_metrics([t for t in r["texts"] if t[key] == c])  # noqa: E731
    metrics = [("accuracy_decided", "accuracy\n(decided)"), ("accuracy_all", "accuracy\n(all)"), ("coverage", "coverage"),
               ("false_supported", "false →\nsupported"), ("true_refuted", "true →\nrefuted")]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5))
    bars(axes[0, 0], [m[1] for m in metrics], lambda r, c: claim_metrics(r["texts"])[dict((b, a) for a, b in metrics)[c]],
         "Overall (errors: lower is better)")
    bars(axes[0, 1], lengths, lambda r, c: by(r, "length", c)["accuracy_all"], "Accuracy (all claims) by length")
    bars(axes[0, 2], kinds, lambda r, c: by(r, "kind", c)["accuracy_all"], "Accuracy (all claims) by composition")
    bars(axes[1, 0], lengths, lambda r, c: by(r, "length", c)["false_supported"], "False claims marked supported, by length")
    bars(axes[1, 1], lengths, lambda r, c: by(r, "length", c)["latency"], "Median latency per text (models only)", pct=False)
    bars(axes[1, 2], lengths, lambda r, c: by(r, "length", c)["first"], "Median time to first verdict (models only)", pct=False)
    axes[1, 1].set_ylabel("seconds")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.suptitle(f"FactAssessor on 27 synthetic texts, same {runs[0]['source']} evidence", y=0.99, fontsize=12)
    describe = {r["key"]: VARIANTS[r["variant"]] + (f", {r['llm_model'].removeprefix('openai:')}" if r.get("llm_model") else "")
                for r in runs}
    fig.legend(handles, [f"{n}: {describe[n]}" for n in labels], loc="upper center", bbox_to_anchor=(0.5, 0.955),
               ncol=len(runs), frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    fig.savefig(RESULTS / "eval-comparison.png", dpi=130)

    verdicts = ["supported", "refuted", "contested", "unverified", "skipped"]
    vcolors = ["#55A868", "#C44E52", "#DD8452", "#8C8C8C", "#CCCCCC"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8), sharey=True)
    for ax, gold in zip(axes, (True, False)):
        left = [0.0] * len(runs)
        for v, color in zip(verdicts, vcolors):
            shares = []
            for r in runs:
                atoms = [a for t in r["texts"] for a in t["atoms"] if a["gold"] is gold]
                shares.append(sum(a["verdict"] == v for a in atoms) / len(atoms))
            b = ax.barh(names, shares, left=left, color=color, label=v)
            ax.bar_label(b, labels=[f"{s:.0%}" if s >= 0.04 else "" for s in shares], label_type="center", fontsize=8)
            left = [x + s for x, s in zip(left, shares)]
        ax.set_title(f"{'True' if gold else 'False'} claims: verdicts (want {'supported' if gold else 'refuted'})", fontsize=10)
        ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
        ax.spines[["top", "right"]].set_visible(False)
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, frameon=False)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(RESULTS / "eval-verdicts.png", dpi=130)
    plt.close("all")


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--dataset", choices=["synthetic", "paired"], default="synthetic")
    b = sub.add_parser("build", parents=[common], help="build the texts (synthetic: from data/fact_pairs.json)")
    b.add_argument("--xlsx", help="paired: the workbook")
    b.add_argument("--sheet", help="paired: the sheet with the passage pairs")
    rec = sub.add_parser("record", parents=[common], help="record atoms, hits, and pages once")
    for p in (rec, r := sub.add_parser("run", parents=[common], help="evaluate variants")):
        p.add_argument("--search-type", choices=["general", "science"], default="general", help="the web, or scholarly literature")
        p.add_argument("--searcher", choices=["ddg", "searxng", "serper"], default="searxng", help="for record and run --live")
        p.add_argument("--searxng-url", default="http://localhost:8080", help="your SearXNG instance (JSON enabled)")
        p.add_argument("--llm-model", default=LLM_MODEL, help="LLM for search queries and the llm judge (the atomizer is gpt-6-luna)")
    r.add_argument("variant", choices=[*VARIANTS, "all"])
    r.add_argument("--timeout", type=float, default=15.0, help="per-claim timeout (FactAssessor default 15s)")
    r.add_argument("--limit", type=int, help="only the first N texts (e.g. a quick live timing run)")
    r.add_argument("--strong", type=float, default=0.7, help="FactAssessor(strong_evidence=): min prob for a passage to count")
    r.add_argument("--in-domain", action="store_true", help="paired: check against the fetched source papers (fetch-papers)")
    r.add_argument("--evidence", help="replay evidence-<tag>.json.gz instead of the default (e.g. llmq from requery)")
    r.add_argument("--live", action="store_true", help="live atomizer, search (--searcher), and crawling instead of recorded evidence")
    sub.add_parser("report", parents=[common], help="rebuild data/results/eval-comparison.md and the plots")
    rc = sub.add_parser("recrawl", parents=[common], help="re-read failed pages over plain HTTP, papers via arXiv / OpenAlex (needs --extra pdf)")
    rc.add_argument("--evidence", help="evidence-<tag>.json.gz instead of the default")
    fp = sub.add_parser("fetch-papers", parents=[common], help="paired: each pair's source paper, for in-domain checks")
    fp.add_argument("--xlsx", required=True, help="the workbook")
    fp.add_argument("--links-sheet", required=True, help="the sheet with each source's open-access link")
    rq = sub.add_parser("requery", parents=[common], help="search the recorded atoms again with LLM-written queries")
    rq.add_argument("--tag", default="llmq", help="evidence-<tag>.json.gz (default llmq)")
    rq.add_argument("--searcher", choices=["ddg", "searxng", "serper"], default="searxng")
    rq.add_argument("--search-type", choices=["general", "science"], default="general", help="the web, or scholarly literature")
    rq.add_argument("--searxng-url", default="http://localhost:8080")
    rq.add_argument("--llm-model", default=LLM_MODEL)
    rq.add_argument("--queries", choices=["llm", "claim"], default="llm", help="LLM-written queries or the claim text")
    rq.add_argument("--pairs", type=int, help="paired: only the first N pairs")
    args = parser.parse_args()
    use_dataset(args.dataset)

    if args.cmd == "build":
        if DATASET == "paired":
            if not (args.xlsx and args.sheet):
                raise SystemExit("--xlsx and --sheet: the workbook and its sheet of passage pairs")
            texts = build_paired(args.xlsx, args.sheet)
            TEXTS.parent.mkdir(parents=True, exist_ok=True)
        else:
            texts = build_texts(json.loads((DATA / "fact_pairs.json").read_text()))
        duplicates = sorted({t["id"] for t in texts if sum(u["id"] == t["id"] for u in texts) > 1})
        if duplicates:  # evidence is keyed by id: a collision would give one text another's atoms
            raise SystemExit(f"duplicate text ids: {duplicates[:5]}")
        TEXTS.write_text("".join(json.dumps(t, ensure_ascii=False) + "\n" for t in texts))
        print(f"{len(texts)} texts, {sum(len(t['sentences']) for t in texts)} sentences -> {TEXTS}")
    elif args.cmd == "record":
        await record(args.searcher, args.searxng_url, args.llm_model, args.search_type)
    elif args.cmd == "recrawl":
        use_evidence(args.evidence)
        await recrawl_open_access()
    elif args.cmd == "fetch-papers":
        await fetch_papers(args.xlsx, args.links_sheet)
    elif args.cmd == "requery":
        await requery(args.tag, args.searcher, args.searxng_url, args.llm_model, args.queries, args.pairs, args.search_type)
    elif args.cmd == "run":
        use_evidence(args.evidence)
        RESULTS.mkdir(parents=True, exist_ok=True)
        for variant in VARIANTS if args.variant == "all" else [args.variant]:
            res = await run(variant, args.live, args.timeout, args.searcher, args.searxng_url, args.llm_model, args.limit, args.strong, args.in_domain, args.search_type)
            name = f"eval-{variant}{'-live' if args.live else ''}{'-' + args.evidence if args.evidence else ''}"
            name += "-in-domain" if args.in_domain else ""
            name += f"-strong{args.strong:g}" if args.strong != 0.7 else ""
            (RESULTS / f"{name}.json").write_text(json.dumps(res, indent=1))
            text = paired_report(res) if DATASET == "paired" else report(res)
            (RESULTS / f"{name}.md").write_text(text)
            print("\n" + text)
    if DATASET == "paired" and args.cmd == "report":
        for path in sorted(RESULTS.glob("eval-*.json")):
            text = paired_report(json.loads(path.read_text()))
            path.with_suffix(".md").write_text(text)
            print(text)
    elif DATASET != "paired" and args.cmd in ("run", "report"):
        (RESULTS / "eval-comparison.md").write_text(comparison())
        plot()
        print(f"-> {RESULTS / 'eval-comparison.md'}, eval-comparison.png, eval-verdicts.png")


if __name__ == "__main__":
    asyncio.run(main())
