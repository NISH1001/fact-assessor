"""Document -> text: PDF or HTML bytes to clean plain text, for anything that reads documents (crawlers, local
files, the eval). Format handling lives here once, so every crawler reads PDFs the same way.

- `extract(body, content_type)`: decides the format from the content type or the bytes themselves (`%PDF`, `<html`),
  returns `(title, text)` with the text cleaned (`passages.clean_text`), or None if it's unreadable or empty.
- `pdf_text(data)`: PDFium (pypdfium2, `fact-assessor[pdf]`); thread-safe.
- `html_text(html)`: BeautifulSoup + lxml, without scripts, menus, headers, footers; formulas as their TeX.

All are synchronous and CPU-bound: call them from async code with `asyncio.to_thread`.
"""

from __future__ import annotations

import threading

from factassessor.passages import clean_text

_PDFIUM_LOCK = threading.Lock()  # PDFium must never be called from two threads at once


def pdf_text(data: bytes) -> str:
    """PDF text with PDFium (pypdfium2, fact-assessor[pdf]): on 6 arXiv papers 22ms each vs pypdf's 171ms, and no
    words broken across lines (pypdf: 41 per 1,000 words, which hurts passage matching). Raises on a broken PDF.

    Runs in a worker process (`PDF_WORKERS` of them, started on first use): PDFium is C code parsing documents from
    the web, and a crash there (a segfault, seen in a live eval run) would otherwise kill the whole pipeline. A
    crashed or stuck worker loses only that PDF and is replaced."""
    text = _WORKERS.run(data)
    if text is None:
        raise RuntimeError("the PDF worker crashed or timed out on this document")
    return text


def _pdf_text_here(data: bytes) -> str:
    """PDF text in this process. Every page is closed under the lock, not left to the garbage collector."""
    import pypdfium2 as pdfium

    with _PDFIUM_LOCK:
        doc = pdfium.PdfDocument(data)
        try:
            parts = []
            for i in range(len(doc)):
                page = doc[i]
                textpage = page.get_textpage()
                try:
                    parts.append(textpage.get_text_range())
                finally:
                    textpage.close()
                    page.close()
            return "\n".join(parts)
        finally:
            doc.close()


# --- PDF worker processes ----------------------------------------------------------------------------------------
# A worker is `python -c "from factassessor.extract import _serve; _serve()"`, reading requests on stdin and answering
# on stdout (not multiprocessing: in spawn mode it re-imports the caller's main script in every worker).
# Request: 8-byte length + PDF bytes. Reply: 1-byte status (0 text, 1 error) + 8-byte length + UTF-8 text.

PDF_WORKERS = 2  # PDFium is single-threaded within a process; each worker is a process of its own
PDF_TIMEOUT = 30.0  # a worker still parsing after this is stuck: killed and replaced
_TEST_CRASH = b"__test_crash__"  # test hook: a worker given exactly these bytes aborts, as a PDFium segfault would


def _serve() -> None:
    import os
    import sys

    stdin, stdout = sys.stdin.buffer, sys.stdout.buffer
    while header := stdin.read(8):
        data = stdin.read(int.from_bytes(header, "big"))
        if data == _TEST_CRASH:
            os.abort()
        try:
            status, reply = 0, _pdf_text_here(data).encode("utf-8", errors="replace")
        except Exception as exc:
            status, reply = 1, repr(exc).encode()
        stdout.write(bytes([status]) + len(reply).to_bytes(8, "big") + reply)
        stdout.flush()


class _Worker:
    def __init__(self) -> None:
        import subprocess
        import sys

        self.proc = subprocess.Popen(
            [sys.executable, "-c", "from factassessor.extract import _serve; _serve()"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        )

    def alive(self) -> bool:
        return self.proc.poll() is None

    def run(self, data: bytes) -> tuple[int, str] | None:
        """(status, text or error), or None if the worker died or got stuck (it is then killed)."""
        watchdog = threading.Timer(PDF_TIMEOUT, self.proc.kill)
        watchdog.start()
        try:
            self.proc.stdin.write(len(data).to_bytes(8, "big") + data)  # type: ignore[union-attr]
            self.proc.stdin.flush()  # type: ignore[union-attr]
            head = self.proc.stdout.read(9)  # type: ignore[union-attr]
            if len(head) < 9:
                return None
            body = self.proc.stdout.read(int.from_bytes(head[1:], "big"))  # type: ignore[union-attr]
            return head[0], body.decode("utf-8", errors="replace")
        except (BrokenPipeError, OSError):
            return None
        finally:
            watchdog.cancel()


class _WorkerPool:
    def __init__(self) -> None:
        self._idle: list[_Worker] = []
        self._lock = threading.Lock()
        self._free = threading.Semaphore(PDF_WORKERS)
        self._all: list[_Worker] = []

    def pids(self) -> list[int]:
        return [w.proc.pid for w in self._all if w.alive()]

    def _take(self, fresh: bool = False) -> _Worker:
        with self._lock:
            while self._idle and not fresh:
                if (w := self._idle.pop()).alive():
                    return w
            w = _Worker()
            self._all = [x for x in self._all if x.alive()] + [w]
            return w

    def _give_back(self, w: _Worker) -> None:
        with self._lock:
            if w.alive():
                self._idle.append(w)

    def run(self, data: bytes) -> str | None:
        """The PDF's text; raises ValueError on a broken PDF; None if parsing it crashed the worker (twice)."""
        with self._free:
            for attempt in range(2):  # an idle worker may have died since its last job: retry once on a new one
                w = self._take(fresh=attempt > 0)
                out = w.run(data)
                if out is None:
                    w.proc.kill()  # dead or stuck: never reused
                    continue
                self._give_back(w)
                status, text = out
                if status:
                    raise ValueError(text)
                return text
            return None


_WORKERS = _WorkerPool()


_NOT_CONTENT = ["script", "style", "noscript", "template", "svg", "iframe", "nav", "header", "footer", "aside", "form"]


def html_text(html: str | bytes) -> tuple[str, str]:
    """HTML -> (title, text), without scripts, styles, menus, headers, footers, or forms. Bytes are decoded by
    BeautifulSoup (from the page's own `<meta charset>`).

    Formulas (MathML, as in arXiv's HTML papers) become their TeX (`alttext`) once: `get_text()` alone would print
    every MathML token on its own line and then the TeX annotation again."""
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "lxml")
    title = soup.title.get_text(strip=True) if soup.title else ""
    for tag in soup(_NOT_CONTENT):
        tag.decompose()
    for math in soup("math"):
        tex = math.get("alttext") or (math.find("annotation") or math).get_text(" ", strip=True)
        math.replace_with(f" {tex} ")
    return title, soup.get_text("\n")


def _is_pdf(body: bytes, content_type: str) -> bool:
    return "pdf" in content_type or body.lstrip()[:5] == b"%PDF-"


def _is_html(body: bytes, content_type: str) -> bool:
    if "html" in content_type or "xml" in content_type:
        return True
    head = body.lstrip()[:100].lower()
    return head.startswith((b"<!doctype html", b"<html"))


def extract(body: bytes, content_type: str = "", encoding: str | None = None) -> tuple[str, str] | None:
    """PDF or HTML bytes -> (title, clean text), or None: another format, a broken file, or no text at all.
    The format comes from `content_type` or, when that's missing or generic (`application/octet-stream`), from
    the bytes. `encoding` (from the HTTP response) decodes HTML; without it, the page's own charset is used.
    PDFs have no title here: the caller knows it better (search hit, metadata)."""
    content_type = content_type.lower()
    try:
        if _is_pdf(body, content_type):
            title, text = "", pdf_text(body)
        elif _is_html(body, content_type):
            title, text = html_text(body.decode(encoding, errors="replace") if encoding else body)
        else:
            return None
    except Exception:  # broken PDFs, undecodable bytes
        return None
    text = clean_text(text)
    return (title, text) if text else None
