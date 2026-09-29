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
    words broken across lines (pypdf: 41 per 1,000 words, which hurts passage matching). Raises on a broken PDF."""
    import pypdfium2 as pdfium

    with _PDFIUM_LOCK:
        doc = pdfium.PdfDocument(data)
        try:
            return "\n".join(doc[i].get_textpage().get_text_range() for i in range(len(doc)))
        finally:
            doc.close()


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
