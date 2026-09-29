from factassessor.extract import extract, html_text, pdf_text
from tests.pdfs import minimal_pdf

PAGE = b"""<html><head><title>Nepal earthquake</title><script>var x = 1;</script></head>
<body><nav>Home | About</nav><p>The earthquake struck on <b>25 April 2015</b> with a magnitude of 7.8.</p>
<footer>Privacy policy</footer></body></html>"""


def test_pdf_text_reads_a_real_pdf():
    text = pdf_text(minimal_pdf("Secondary forests recover biomass quickly.", "They gained 122 Mg per ha."))
    assert "Secondary forests recover biomass quickly." in text and "122 Mg per ha" in text


def test_html_text_keeps_title_and_content_and_drops_chrome():
    title, text = html_text(PAGE.decode())
    assert title == "Nepal earthquake" and "magnitude of 7.8" in text
    for junk in ("var x", "Home | About", "Privacy policy"):
        assert junk not in text


def test_html_math_becomes_its_tex_once_not_mathml_token_soup():
    # arXiv's HTML (LaTeXML): each formula is MathML tokens plus a TeX annotation; get_text() would print both
    page = """<html><body><p>Biomass grew by
    <math alttext="122\\,\\mathrm{Mg\\,ha^{-1}}"><semantics><mrow><mn>122</mn><mi>Mg</mi><msup><mi>ha</mi><mrow>
    <mo>-</mo><mn>1</mn></mrow></msup></mrow><annotation encoding="application/x-tex">122\\,\\mathrm{Mg\\,ha^{-1}}</annotation>
    </semantics></math> in 20 years.</p></body></html>"""
    _, text = html_text(page)
    assert text.count("122") == 1 and "ha^{-1}" in text and "in 20 years" in text


def test_extract_decides_by_content_type_or_the_bytes_themselves():
    pdf = minimal_pdf("A paper about lunar pits.")
    assert extract(pdf, "application/pdf") == ("", "A paper about lunar pits.")
    assert extract(pdf, "application/octet-stream")[1] == "A paper about lunar pits."  # %PDF magic, mislabelled
    assert extract(pdf, "")[1] == "A paper about lunar pits."
    title, text = extract(PAGE, "text/html; charset=utf-8")
    assert title == "Nepal earthquake" and "magnitude of 7.8" in text
    assert extract(b"<!DOCTYPE html>" + PAGE, "")[0] == "Nepal earthquake"  # no type given: sniffed


def test_extract_text_is_clean():
    _, text = extract(b"<html><body><p>Line one.</p>\n\n\n<p>  Line two.  </p></body></html>", "text/html")
    assert text == "Line one.\nLine two."


def test_extract_uses_the_given_encoding():
    body = "<html><body><p>Café São Paulo</p></body></html>".encode("latin-1")
    assert extract(body, "text/html", encoding="latin-1")[1] == "Café São Paulo"


def test_unreadable_or_empty_documents_are_none():
    assert extract(b"\x89PNG\r\n\x1a\n...", "image/png") is None
    assert extract(b'{"a": 1}', "application/json") is None
    assert extract(b"%PDF-1.4 truncated garbage", "application/pdf") is None  # a broken PDF: None, not an error
    assert extract(b"<html><body><script>app()</script></body></html>", "text/html") is None  # nothing to read


def test_pdf_text_is_safe_from_many_threads_at_once():
    # PDFium is not thread-safe: pages must be closed under the lock, not left to the garbage collector, which can
    # run in another thread while one is inside PDFium (a live eval run aborted the process that way)
    import gc
    from concurrent.futures import ThreadPoolExecutor

    pdf = minimal_pdf(*[f"Line {i} of a many-line paper about forests." for i in range(40)])
    gc.set_threshold(1)  # collect constantly, in whichever thread allocates
    try:
        with ThreadPoolExecutor(16) as pool:
            texts = list(pool.map(pdf_text, [pdf] * 400))
    finally:
        gc.set_threshold(700, 10, 10)
    assert all("Line 39 of a many-line paper" in t for t in texts)


# --- PDFs are parsed in worker processes: a PDFium crash (a real segfault in a live run) loses one PDF, not the run --

def test_pdfs_are_parsed_in_a_worker_process_not_this_one():
    import os

    from factassessor import extract as ex

    assert "forests" in pdf_text(minimal_pdf("Secondary forests recover."))
    assert ex._WORKERS.pids() and os.getpid() not in ex._WORKERS.pids()


def test_a_dead_worker_is_replaced_and_the_pdf_still_read():
    import os
    import signal

    from factassessor import extract as ex

    pdf_text(minimal_pdf("warm up"))
    for pid in ex._WORKERS.pids():
        os.kill(pid, signal.SIGKILL)  # what a PDFium segfault does to the worker
    assert "after the crash" in pdf_text(minimal_pdf("Read after the crash."))


def test_a_pdf_that_kills_its_worker_is_none_from_extract():
    from factassessor import extract as ex

    assert ex._WORKERS.run(b"__test_crash__") is None  # the worker aborts on this input (test hook)
    assert "still works" in pdf_text(minimal_pdf("It still works."))


def test_worker_results_match_in_process_extraction():
    from factassessor import extract as ex

    pdf = minimal_pdf(*[f"Line {i} about biomass." for i in range(30)])
    assert pdf_text(pdf) == ex._pdf_text_here(pdf)
