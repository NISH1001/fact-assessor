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
