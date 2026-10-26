def test_clean_text_strips_markdown_and_citations_but_keeps_content():
    from factassessor.passages import clean_text

    md = """## The most visited monument
**As France's symbol**, it welcomes _almost_ 7 million visitors.[6][7] She named it _polonium_.[c]
| Height | 330 m |
|---|---|
| Opened | 1889 |
Visitors a year
7 000 000
  * Facebook
  * Twitter
  * Paris is the capital and largest city of France
---
"""
    assert clean_text(md) == (
        "The most visited monument\n"
        "As France's symbol, it welcomes almost 7 million visitors. She named it polonium.\n"
        "Height 330 m\n"
        "Opened 1889\n"
        "Visitors a year\n"
        "7 000 000\n"
        "Paris is the capital and largest city of France"
    )


def test_clean_text_drops_leaked_html_and_script_data():
    from factassessor.passages import clean_text

    md = """Real sentence about the earthquake.
direct-relief-support-continues/"><script type="application/ld+json"> { "@context": "https://schema.org", "@type": "Article" }</script>
<div class="x">Inline tag text survives</div>
{"@context": "https://schema.org", "@graph": [{"@type": "WebPage"}]}
Another real sentence."""
    assert clean_text(md) == "Real sentence about the earthquake.\nInline tag text survives\nAnother real sentence."


def test_normalize_text_makes_unit_notations_comparable():
    from factassessor.passages import normalize_text

    # the same value, written two ways: a claim's superscripts vs a paper's minus sign and digit
    assert normalize_text("247 Mg ha⁻¹") == normalize_text("247 Mg ha−1") == "247 Mg ha-1"
    assert normalize_text("6.9 Mg ha⁻¹ yr⁻¹") == "6.9 Mg ha-1 yr-1"
    assert normalize_text("CO₂ at 10 µm") == "CO2 at 10 μm"
    assert normalize_text("ﬁeld") == "field"
    assert normalize_text("an ordinary sentence, 1903.") == "an ordinary sentence, 1903."
