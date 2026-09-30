from factassessor.utils import locate, sentences

TEXT = "It was believed that Nepal's earthquake in 2017 of 7.8 magnitude scale caused massive damage. Total lives lost were 1 million people."


def test_sentences_end_at_punctuation_and_keep_decimals_whole():
    assert [TEXT[s:e] for s, e in sentences(TEXT)] == [
        "It was believed that Nepal's earthquake in 2017 of 7.8 magnitude scale caused massive damage.",
        "Total lives lost were 1 million people.",
    ]
    assert sentences("") == []


def test_locate_picks_the_sentence_the_claim_was_made_from():
    span = lambda claim: TEXT[slice(*locate(claim, TEXT))]
    assert span("The Nepal earthquake had a magnitude of 7.8.").startswith("It was believed")
    assert span("The Nepal earthquake killed 1 million people.") == "Total lives lost were 1 million people."
    assert span("Something about nothing here.").startswith("It was believed")  # no match: the first sentence


def test_locate_weighs_down_context_the_atomizer_repeats_in_every_claim():
    # every claim of a scientific text names the study; the claim's own detail must decide the sentence
    text = ("A 2025 study in Connecticut modelled aboveground biomass with random forests. "
            "The 2025 Connecticut study used 67 explanatory variables from LiDAR and Sentinel-2. "
            "The 2025 Connecticut study validated its models on 142 FIA subplots.")
    claim = "The 2025 study in Connecticut used 67 explanatory variables."
    assert text[slice(*locate(claim, text))].startswith("The 2025 Connecticut study used 67")


def test_locate_on_text_without_sentence_punctuation_is_the_whole_text():
    assert locate("anything", "no punctuation at all") == (0, len("no punctuation at all"))
