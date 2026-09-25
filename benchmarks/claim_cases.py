"""Is this a factual claim worth checking? Statement / expected (1 = claim, 0 = not) cases for comparing claim
filters (see compare_claim_filters.py)."""

CASES: list[tuple[str, int]] = [
    ('Sanjog is a computer scientist at UAH.', 1),
    ('Sanjog works in the AKD project.', 1),
    ('Marie Curie won the Nobel Prize in Physics in 1903.', 1),
    ('Paradox lives on Mars.', 1),
    ('The Eiffel Tower is 330 meters tall.', 1),
    ('Apple is based in Cupertino.', 1),
    ('NASA was founded in 1958.', 1),
    ('The company reported revenue of $4 billion last quarter.', 1),
    ('Python was created by Guido van Rossum.', 1),
    ('The drug reduced mortality by 20 percent in trials.', 1),
    ('Hi, i am paradox.', 0),
    ('I think pizza is the best food.', 0),
    ('Can you help me?', 0),
    ('Thanks so much!', 0),
    ('Let us look at the next slide.', 0),
    ('This movie is amazing.', 0),
    ('We should invest more in education.', 0),
    ('Please send me the report.', 0),
    ('What a beautiful day.', 0),
]
