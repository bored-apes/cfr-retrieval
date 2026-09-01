"""FTS5 query sanitisation.

Unsanitised user text reaches sqlite as MATCH syntax, so a stray quote or the
word "NEAR" raises OperationalError on a perfectly ordinary question.
"""

import pytest

from cfr import db
from cfr.search import lexical


@pytest.fixture()
def conn(tmp_path):
    c = db.connect(tmp_path / "t.db")
    db.init(c)
    rows = [
        ("c1", "structured", "Title 40 > 262.17", "A large quantity generator accumulates hazardous waste on site for no more than 90 days."),
        ("c2", "structured", "Title 40 > 262.16", "A small quantity generator may accumulate hazardous waste for no more than 180 days."),
        ("c3", "structured", "Title 29 > 1910.132", "The employer shall provide protective equipment at no cost to employees."),
    ]
    c.executemany(
        "INSERT INTO chunks_fts (chunk_id, strategy, heading_path, text) VALUES (?,?,?,?)", rows
    )
    c.commit()
    return c


@pytest.mark.parametrize("q", [
    'what is a "large quantity generator"?',
    "NEAR AND OR NOT",
    "262.17*",
    "-hazardous",
    "waste^2",
    "(unbalanced",
    "'; DROP TABLE chunks; --",
    "  ",
    "???",
    "\\",
])
def test_arbitrary_input_never_raises(conn, q):
    """Every one of these is valid FTS5 syntax that would otherwise explode."""
    lexical.search(conn, q, "structured", limit=5)


def test_finds_exact_section_number(conn):
    """The thing dense retrieval is worst at."""
    hits = lexical.search(conn, "262.17", "structured", limit=5)
    assert hits and hits[0][0] == "c1"


def test_scores_are_higher_is_better(conn):
    hits = lexical.search(conn, "hazardous waste generator", "structured", limit=5)
    assert len(hits) >= 2
    assert hits[0][1] >= hits[-1][1]


def test_strategy_isolation(conn):
    assert lexical.search(conn, "hazardous", "contextual", limit=5) == []


def test_stopwords_do_not_swallow_the_query(conn):
    hits = lexical.search(conn, "How long can the generator keep it on site?", "structured", limit=5)
    assert hits


def test_sanitize_quotes_every_token():
    out = lexical.sanitize('hazardous "waste" 262.17')
    assert '"hazardous"' in out and '"262.17"' in out
    assert " OR " in out
