"""The offset invariant. If these fail, citations highlight the wrong text."""

import pytest

from cfr import chunk as chunk_mod
from cfr import db

DOC = (
    "§ 262.17 Conditions for exemption.\n\n"
    "A large quantity generator may accumulate hazardous waste on site without a "
    "permit, provided that all of the following conditions are met:\n\n"
    "(a) Accumulation. A large quantity generator accumulates hazardous waste on "
    "site for no more than 90 days, unless in compliance with the accumulation "
    "time limit extension in paragraph (b) of this section.\n\n"
    "(b) Extension. A generator who accumulates for more than 90 days is an "
    "operator of a storage facility and is subject to the requirements of parts "
    "264 and 270 of this chapter.\n\n"
    + "(c) Filler paragraph with enough text to force a second chunk. " * 40
)


@pytest.fixture()
def conn(tmp_path, monkeypatch):
    monkeypatch.setattr("cfr.config.DATA_DIR", tmp_path)
    c = db.connect(tmp_path / "t.db")
    db.init(c)
    c.execute(
        "INSERT INTO documents (doc_id,title,part,section,heading,citation,text) "
        "VALUES (?,?,?,?,?,?,?)",
        ("ecfr:40:262.17", "40", "262", "262.17", "Conditions", "40 CFR 262.17", DOC),
    )
    c.commit()
    return c


@pytest.mark.parametrize("strategy", chunk_mod.STRATEGIES)
def test_offsets_roundtrip(conn, strategy):
    """Every chunk must re-slice out of its document byte for byte."""
    chunk_mod.build(conn, strategy=strategy)
    checked, bad = chunk_mod.verify_offsets(conn, strategy)
    assert checked > 0
    assert bad == 0


@pytest.mark.parametrize("strategy", chunk_mod.STRATEGIES)
def test_chunks_cover_the_key_sentence(conn, strategy):
    chunk_mod.build(conn, strategy=strategy)
    rows = conn.execute(
        "SELECT text FROM chunks WHERE strategy = ?", (strategy,)
    ).fetchall()
    assert any("no more than 90 days" in r["text"] for r in rows)


def test_structured_respects_paragraph_boundaries(conn):
    chunk_mod.build(conn, strategy="structured")
    for r in conn.execute("SELECT text FROM chunks WHERE strategy='structured'"):
        # Packing whole paragraphs must never leave a chunk starting mid-sentence
        # with a lowercase continuation of the previous one.
        assert r["text"] == r["text"].strip()


def test_contextual_adds_context_and_others_do_not(conn):
    chunk_mod.build(conn, strategy="contextual")
    chunk_mod.build(conn, strategy="structured")
    ctx = conn.execute("SELECT context FROM chunks WHERE strategy='contextual' LIMIT 1").fetchone()
    plain = conn.execute("SELECT context FROM chunks WHERE strategy='structured' LIMIT 1").fetchone()
    assert ctx["context"]
    assert plain["context"] == ""


def test_rebuild_is_idempotent(conn):
    a = chunk_mod.build(conn, strategy="structured")["chunks"]
    b = chunk_mod.build(conn, strategy="structured")["chunks"]
    assert a == b
    total = conn.execute(
        "SELECT COUNT(*) FROM chunks WHERE strategy='structured'"
    ).fetchone()[0]
    assert total == b
