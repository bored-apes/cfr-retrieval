"""SQLite storage.

One file holds the whole system: source documents, chunks, the BM25 index
(FTS5), and the dense vectors. That keeps the project deployable as a single
read-only artifact and free to host.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Iterable, Optional

from . import config

SCHEMA = """
PRAGMA journal_mode = WAL;

-- Source documents, one row per CFR section. `text` is the canonical string
-- that every chunk offset refers to. Never rewrite it in place: doing so
-- silently invalidates every stored offset.
CREATE TABLE IF NOT EXISTS documents (
    doc_id      TEXT PRIMARY KEY,
    title       TEXT NOT NULL,
    part        TEXT NOT NULL,
    section     TEXT NOT NULL,
    heading     TEXT NOT NULL,
    citation    TEXT NOT NULL,
    text        TEXT NOT NULL,
    source_url  TEXT,
    fetched_at  TEXT
);

-- Chunks carry char_start/char_end into documents.text. This is what makes
-- span-level citation highlighting possible later.
CREATE TABLE IF NOT EXISTS chunks (
    chunk_id     TEXT PRIMARY KEY,
    doc_id       TEXT NOT NULL REFERENCES documents(doc_id),
    strategy     TEXT NOT NULL,
    ordinal      INTEGER NOT NULL,
    char_start   INTEGER NOT NULL,
    char_end     INTEGER NOT NULL,
    heading_path TEXT NOT NULL,
    context      TEXT NOT NULL DEFAULT '',
    text         TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_chunks_doc      ON chunks(doc_id);
CREATE INDEX IF NOT EXISTS idx_chunks_strategy ON chunks(strategy);

-- Lexical index. Standalone (not external-content) so that rebuilding one
-- strategy never corrupts another.
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    chunk_id UNINDEXED,
    strategy UNINDEXED,
    heading_path,
    text,
    tokenize = 'porter unicode61'
);

CREATE TABLE IF NOT EXISTS vectors (
    chunk_id TEXT PRIMARY KEY REFERENCES chunks(chunk_id),
    strategy TEXT NOT NULL,
    model    TEXT NOT NULL,
    dim      INTEGER NOT NULL,
    vec      BLOB NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_vectors_strategy ON vectors(strategy);

-- Answer cache. `qvec` lets us serve semantically equivalent repeat questions
-- without another generation call.
CREATE TABLE IF NOT EXISTS answer_cache (
    query_hash TEXT PRIMARY KEY,
    query      TEXT NOT NULL,
    qvec       BLOB,
    payload    TEXT NOT NULL,
    created_at TEXT NOT NULL
);

-- One row per UTC day, so the generation budget survives restarts.
CREATE TABLE IF NOT EXISTS budget (
    day   TEXT PRIMARY KEY,
    calls INTEGER NOT NULL DEFAULT 0
);

-- Relevance judgements: the hand-built answer key.
-- `judged_by` matters: a machine-generated answer key measures the machine
-- unless you can separate it from human judgement and audit it.
CREATE TABLE IF NOT EXISTS qrels (
    query_id  TEXT NOT NULL,
    doc_id    TEXT NOT NULL,
    grade     INTEGER NOT NULL,
    judged_at TEXT,
    judged_by TEXT NOT NULL DEFAULT 'human',
    note      TEXT,
    PRIMARY KEY (query_id, doc_id)
);
"""


def connect(path: Optional[Path] = None, readonly: bool = False) -> sqlite3.Connection:
    path = Path(path or config.DB_PATH)
    if readonly:
        conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True, check_same_thread=False)
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA)
    conn.commit()


def clear_strategy(conn: sqlite3.Connection, strategy: str) -> None:
    """Drop one chunking strategy's chunks, FTS rows and vectors."""
    conn.execute("DELETE FROM vectors WHERE strategy = ?", (strategy,))
    conn.execute("DELETE FROM chunks_fts WHERE strategy = ?", (strategy,))
    conn.execute("DELETE FROM chunks WHERE strategy = ?", (strategy,))
    conn.commit()


def stats(conn: sqlite3.Connection) -> dict:
    def scalar(sql: str, *args) -> int:
        row = conn.execute(sql, args).fetchone()
        return int(row[0]) if row and row[0] is not None else 0

    strategies = [
        dict(r) for r in conn.execute(
            "SELECT strategy, COUNT(*) AS chunks FROM chunks GROUP BY strategy ORDER BY strategy"
        )
    ]
    for s in strategies:
        s["vectors"] = scalar("SELECT COUNT(*) FROM vectors WHERE strategy = ?", s["strategy"])
    return {
        "documents": scalar("SELECT COUNT(*) FROM documents"),
        "chars": scalar("SELECT COALESCE(SUM(LENGTH(text)), 0) FROM documents"),
        "strategies": strategies,
        "qrels": scalar("SELECT COUNT(*) FROM qrels"),
        "judged_queries": scalar("SELECT COUNT(DISTINCT query_id) FROM qrels"),
    }


def iter_documents(conn: sqlite3.Connection) -> Iterable[sqlite3.Row]:
    yield from conn.execute("SELECT * FROM documents ORDER BY doc_id")
