"""Chunking strategies.

Every chunk records char_start/char_end into documents.text. That is the single
non-negotiable invariant of this module: lose it and span-level citation
highlighting becomes impossible without re-embedding the whole corpus.

Four strategies ship so the ablation has something to compare:

  fixed       fixed-size windows with overlap. The dumb baseline; hard to beat.
  section     one chunk per CFR section. Maximum context, worst precision.
  structured  pack whole paragraphs up to a target size. Usually the winner
              for regulatory text, which is already written in numbered units.
  contextual  structured, plus a context prefix prepended before embedding so
              an orphaned paragraph ("...within 24 hours") still embeds near
              its subject.
"""

from __future__ import annotations

import json
import sqlite3
from typing import Dict, List, Optional, Tuple

from . import config, db

STRATEGIES = ("fixed", "section", "structured", "contextual")


def _paragraphs(text: str) -> List[Tuple[int, int]]:
    """Offsets of each paragraph in `text`, splitting on blank lines."""
    spans: List[Tuple[int, int]] = []
    start = 0
    n = len(text)
    while start < n:
        while start < n and text[start] in "\n \t":
            start += 1
        if start >= n:
            break
        end = text.find("\n\n", start)
        if end == -1:
            end = n
        stripped_end = end
        while stripped_end > start and text[stripped_end - 1] in "\n \t":
            stripped_end -= 1
        if stripped_end > start:
            spans.append((start, stripped_end))
        start = end + 2
    return spans


def chunk_fixed(text: str) -> List[Tuple[int, int]]:
    """Character windows with overlap, snapped to whitespace boundaries."""
    size = config.CHUNK_TARGET_CHARS
    overlap = config.CHUNK_OVERLAP_CHARS
    spans: List[Tuple[int, int]] = []
    n = len(text)
    start = 0
    while start < n:
        end = min(start + size, n)
        if end < n:
            # Back off to the last whitespace so we do not split a word.
            cut = text.rfind(" ", start + size // 2, end)
            if cut > start:
                end = cut
        if end - start >= config.CHUNK_MIN_CHARS or not spans:
            spans.append((start, end))
        if end >= n:
            break
        start = max(end - overlap, start + 1)
    return spans


def chunk_section(text: str) -> List[Tuple[int, int]]:
    return [(0, len(text))]


def chunk_structured(text: str) -> List[Tuple[int, int]]:
    """Greedily pack whole paragraphs up to the target size.

    A paragraph longer than the target on its own is handed to the fixed-window
    splitter rather than emitted oversized.
    """
    target = config.CHUNK_TARGET_CHARS
    spans: List[Tuple[int, int]] = []
    cur_start: Optional[int] = None
    cur_end = 0

    for p_start, p_end in _paragraphs(text):
        p_len = p_end - p_start
        if p_len > target:
            if cur_start is not None:
                spans.append((cur_start, cur_end))
                cur_start = None
            for s, e in chunk_fixed(text[p_start:p_end]):
                spans.append((p_start + s, p_start + e))
            continue
        if cur_start is None:
            cur_start, cur_end = p_start, p_end
        elif p_end - cur_start <= target:
            cur_end = p_end
        else:
            spans.append((cur_start, cur_end))
            cur_start, cur_end = p_start, p_end

    if cur_start is not None:
        spans.append((cur_start, cur_end))
    return spans or [(0, len(text))]


_SPLITTERS = {
    "fixed": chunk_fixed,
    "section": chunk_section,
    "structured": chunk_structured,
    "contextual": chunk_structured,
}


def _context_for(doc: sqlite3.Row, heading_path: str) -> str:
    """The prefix prepended to a chunk before embedding.

    Cheap and deterministic: the section's place in the hierarchy. An LLM-written
    one-line summary of the parent section usually does better and costs one
    call per chunk at index time - wire it in here and measure the difference
    rather than assuming.
    """
    parts = [p for p in heading_path.split(" > ") if p][:-1]
    return "{}. {}".format(" ".join(parts), doc["heading"]).strip()


def build(
    conn: sqlite3.Connection,
    strategy: str = "structured",
    rebuild: bool = True,
) -> Dict[str, int]:
    if strategy not in STRATEGIES:
        raise ValueError("unknown strategy {!r}; expected one of {}".format(strategy, STRATEGIES))

    heading_paths: Dict[str, str] = {}
    hp_file = config.DATA_DIR / "heading_paths.json"
    if hp_file.exists():
        heading_paths = json.loads(hp_file.read_text(encoding="utf-8"))

    if rebuild:
        db.clear_strategy(conn, strategy)

    split = _SPLITTERS[strategy]
    n_chunks = 0
    rows: List[tuple] = []
    fts_rows: List[tuple] = []

    for doc in db.iter_documents(conn):
        text = doc["text"]
        path = heading_paths.get(doc["doc_id"], doc["heading"])
        context = _context_for(doc, path) if strategy == "contextual" else ""

        for i, (start, end) in enumerate(split(text)):
            body = text[start:end].strip()
            if len(body) < config.CHUNK_MIN_CHARS and i > 0:
                continue
            chunk_id = "{}#{}:{}".format(doc["doc_id"], strategy, i)
            rows.append((chunk_id, doc["doc_id"], strategy, i, start, end, path, context, body))
            # The lexical index sees the context prefix too, so both retrievers
            # are looking at the same text for a given strategy.
            fts_rows.append((chunk_id, strategy, path, (context + "\n" + body) if context else body))
            n_chunks += 1

    conn.executemany(
        """INSERT OR REPLACE INTO chunks
           (chunk_id, doc_id, strategy, ordinal, char_start, char_end, heading_path, context, text)
           VALUES (?,?,?,?,?,?,?,?,?)""",
        rows,
    )
    conn.executemany(
        "INSERT INTO chunks_fts (chunk_id, strategy, heading_path, text) VALUES (?,?,?,?)",
        fts_rows,
    )
    conn.commit()
    return {"strategy": strategy, "chunks": n_chunks}


def verify_offsets(conn: sqlite3.Connection, strategy: str, limit: int = 0) -> Tuple[int, int]:
    """Re-slice every chunk out of its document and confirm the text matches.

    Run this after any change to the chunker. A silent offset drift produces
    citations that highlight the wrong sentence, which is worse than no
    highlighting at all.
    """
    sql = (
        "SELECT c.chunk_id, c.char_start, c.char_end, c.text, d.text AS doc_text "
        "FROM chunks c JOIN documents d USING (doc_id) WHERE c.strategy = ?"
    )
    if limit:
        sql += " LIMIT {}".format(int(limit))
    checked = bad = 0
    for row in conn.execute(sql, (strategy,)):
        checked += 1
        if row["doc_text"][row["char_start"]:row["char_end"]].strip() != row["text"]:
            bad += 1
    return checked, bad
