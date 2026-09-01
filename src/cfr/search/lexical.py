"""BM25 over SQLite FTS5.

This is the half of hybrid search that handles what embeddings are bad at:
exact section numbers, defined terms, acronyms, chemical names - anything rare
enough that a fuzzy sense of meaning loses it.
"""

from __future__ import annotations

import re
import sqlite3
from typing import List, Tuple

# Section numbers ("262.17", "1910.1200") must survive as single tokens.
_TOKEN = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?|\d+(?:\.\d+)+|\d+")

# Words that match nearly every regulation and only add noise to an OR query.
_STOP = frozenset(
    """a an the and or of to in for on at by is are was were be been being as
    that this these those it its from with which what when where who whom how
    do does did can could shall should may might must will would i you we they
    my your our their there here if then than so such not no nor but""".split()
)


def sanitize(query: str) -> str:
    """Turn arbitrary user text into a safe FTS5 MATCH expression.

    Everything is quoted, so FTS5 operators typed by a user ("NEAR", "*", "-")
    are treated as literal search terms rather than syntax. An unquoted query
    would let a stray quote character raise sqlite3.OperationalError on every
    request.
    """
    tokens = _TOKEN.findall(query.lower())
    kept = [t for t in tokens if t not in _STOP and len(t) > 1]
    if not kept:
        kept = [t for t in tokens if len(t) > 1]
    if not kept:
        return ""
    return " OR ".join('"{}"'.format(t.replace('"', '""')) for t in kept)


def search(
    conn: sqlite3.Connection,
    query: str,
    strategy: str,
    limit: int = 100,
) -> List[Tuple[str, float]]:
    """Return (chunk_id, score) with higher scores better."""
    match = sanitize(query)
    if not match:
        return []
    try:
        rows = conn.execute(
            """SELECT chunk_id, bm25(chunks_fts, 2.0, 1.0) AS score
               FROM chunks_fts
               WHERE chunks_fts MATCH ? AND strategy = ?
               ORDER BY score
               LIMIT ?""",
            (match, strategy, limit),
        ).fetchall()
    except sqlite3.OperationalError:
        # A malformed MATCH should degrade to "no lexical hits", never a 500.
        return []
    # FTS5 bm25() is negated (more negative = better). Flip it so every
    # retriever in this package agrees that bigger means better.
    return [(r["chunk_id"], -float(r["score"])) for r in rows]
