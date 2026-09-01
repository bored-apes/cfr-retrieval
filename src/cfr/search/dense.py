"""Dense retrieval: cosine similarity over the chunk vectors.

At this corpus size an exhaustive numpy dot product is both simpler and faster
than an approximate index - 200k x 384 floats is ~300MB and a full scan takes
single-digit milliseconds. Swap in HNSW only when the matrix stops fitting in
memory; doing it earlier trades exact results for nothing.
"""

from __future__ import annotations

import sqlite3
import threading
from typing import Dict, List, Optional, Tuple

import numpy as np

from .. import embed

_CACHE: Dict[str, Tuple[List[str], np.ndarray]] = {}
_LOCK = threading.Lock()


def load(conn: sqlite3.Connection, strategy: str) -> Tuple[List[str], np.ndarray]:
    """Load and L2-normalise one strategy's vectors, memoised per process."""
    with _LOCK:
        if strategy in _CACHE:
            return _CACHE[strategy]

        rows = conn.execute(
            "SELECT chunk_id, dim, vec FROM vectors WHERE strategy = ? ORDER BY chunk_id",
            (strategy,),
        ).fetchall()
        if not rows:
            # Deliberately not cached. A server started while the index is still
            # building would otherwise pin an empty matrix for its whole
            # lifetime and silently return no dense results forever.
            return ([], np.zeros((0, 0), dtype=np.float32))

        dim = int(rows[0]["dim"])
        ids = [r["chunk_id"] for r in rows]
        mat = np.frombuffer(b"".join(r["vec"] for r in rows), dtype=np.float32)
        mat = mat.reshape(len(rows), dim)
        # Normalise once at load so query time is a plain dot product.
        _CACHE[strategy] = (ids, embed.normalize(mat.copy()))
        return _CACHE[strategy]


def clear_cache() -> None:
    with _LOCK:
        _CACHE.clear()


def search(
    conn: sqlite3.Connection,
    query: str,
    strategy: str,
    limit: int = 100,
    qvec: Optional[np.ndarray] = None,
) -> List[Tuple[str, float]]:
    ids, mat = load(conn, strategy)
    if not ids:
        return []
    if qvec is None:
        qvec = embed.embed_query(query)
    q = embed.normalize(np.asarray(qvec, dtype=np.float32).reshape(1, -1))[0]

    sims = mat @ q
    limit = min(limit, len(ids))
    top = np.argpartition(-sims, limit - 1)[:limit]
    top = top[np.argsort(-sims[top])]
    return [(ids[int(i)], float(sims[int(i)])) for i in top]
