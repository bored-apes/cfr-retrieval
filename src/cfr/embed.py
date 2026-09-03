"""Embedding: text in, coordinates out.

fastembed runs ONNX models on CPU, so there is no torch dependency and the same
model files can later be shipped to the browser via transformers.js. bge-small
is 384-dimensional, which keeps 200k chunks inside a free vector-store tier.
"""

from __future__ import annotations

import sqlite3
from typing import Sequence

import numpy as np

from . import config

_MODEL = None


def _model():
    """Load lazily. Importing this module must not trigger a model download."""
    global _MODEL
    if _MODEL is None:
        from fastembed import TextEmbedding

        _MODEL = TextEmbedding(model_name=config.EMBED_MODEL)
    return _MODEL


def embed_documents(texts: Sequence[str], batch_size: int = 64) -> np.ndarray:
    vecs = list(_model().embed(list(texts), batch_size=batch_size))
    return np.asarray(vecs, dtype=np.float32)


def embed_query(text: str) -> np.ndarray:
    """Embed a user query.

    Verified empirically: for BAAI/bge-small-en-v1.5, fastembed's query_embed
    applies NO instruction prefix - it is byte-identical to embed() on the same
    string (cosine 1.000000). BGE's model card suggests prefixing short queries
    with "Represent this sentence for searching relevant passages: ", and doing
    so moves the vector materially (cosine 0.98 against the unprefixed one).

    The index was built without a prefix, so queries must be too. Anything that
    re-implements this - the browser port in static/ - has to match, or every
    dense result silently degrades.
    """
    vec = next(iter(_model().query_embed([text])))
    return np.asarray(vec, dtype=np.float32)


def normalize(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=-1, keepdims=True)
    np.maximum(norms, 1e-12, out=norms)
    return mat / norms


def build(
    conn: sqlite3.Connection,
    strategy: str = "structured",
    batch_size: int = 64,
    progress: bool = True,
    resume: bool = True,
) -> dict:
    """Embed every chunk of one strategy and store the vectors.

    Commits per batch and skips chunks that already have a vector for this
    model. Embedding a full corpus is a tens-of-minutes CPU job; committing only
    at the end means an interrupt at minute 19 throws away all of it.
    """
    rows = conn.execute(
        "SELECT chunk_id, context, text FROM chunks WHERE strategy = ? ORDER BY chunk_id",
        (strategy,),
    ).fetchall()
    if not rows:
        return {"strategy": strategy, "vectors": 0, "skipped": 0}

    if resume:
        done = {
            r["chunk_id"]
            for r in conn.execute(
                "SELECT chunk_id FROM vectors WHERE strategy = ? AND model = ?",
                (strategy, config.EMBED_MODEL),
            )
        }
    else:
        conn.execute("DELETE FROM vectors WHERE strategy = ?", (strategy,))
        conn.commit()
        done = set()

    pending = [r for r in rows if r["chunk_id"] not in done]
    skipped = len(rows) - len(pending)
    if not pending:
        return {"strategy": strategy, "vectors": 0, "skipped": skipped, "total": len(rows)}

    total = len(pending)
    written = 0
    dim = config.EMBED_DIM

    for start in range(0, total, batch_size):
        batch = pending[start : start + batch_size]
        # The context prefix is part of what gets embedded - that is the entire
        # point of the contextual strategy.
        texts = [
            (r["context"] + "\n" + r["text"]) if r["context"] else r["text"] for r in batch
        ]
        mat = embed_documents(texts, batch_size=batch_size)
        dim = int(mat.shape[1])
        conn.executemany(
            "INSERT OR REPLACE INTO vectors (chunk_id, strategy, model, dim, vec) VALUES (?,?,?,?,?)",
            [
                (r["chunk_id"], strategy, config.EMBED_MODEL, dim, mat[i].tobytes())
                for i, r in enumerate(batch)
            ],
        )
        conn.commit()  # durable per batch, so an interrupt costs one batch
        written += len(batch)
        if progress:
            print(
                "    embedded {}/{} ({:.0f}%)".format(written, total, 100 * written / total),
                end="\r",
                flush=True,
            )
    if progress:
        print()
    return {"strategy": strategy, "vectors": written, "skipped": skipped,
            "total": len(rows), "dim": dim}
