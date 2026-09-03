"""Export the index as a static bundle the browser can serve without a server.

The whole pipeline moves client-side: BM25 is rebuilt in JS from the shipped
chunk text, dense vectors ship int8-quantised, and both models are pulled from
the HF CDN by transformers.js and cached in the browser. That makes the app a
pile of static files, which is free to host anywhere forever.

Run: python scripts/export_static.py
"""

from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cfr import config, db, embed  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "static" / "data"
STRATEGY = "structured"
CANARY_TEXT = "How long can a large quantity generator keep hazardous waste on site?"


def write_gz(path: Path, obj) -> int:
    raw = json.dumps(obj, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    with gzip.open(path, "wb", compresslevel=9) as fh:
        fh.write(raw)
    return path.stat().st_size


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    conn = db.connect(readonly=True)

    # --- documents: needed for span highlighting in the source view ---------
    docs = {}
    for r in conn.execute("SELECT doc_id, citation, heading, text, source_url FROM documents"):
        docs[r["doc_id"]] = {
            "c": r["citation"],
            "h": r["heading"],
            "t": r["text"],
            "u": r["source_url"] or "",
        }
    n_docs = write_gz(OUT / "docs.json.gz", docs)

    # --- chunks: retrieval units, with offsets back into the documents ------
    rows = conn.execute(
        "SELECT chunk_id, doc_id, text, char_start, char_end "
        "FROM chunks WHERE strategy = ? ORDER BY chunk_id",
        (STRATEGY,),
    ).fetchall()
    chunks = [
        {
            "i": r["chunk_id"],
            "d": r["doc_id"],
            "t": r["text"],
            "s": int(r["char_start"]),
            "e": int(r["char_end"]),
        }
        for r in rows
    ]
    n_chunks = write_gz(OUT / "chunks.json.gz", chunks)

    # --- vectors: int8, in the same order as chunks -------------------------
    vec_rows = conn.execute(
        "SELECT chunk_id, dim, vec FROM vectors WHERE strategy = ? ORDER BY chunk_id",
        (STRATEGY,),
    ).fetchall()
    if [r["chunk_id"] for r in vec_rows] != [c["i"] for c in chunks]:
        raise SystemExit("chunk/vector order mismatch - the browser indexes by row number")

    dim = int(vec_rows[0]["dim"])
    mat = np.frombuffer(b"".join(r["vec"] for r in vec_rows), dtype=np.float32).reshape(
        len(vec_rows), dim
    )
    mat = embed.normalize(mat.copy())

    # A single global scale. Components of a normalised 384-d vector cluster
    # near 1/sqrt(384) ~= 0.05, so scaling by 127 (assuming max 1.0) would throw
    # away most of the resolution; scale by the actual maximum instead.
    vmax = float(np.abs(mat).max())
    scale = 127.0 / vmax
    q = np.clip(np.rint(mat * scale), -127, 127).astype(np.int8)
    (OUT / "vectors.i8.bin").write_bytes(q.tobytes())

    # Prove the quantisation did not change what gets retrieved.
    deq = q.astype(np.float32) / scale
    deq /= np.linalg.norm(deq, axis=1, keepdims=True)
    rng = np.random.default_rng(0)
    probes = rng.choice(len(mat), size=min(50, len(mat)), replace=False)
    overlap = []
    for i in probes:
        a = np.argsort(-(mat @ mat[i]))[:10]
        b = np.argsort(-(deq @ deq[i]))[:10]
        overlap.append(len(set(a.tolist()) & set(b.tolist())) / 10.0)
    top10_agreement = float(np.mean(overlap))

    canary = embed.embed_query(CANARY_TEXT)
    canary = canary / np.linalg.norm(canary)

    meta = {
        "strategy": STRATEGY,
        "documents": len(docs),
        "chunks": len(chunks),
        "dim": dim,
        "vector_scale": scale,
        "embed_model": "Xenova/bge-small-en-v1.5",
        "rerank_model": "Xenova/ms-marco-MiniLM-L-6-v2",
        # No instruction prefix: the index was built without one (verified -
        # fastembed's query_embed is identical to embed for this model), so the
        # browser must not add one either.
        "query_prefix": "",
        "abstain_threshold": config.ABSTAIN_THRESHOLD,
        "rerank_temperature": config.RERANK_TEMPERATURE,
        "rerank_top_n": config.RERANK_TOP_N,
        "rrf_k": config.RRF_K,
        "candidates": config.CANDIDATES_PER_RETRIEVER,
        "quote_match_threshold": config.QUOTE_MATCH_THRESHOLD,
        "int8_top10_agreement": round(top10_agreement, 4),
        # A reference embedding the browser re-computes at boot. If its port
        # gets pooling or the query prefix wrong, every dense result degrades
        # silently; comparing against this catches it immediately and loudly.
        "canary": {
            "text": CANARY_TEXT,
            "vector": [round(float(x), 6) for x in canary],
        },
    }
    (OUT / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    mb = lambda b: "{:.1f} MB".format(b / 1e6)  # noqa: E731
    print("wrote to {}".format(OUT))
    print("  docs.json.gz      {:>9}  ({} documents)".format(mb(n_docs), len(docs)))
    print("  chunks.json.gz    {:>9}  ({} chunks)".format(mb(n_chunks), len(chunks)))
    print("  vectors.i8.bin    {:>9}  ({}x{} int8)".format(
        mb((OUT / "vectors.i8.bin").stat().st_size), len(chunks), dim))
    total = n_docs + n_chunks + (OUT / "vectors.i8.bin").stat().st_size
    print("  ---------------------------")
    print("  initial payload   {:>9}".format(mb(total)))
    print("\n  int8 top-10 agreement vs float32: {:.1%}".format(top10_agreement))
    if top10_agreement < 0.95:
        print("  ! quantisation is changing results - consider float16 instead")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
