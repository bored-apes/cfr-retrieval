"""Build the labelling pool.

The mistake that quietly ruins a relevance set is labelling whatever your
current system returns: the ground truth then encodes your system's biases and
guarantees it scores well. Pooling is the standard fix - run several different
configurations, take the top-N from each, label the union. Systems you have not
built yet get a fair shot at documents your current one never surfaces.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

from .. import config, db
from ..search import Retriever
from .run import ABLATION, load_queries


def main(args) -> int:
    conn = db.connect()
    db.init(conn)
    queries = load_queries(getattr(args, "queries", None))
    depth = getattr(args, "depth", 20)
    out_path = Path(getattr(args, "out", None) or (config.EVAL_DIR / "pool.jsonl"))

    judged = {
        (r["query_id"], r["doc_id"])
        for r in conn.execute("SELECT query_id, doc_id FROM qrels")
    }

    from .. import embed

    pooled: List[Dict] = []
    for n, q in enumerate(queries, 1):
        qid = q["query_id"]
        qvec = embed.embed_query(q["query"])
        seen: Dict[str, Dict] = {}

        for cfg in ABLATION:
            cfg_deep = type(cfg)(**{**cfg.__dict__, "final_k": depth, "rerank_top_n": depth * 3})
            res = Retriever(conn, cfg_deep).search(q["query"], qvec=qvec)
            for rank, hit in enumerate(res.hits[:depth], start=1):
                entry = seen.setdefault(hit.doc_id, {
                    "doc_id": hit.doc_id,
                    "citation": hit.citation,
                    "heading": hit.heading,
                    "snippet": " ".join(hit.text.split())[:600],
                    "found_by": [],
                    "best_rank": rank,
                })
                entry["found_by"].append(cfg.name)
                entry["best_rank"] = min(entry["best_rank"], rank)

        for doc_id, entry in seen.items():
            if (qid, doc_id) in judged:
                continue
            entry["found_by"] = sorted(set(entry["found_by"]))
            pooled.append({"query_id": qid, "query": q["query"], "type": q.get("type", ""), **entry})

        print("  [{:>3}/{}] {:<58} {:>3} candidates".format(
            n, len(queries), q["query"][:58], len(seen)), flush=True)

    # Rarely-found documents first: those are where pooling earns its keep, and
    # where your current system is most likely to be wrong.
    pooled.sort(key=lambda e: (len(e["found_by"]), e["best_rank"]))
    out_path.write_text(
        "\n".join(json.dumps(e, ensure_ascii=False) for e in pooled) + "\n", encoding="utf-8"
    )

    print("\n{} unjudged (query, document) pairs -> {}".format(len(pooled), out_path))
    print("Judge them at http://127.0.0.1:8000/label after `cfr serve`.")
    print("\nGrades: 0 irrelevant | 1 background | 2 partly answers | 3 directly answers")
    print("Budget roughly two hours per fifty queries. This is the differentiator;")
    print("it is meant to be slow.")
    return 0
