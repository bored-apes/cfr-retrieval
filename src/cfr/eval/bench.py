"""Latency benchmark across rerank depths.

The reranker is the only stage whose cost scales with how many candidates you
feed it, so `rerank_top_n` is the main latency dial in the system. This measures
what each setting costs so the choice is made from numbers rather than a guess.

Pair it with `cfr eval` at the same depths: cost here, benefit there.
"""

from __future__ import annotations

import time
from typing import Dict, List, Sequence

from .. import db
from ..search import RetrievalConfig, Retriever
from .run import load_queries


DEPTHS = (10, 25, 50, 100)


def _percentile(values: Sequence[float], p: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(p * len(ordered)))]


def main(args) -> int:
    conn = db.connect(readonly=True)
    queries = load_queries(getattr(args, "queries", None))
    sample = queries[: getattr(args, "n", 20)]
    strategy = getattr(args, "strategy", "structured")
    depths = getattr(args, "depths", None) or list(DEPTHS)

    from .. import embed
    from ..search import rerank

    print("warming models...")
    embed.embed_query("warmup")
    rerank.warm()

    # Baseline: the recall stage on its own.
    base_cfg = RetrievalConfig(name="no rerank", strategy=strategy, use_rerank=False, final_k=10)
    base = Retriever(conn, base_cfg)
    stage_times: Dict[str, List[float]] = {"lexical": [], "dense": [], "fusion": []}
    totals: List[float] = []
    for q in sample:
        t0 = time.perf_counter()
        res = base.search(q["query"])
        totals.append((time.perf_counter() - t0) * 1000)
        for k in stage_times:
            if k in res.timings_ms:
                stage_times[k].append(res.timings_ms[k])

    print("\n{} queries, strategy={}\n".format(len(sample), strategy))
    print("recall stage (no rerank)")
    for k, vals in stage_times.items():
        if vals:
            print("  {:<10} p50 {:>7.0f} ms   p95 {:>7.0f} ms".format(
                k, _percentile(vals, 0.5), _percentile(vals, 0.95)))
    print("  {:<10} p50 {:>7.0f} ms   p95 {:>7.0f} ms".format(
        "TOTAL", _percentile(totals, 0.5), _percentile(totals, 0.95)))

    head = "\n{:>12} {:>12} {:>12} {:>14}".format("rerank_top_n", "p50 ms", "p95 ms", "vs no-rerank")
    print(head)
    print("-" * (len(head) - 1))
    baseline_p50 = _percentile(totals, 0.5)
    print("{:>12} {:>12.0f} {:>12.0f} {:>14}".format(
        0, baseline_p50, _percentile(totals, 0.95), "-"))

    for depth in depths:
        cfg = RetrievalConfig(name="rerank@{}".format(depth), strategy=strategy,
                              use_rerank=True, rerank_top_n=depth, final_k=10)
        r = Retriever(conn, cfg)
        times: List[float] = []
        for q in sample:
            t0 = time.perf_counter()
            r.search(q["query"])
            times.append((time.perf_counter() - t0) * 1000)
        p50 = _percentile(times, 0.5)
        print("{:>12} {:>12.0f} {:>12.0f} {:>13.1f}x".format(
            depth, p50, _percentile(times, 0.95),
            p50 / baseline_p50 if baseline_p50 else float("nan")))

    print("\nCost only. Run `cfr eval` at the same depths for the benefit side -")
    print("a depth is worth paying for only if nDCG@10 moves more than the CI width.")
    return 0
