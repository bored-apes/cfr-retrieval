"""Retrieval metrics, plus the bootstrap that stops you overclaiming.

All metrics operate on *document* ids, not chunk ids: a query is answered by a
CFR section, and which chunk of that section surfaced is an implementation
detail. Ranked chunk lists are collapsed to first-occurrence document order
before scoring.
"""

from __future__ import annotations

import math
import random
from typing import Dict, List, Optional, Sequence, Tuple

Qrels = Dict[str, int]  # doc_id -> graded relevance, 0..3


def dedupe_docs(ranked_doc_ids: Sequence[str]) -> List[str]:
    """Keep first occurrence only, preserving order."""
    seen = set()
    out = []
    for d in ranked_doc_ids:
        if d not in seen:
            seen.add(d)
            out.append(d)
    return out


def recall_at_k(ranked: Sequence[str], qrels: Qrels, k: int) -> Optional[float]:
    relevant = {d for d, g in qrels.items() if g > 0}
    if not relevant:
        return None  # undefined for unanswerable queries
    found = len(relevant.intersection(ranked[:k]))
    return found / len(relevant)


def mrr_at_k(ranked: Sequence[str], qrels: Qrels, k: int) -> Optional[float]:
    if not any(g > 0 for g in qrels.values()):
        return None
    for i, doc_id in enumerate(ranked[:k], start=1):
        if qrels.get(doc_id, 0) > 0:
            return 1.0 / i
    return 0.0


def ndcg_at_k(ranked: Sequence[str], qrels: Qrels, k: int) -> Optional[float]:
    """Graded nDCG with the standard 2^rel - 1 gain and log2 discount.

    Graded relevance is why the labelling guide asks for 0-3 rather than a
    yes/no: with binary labels this collapses to something much less sensitive
    to the difference between "the answer" and "related background".
    """
    if not any(g > 0 for g in qrels.values()):
        return None

    dcg = 0.0
    for i, doc_id in enumerate(ranked[:k], start=1):
        gain = (2 ** qrels.get(doc_id, 0)) - 1
        if gain:
            dcg += gain / math.log2(i + 1)

    ideal = sorted((g for g in qrels.values() if g > 0), reverse=True)[:k]
    idcg = sum(((2 ** g) - 1) / math.log2(i + 1) for i, g in enumerate(ideal, start=1))
    return (dcg / idcg) if idcg > 0 else None


def bootstrap_ci(
    values: Sequence[float],
    iterations: int = 1000,
    confidence: float = 0.95,
    seed: int = 20240901,
) -> Tuple[float, float, float]:
    """Mean and a percentile bootstrap CI, resampling queries with replacement.

    With ~200 queries this typically puts a +/- 0.02-0.04 band on nDCG. Any
    difference smaller than that band is noise, and reporting it as an
    improvement is the single most common way these projects mislead.
    """
    vals = [v for v in values if v is not None]
    if not vals:
        return (float("nan"), float("nan"), float("nan"))
    mean = sum(vals) / len(vals)
    if len(vals) < 2 or iterations < 2:
        return (mean, mean, mean)

    rng = random.Random(seed)
    n = len(vals)
    means = []
    for _ in range(iterations):
        means.append(sum(vals[rng.randrange(n)] for _ in range(n)) / n)
    means.sort()
    lo_i = int((1 - confidence) / 2 * iterations)
    hi_i = min(iterations - 1, int((1 + confidence) / 2 * iterations))
    return (mean, means[lo_i], means[hi_i])


def paired_bootstrap(
    a: Sequence[Optional[float]],
    b: Sequence[Optional[float]],
    iterations: int = 1000,
    seed: int = 20240901,
) -> Dict[str, float]:
    """Is config B actually better than config A, on the same queries?

    Paired resampling is the right test here because both configs answered the
    identical query set - comparing two independent CIs throws that pairing
    away and is needlessly conservative.
    """
    pairs = [(x, y) for x, y in zip(a, b) if x is not None and y is not None]
    if len(pairs) < 2:
        return {"delta": float("nan"), "lo": float("nan"), "hi": float("nan"), "p_better": float("nan")}

    rng = random.Random(seed)
    n = len(pairs)
    observed = sum(y - x for x, y in pairs) / n
    deltas = []
    for _ in range(iterations):
        s = 0.0
        for _ in range(n):
            x, y = pairs[rng.randrange(n)]
            s += y - x
        deltas.append(s / n)
    deltas.sort()
    lo = deltas[int(0.025 * iterations)]
    hi = deltas[min(iterations - 1, int(0.975 * iterations))]
    p_better = sum(1 for d in deltas if d > 0) / len(deltas)
    return {"delta": observed, "lo": lo, "hi": hi, "p_better": p_better}


def significant(lo: float, hi: float) -> bool:
    """True when the CI for a difference excludes zero."""
    if math.isnan(lo) or math.isnan(hi):
        return False
    return (lo > 0 and hi > 0) or (lo < 0 and hi < 0)
