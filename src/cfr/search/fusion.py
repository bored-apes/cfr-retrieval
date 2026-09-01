"""Reciprocal rank fusion.

BM25 scores are unbounded and cosine similarities sit in [-1, 1]. Blending them
directly means normalising two distributions that shift per query, which is
fragile. RRF throws the scores away and uses only rank position, which removes
the problem entirely and is very hard to beat.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Sequence, Tuple


def rrf(
    rankings: Sequence[Sequence[Tuple[str, float]]],
    k: int = 60,
    weights: Sequence[float] = (),
) -> List[Tuple[str, float]]:
    """Fuse ranked lists into one.

    rankings: one ranked (id, score) list per retriever, best first.
    k:        damping constant. 60 is the value from the original paper and
              almost never needs tuning.
    weights:  optional per-retriever multiplier, defaults to 1.0 each.
    """
    if not weights:
        weights = [1.0] * len(rankings)
    if len(weights) != len(rankings):
        raise ValueError("weights must match the number of rankings")

    scores: Dict[str, float] = defaultdict(float)
    for ranking, weight in zip(rankings, weights):
        for rank, (doc_id, _score) in enumerate(ranking, start=1):
            scores[doc_id] += weight / (k + rank)
    return sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
