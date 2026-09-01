"""Cross-encoder reranking.

The bi-encoder that built the index encoded queries and passages separately -
that is what makes it indexable, and also what caps its accuracy, because the
passage was turned into coordinates before anyone knew what would be asked.

A cross-encoder reads the query and one passage together in a single forward
pass, so attention runs across both. Much more accurate, and impossible to
precompute: cost is linear in candidates, which is exactly why it only ever
sees a shortlist.
"""

from __future__ import annotations

import math
from typing import List, Optional, Sequence

from .. import config

_MODEL = None


def _model():
    global _MODEL
    if _MODEL is None:
        from fastembed.rerank.cross_encoder import TextCrossEncoder

        _MODEL = TextCrossEncoder(model_name=config.RERANK_MODEL)
    return _MODEL


def sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


def to_confidence(logit: float, temperature: Optional[float] = None) -> float:
    """Map a raw cross-encoder logit into a usable (0, 1) confidence.

    A plain sigmoid is useless here: observed logits cluster near +10 for
    relevant passages and -11 for irrelevant ones, and sigmoid maps both to
    1.000 and 0.000. Every score then looks identical, the abstention gate can
    never fire, and the confidence shown to users is a constant.

    Temperature scaling restores the range. It is monotonic, so ranking is
    unaffected - this only changes the number you threshold and display.
    """
    t = config.RERANK_TEMPERATURE if temperature is None else temperature
    return sigmoid(logit / max(t, 1e-6))


def rerank_logits(
    query: str,
    passages: Sequence[str],
    batch_size: Optional[int] = None,
) -> List[float]:
    """Raw cross-encoder scores. Unbounded; higher is better."""
    if not passages:
        return []
    bs = config.RERANK_BATCH_SIZE if batch_size is None else batch_size
    if config.RERANK_MAX_CHARS > 0:
        passages = [p[: config.RERANK_MAX_CHARS] for p in passages]
    return [float(s) for s in _model().rerank(query, list(passages), batch_size=bs)]


def rerank(
    query: str,
    passages: Sequence[str],
    batch_size: Optional[int] = None,
) -> List[float]:
    """Temperature-scaled scores in (0, 1). Higher is better."""
    return [to_confidence(s) for s in rerank_logits(query, passages, batch_size)]


def warm() -> None:
    """Force the model download/load so the first user request is not slow."""
    _model()
    rerank("warmup", ["warmup passage"])
