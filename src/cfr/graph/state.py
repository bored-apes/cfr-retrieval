"""State passed between graph nodes.

Everything here must survive checkpoint serialisation, so hits travel as plain
dicts rather than Hit dataclasses and the query embedding is kept as a list.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from typing_extensions import TypedDict


class CFRState(TypedDict, total=False):
    # -- input --------------------------------------------------------------
    query: str
    strategy: str

    # -- retrieval ----------------------------------------------------------
    qvec: Optional[List[float]]
    hits: List[Dict[str, Any]]
    candidate_count: int
    timings_ms: Dict[str, float]

    # -- gate ---------------------------------------------------------------
    confidence: float
    abstained: bool
    abstain_reason: str

    # -- generation ---------------------------------------------------------
    answer: str
    citations: List[Dict[str, Any]]
    citations_dropped: int
    # Fed back into the prompt when a retry is triggered, so the second attempt
    # knows which quotes failed rather than repeating the same mistake.
    verify_feedback: str
    generate_attempts: int

    # -- human in the loop --------------------------------------------------
    needs_review: bool
    review_decision: str          # "approve" | "reject" | "" (not yet reviewed)
    reviewer_note: str

    # -- outcome ------------------------------------------------------------
    status: str
    error: str
