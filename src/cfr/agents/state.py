"""State for the multi-agent variant.

Extends the single-graph state with the bookkeeping a supervisor needs: who has
run, how many attempts each specialist has had, and what the auditor concluded
about *whose* fault a failure was.
"""

from __future__ import annotations

from typing import Any, Dict, List

from typing_extensions import TypedDict


class AgentState(TypedDict, total=False):
    # -- input --------------------------------------------------------------
    query: str
    strategy: str

    # -- researcher ---------------------------------------------------------
    # The query actually searched with. The researcher may rewrite the user's
    # wording into regulatory vocabulary - that gap is the whole premise of the
    # project, so it is the one place worth spending a model call.
    search_query: str
    query_variants: List[str]
    hits: List[Dict[str, Any]]
    candidate_count: int
    confidence: float
    research_attempts: int

    # -- writer -------------------------------------------------------------
    answer: str
    citations: List[Dict[str, Any]]
    write_attempts: int
    # Which research generation this draft was written from. An answer whose
    # drafted_from lags research_attempts was written against sources that have
    # since been replaced, and must not ship.
    drafted_from: int
    sufficient: bool

    # -- auditor ------------------------------------------------------------
    citations_dropped: int
    # "writer" - quotes were fabricated or misquoted; the sources were fine.
    # "researcher" - the sources genuinely do not support an answer.
    # "" - nothing wrong.
    fault: str
    audit_note: str

    # -- supervisor ---------------------------------------------------------
    next_agent: str
    route_history: List[str]
    timings_ms: Dict[str, float]

    # -- human in the loop --------------------------------------------------
    needs_review: bool
    review_decision: str
    reviewer_note: str

    # -- outcome ------------------------------------------------------------
    status: str
    abstain_reason: str
    error: str
