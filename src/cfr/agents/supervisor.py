"""The supervisor.

Holds routing authority: specialists finish and report, and this decides who
runs next. That is what makes it multi-agent rather than a pipeline with
renamed stages - the researcher does not know the writer exists.

**Routing is deterministic on purpose.** An LLM supervisor is the fashionable
choice, but every decision here is a function of state that has already been
measured: a confidence score against a calibrated threshold, attempt counters
against a budget, and the auditor's fault attribution. Handing those to a model
would add latency, cost and nondeterminism to a decision that has an exactly
correct answer. Model judgment is spent where it earns its place instead - the
researcher's query reformulation, where the whole problem is vocabulary.

The invariant the routing enforces:

    an answer may only ship if it was drafted from the sources currently
    in state, and every quote in it was verified against them.

That is why `drafted_from` exists. Without it, sending work back to the
researcher produces new sources while the previous draft - written against
sources that are now gone - stays in state and ships.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

from .. import config
from . import roles
from .state import AgentState

DONE = "done"
RESEARCHER = "researcher"
WRITER = "writer"
AUDITOR = "auditor"
REVIEW = "review"

Decision = Tuple[str, Optional[str], Optional[str]]


def supervise(state: AgentState) -> Dict[str, Any]:
    """Decide who runs next, and record the decision in state."""
    decision, status, reason = _decide(state)

    out: Dict[str, Any] = {
        "next_agent": decision,
        "route_history": list(state.get("route_history") or []) + [decision],
    }
    if status:
        out["status"] = status
    if reason:
        out["abstain_reason"] = reason

    if decision == DONE:
        out.update(_terminal(state, status or state.get("status") or "", reason or ""))
    return out


def _decide(state: AgentState) -> Decision:
    status = state.get("status") or ""

    # A provider outage is not something more routing can fix.
    if status in ("generation_failed", "retrieval_only"):
        return DONE, None, None

    research_attempts = int(state.get("research_attempts") or 0)
    write_attempts = int(state.get("write_attempts") or 0)
    drafted_from = int(state.get("drafted_from") or 0)
    confidence = state.get("confidence") or 0.0
    hits = state.get("hits") or []

    # 1. Nothing retrieved yet.
    if research_attempts == 0:
        return RESEARCHER, None, None

    # 2. Retrieval was weak. Rewrite the question into regulatory vocabulary
    #    before giving up - the user not knowing the right words is the failure
    #    mode this system exists to fix.
    if confidence < config.ABSTAIN_THRESHOLD:
        if research_attempts < roles.max_research_attempts():
            return RESEARCHER, None, None
        return DONE, "abstained", ("no_results" if not hits else "low_confidence")

    # 3. No draft exists for the sources currently in state - either nothing has
    #    been written yet, or the researcher has since replaced the sources the
    #    existing draft was written against.
    if drafted_from != research_attempts:
        if write_attempts < roles.MAX_WRITE_ATTEMPTS:
            return WRITER, None, None
        return DONE, "unverified", "drafting_budget_exhausted"

    # 4. Drafted but not yet audited.
    if status == "drafted":
        return AUDITOR, None, None

    # 5. Audited. Act on the attribution.
    fault = state.get("fault") or ""
    if fault == "writer" and write_attempts < roles.MAX_WRITE_ATTEMPTS:
        return WRITER, None, None
    if fault == "researcher" and research_attempts < roles.max_research_attempts():
        return RESEARCHER, None, None
    if fault and not state.get("citations"):
        # Out of retries with nothing verifiable. Refusing beats shipping an
        # answer whose sources could not be confirmed.
        return DONE, "unverified", "citations_unverifiable"

    return DONE, "answered", None


def _terminal(state: AgentState, status: str, reason: str) -> Dict[str, Any]:
    """Fill in the user-facing outcome for a finished run."""
    conf = state.get("confidence") or 0.0

    if status == "abstained":
        msg = {
            "no_results": "Nothing in the indexed parts of the CFR matched this question.",
            "low_confidence": (
                "The closest sections scored {:.2f}, below the {:.2f} confidence "
                "threshold, so this is left unanswered rather than guessed. The "
                "nearest matches are shown below."
            ).format(conf, config.ABSTAIN_THRESHOLD),
        }.get(reason, "Not answered.")
        return {"answer": msg, "citations": [], "needs_review": False}

    if status == "unverified":
        return {
            "answer": (
                "A draft was produced but its quotes could not be matched back to "
                "the sections it cited, so it is withheld. The retrieved sections "
                "are shown below."
            ),
            "citations": [],
            "needs_review": False,
        }

    if status == "answered":
        return {
            "needs_review": bool(state.get("citations"))
            and conf < config.ABSTAIN_THRESHOLD + roles.REVIEW_BAND
        }
    return {}


def route(state: AgentState) -> str:
    """Edge function - reads the decision the supervisor already recorded."""
    nxt = state.get("next_agent") or DONE
    if nxt == DONE and state.get("needs_review"):
        return REVIEW
    return nxt
