"""Supervisor-routed multi-agent graph.

                    ┌──────────────┐
       START ──────►│  supervisor  │◄───────────┐
                    └──────┬───────┘            │
                           │ routes             │ every specialist
        ┌──────────┬───────┴───────┬─────────┐  │ reports back
        ▼          ▼               ▼         ▼  │
   researcher   writer         auditor    review│
        └──────────┴───────────────┴────────────┘
                           │ done
                           ▼
                          END

Specialists never call each other. They finish, return state, and the
supervisor decides what happens next - which is what allows the auditor to send
work back to the *researcher* when the sources are the problem, rather than
re-prompting a writer who never had anything to work with.
"""

from __future__ import annotations

from typing import Any, Optional

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from . import roles, supervisor
from .state import AgentState


def _review(state: AgentState):
    """Human gate. Same contract as the single-graph version."""
    from langgraph.types import interrupt

    decision = interrupt({
        "reason": "confidence in review band",
        "query": state["query"],
        "search_query": state.get("search_query"),
        "confidence": state.get("confidence"),
        "answer": (state.get("answer") or "")[:400],
        "citations": len(state.get("citations") or []),
        "route_history": state.get("route_history"),
        "options": ["approve", "reject"],
    })
    verdict = decision.get("decision", "approve") if isinstance(decision, dict) else str(decision)
    note = decision.get("note", "") if isinstance(decision, dict) else ""
    if verdict == "reject":
        return {"review_decision": "reject", "reviewer_note": note,
                "status": "rejected_by_reviewer", "citations": [],
                "answer": "A reviewer withheld this answer." + (" " + note if note else "")}
    return {"review_decision": "approve", "reviewer_note": note, "status": "answered"}


def build_agent_graph(checkpointer: Optional[Any] = None, with_review: bool = True):
    g = StateGraph(AgentState)

    g.add_node("supervisor", supervisor.supervise)
    g.add_node("researcher", roles.researcher)
    g.add_node("writer", roles.writer)
    g.add_node("auditor", roles.auditor)
    if with_review:
        g.add_node("review", _review)

    g.add_edge(START, "supervisor")

    targets = {
        supervisor.RESEARCHER: "researcher",
        supervisor.WRITER: "writer",
        supervisor.AUDITOR: "auditor",
        supervisor.DONE: END,
    }
    targets[supervisor.REVIEW] = "review" if with_review else END
    g.add_conditional_edges("supervisor", supervisor.route, targets)

    # Every specialist reports back rather than choosing a successor.
    for node in ("researcher", "writer", "auditor"):
        g.add_edge(node, "supervisor")
    if with_review:
        g.add_edge("review", END)

    if checkpointer is None and with_review:
        checkpointer = MemorySaver()
    return g.compile(checkpointer=checkpointer)


def render_mermaid() -> str:
    return build_agent_graph(with_review=True).get_graph().draw_mermaid()
