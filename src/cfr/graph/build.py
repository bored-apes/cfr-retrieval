"""Graph construction.

    START → retrieve → rerank → gate ─┬─ abstain ────────────────→ END
                                      └─ generate → verify ─┬─ retry → generate
                                                            ├─ review → finalise
                                                            └─ done  → finalise

Two things here that the hand-rolled pipeline in `cfr.answer` does not have:

  * a **self-correction loop** — if verification rejects every citation, the
    graph returns to `generate` once with the failing quotes in the prompt,
    instead of serving an answer with its citations silently stripped;
  * an **HITL gate** — answers whose confidence lands just above the abstention
    threshold suspend at `review` until a human approves or rejects, with state
    persisted by the checkpointer so the pause can outlive the process.
"""

from __future__ import annotations

from typing import Any, Optional

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from . import nodes
from .state import CFRState


def build_graph(checkpointer: Optional[Any] = None, with_review: bool = True):
    """Compile the CFR pipeline as a state graph.

    `with_review=False` drops the human gate, which is what the benchmark uses
    so the comparison against the hand-rolled pipeline stays apples-to-apples.
    """
    g = StateGraph(CFRState)

    g.add_node("retrieve", nodes.retrieve)
    g.add_node("rerank", nodes.rerank)
    g.add_node("gate", nodes.gate)
    g.add_node("abstain", nodes.abstain)
    g.add_node("generate", nodes.generate)
    g.add_node("verify", nodes.verify)
    g.add_node("finalise", nodes.finalise)
    if with_review:
        g.add_node("review", nodes.review)

    g.add_edge(START, "retrieve")
    g.add_edge("retrieve", "rerank")
    g.add_edge("rerank", "gate")

    g.add_conditional_edges("gate", nodes.route_after_gate,
                            {"abstain": "abstain", "generate": "generate"})
    g.add_edge("abstain", END)
    g.add_edge("generate", "verify")

    routes = {"retry": "generate", "done": "finalise"}
    routes["review"] = "review" if with_review else "finalise"
    g.add_conditional_edges("verify", nodes.route_after_verify, routes)

    if with_review:
        g.add_edge("review", "finalise")
    g.add_edge("finalise", END)

    # A checkpointer is required for interrupt() to be resumable; default to an
    # in-process saver so the graph is usable without external storage.
    if checkpointer is None and with_review:
        checkpointer = MemorySaver()
    return g.compile(checkpointer=checkpointer)


def render_mermaid() -> str:
    """Mermaid source for the compiled graph - useful in the README."""
    return build_graph(with_review=True).get_graph().draw_mermaid()
