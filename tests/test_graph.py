"""Graph routing.

The node bodies call models; the routing functions do not, so the control flow
is tested directly. These are the branches that decide whether a user gets an
answer, a refusal, a retry or a human — worth pinning down.
"""

import pytest

from cfr.graph import build_graph
from cfr.graph import nodes


# --- gate ------------------------------------------------------------------

def test_gate_abstains_below_threshold(monkeypatch):
    monkeypatch.setattr("cfr.config.ABSTAIN_THRESHOLD", 0.20)
    out = nodes.gate({"hits": [{"rerank_score": 0.05}], "confidence": 0.05})
    assert out["abstained"] and out["abstain_reason"] == "low_confidence"


def test_gate_abstains_with_no_hits():
    out = nodes.gate({"hits": [], "confidence": 0.0})
    assert out["abstained"] and out["abstain_reason"] == "no_results"


def test_gate_flags_marginal_confidence_for_review(monkeypatch):
    """Just above the threshold is exactly where a reviewer is worth the time."""
    monkeypatch.setattr("cfr.config.ABSTAIN_THRESHOLD", 0.20)
    out = nodes.gate({"hits": [{"rerank_score": 0.25}], "confidence": 0.25})
    assert not out["abstained"]
    assert out["needs_review"] is True


def test_gate_lets_confident_answers_through_unreviewed(monkeypatch):
    monkeypatch.setattr("cfr.config.ABSTAIN_THRESHOLD", 0.20)
    out = nodes.gate({"hits": [{"rerank_score": 0.91}], "confidence": 0.91})
    assert not out["abstained"]
    assert out["needs_review"] is False


@pytest.mark.parametrize("abstained,expected", [(True, "abstain"), (False, "generate")])
def test_route_after_gate(abstained, expected):
    assert nodes.route_after_gate({"abstained": abstained}) == expected


# --- verify ----------------------------------------------------------------

def test_retries_once_when_every_citation_fails_verification():
    """The loop the hand-rolled pipeline lacks: it would serve the answer with
    citations silently stripped instead."""
    state = {"citations": [], "verify_feedback": "quote not found", "generate_attempts": 1}
    assert nodes.route_after_verify(state) == "retry"


def test_does_not_retry_forever():
    state = {"citations": [], "verify_feedback": "quote not found",
             "generate_attempts": nodes.MAX_GENERATE_ATTEMPTS}
    assert nodes.route_after_verify(state) != "retry"


def test_proceeds_when_citations_verify():
    state = {"citations": [{"source": 1}], "verify_feedback": "",
             "generate_attempts": 1, "needs_review": False}
    assert nodes.route_after_verify(state) == "done"


def test_routes_to_review_when_flagged():
    state = {"citations": [{"source": 1}], "verify_feedback": "",
             "generate_attempts": 1, "needs_review": True}
    assert nodes.route_after_verify(state) == "review"


def test_generation_failure_skips_retry_and_review():
    """A provider outage is not something a retry or a human can fix."""
    state = {"status": "generation_failed", "citations": [], "needs_review": True}
    assert nodes.route_after_verify(state) == "done"


# --- review ----------------------------------------------------------------

def test_reviewer_rejection_withholds_the_answer(monkeypatch):
    monkeypatch.setattr("langgraph.types.interrupt",
                        lambda payload: {"decision": "reject", "note": "wrong section"})
    out = nodes.review({"query": "q", "confidence": 0.25, "answer": "a", "citations": []})
    assert out["status"] == "rejected_by_reviewer"
    assert out["citations"] == []
    assert "wrong section" in out["answer"]


def test_reviewer_approval_releases_the_answer(monkeypatch):
    monkeypatch.setattr("langgraph.types.interrupt", lambda payload: {"decision": "approve"})
    out = nodes.review({"query": "q", "confidence": 0.25, "answer": "a", "citations": [1]})
    assert out["status"] == "answered"
    assert out["review_decision"] == "approve"


# --- structure -------------------------------------------------------------

def test_graph_compiles_with_and_without_review():
    assert build_graph(with_review=True) is not None
    assert build_graph(with_review=False, checkpointer=None) is not None


def test_graph_has_the_expected_topology():
    m = build_graph(with_review=True).get_graph().draw_mermaid()
    for edge in ("retrieve --> rerank", "rerank --> gate", "generate --> verify"):
        assert edge in m, edge
    assert "retry" in m, "self-correction loop missing"
    assert "review" in m, "HITL gate missing"
