"""Supervisor routing and fault attribution.

The specialists call models; the supervisor does not. So the routing is tested
directly, branch by branch, and the specialists are faked for the end-to-end
runs. What is being pinned down here is the thing that actually decides what a
user receives: who runs next, and when the system gives up.
"""

import pytest

from cfr.agents import build_agent_graph, supervisor
from cfr.agents import roles

R, W, A, D = supervisor.RESEARCHER, supervisor.WRITER, supervisor.AUDITOR, supervisor.DONE

CONFIDENT = 0.90


@pytest.fixture(autouse=True)
def _threshold(monkeypatch):
    monkeypatch.setattr("cfr.config.ABSTAIN_THRESHOLD", 0.20)
    # Reformulation ships disabled - see the block below for why, and
    # test_reformulation_is_off_by_default for the guarantee. The routing tests
    # enable it so the second-attempt branches are reachable at all.
    monkeypatch.setattr("cfr.config.ENABLE_REFORMULATION", True)


def decide(**state):
    return supervisor.supervise(state)["next_agent"]


# --- cold start ------------------------------------------------------------

def test_starts_with_the_researcher():
    assert decide(query="q") == R


# --- weak retrieval --------------------------------------------------------

def test_reformulates_before_giving_up():
    """A first pass that scores badly buys a second one, not a refusal - the
    user not knowing the regulator's vocabulary is the whole problem."""
    assert decide(research_attempts=1, confidence=0.05, hits=[{"a": 1}]) == R


def test_abstains_once_reformulation_is_spent():
    out = supervisor.supervise({"research_attempts": roles.MAX_RESEARCH_ATTEMPTS,
                                "confidence": 0.05, "hits": [{"a": 1}]})
    assert out["next_agent"] == D
    assert out["status"] == "abstained"
    assert out["abstain_reason"] == "low_confidence"


def test_abstains_with_no_results_at_all():
    out = supervisor.supervise({"research_attempts": roles.MAX_RESEARCH_ATTEMPTS,
                                "confidence": 0.0, "hits": []})
    assert out["abstain_reason"] == "no_results"
    assert out["citations"] == []
    assert "Nothing" in out["answer"]


# --- drafting --------------------------------------------------------------

def test_hands_confident_sources_to_the_writer():
    assert decide(research_attempts=1, confidence=CONFIDENT, hits=[{"a": 1}]) == W


def test_sends_a_draft_to_the_auditor():
    assert decide(research_attempts=1, confidence=CONFIDENT, hits=[{"a": 1}],
                  write_attempts=1, drafted_from=1, status="drafted") == A


# --- fault attribution -----------------------------------------------------

def test_a_misquote_is_the_writers_fault_and_buys_a_rewrite():
    assert decide(research_attempts=1, confidence=CONFIDENT, hits=[{"a": 1}],
                  write_attempts=1, drafted_from=1, status="audited",
                  fault="writer", citations=[]) == W


def test_unsupportable_sources_send_work_back_to_the_researcher():
    """Re-prompting a writer who had nothing to work with just produces a more
    confident fabrication. The sources are the problem, so fix the sources."""
    assert decide(research_attempts=1, confidence=CONFIDENT, hits=[{"a": 1}],
                  write_attempts=1, drafted_from=1, status="audited",
                  fault="researcher", citations=[]) == R


def test_a_clean_audit_finishes():
    out = supervisor.supervise({"research_attempts": 1, "confidence": CONFIDENT,
                                "hits": [{"a": 1}], "write_attempts": 1,
                                "drafted_from": 1, "status": "audited",
                                "fault": "", "citations": [{"source": 1}]})
    assert out["next_agent"] == D
    assert out["status"] == "answered"


# --- the invariant ---------------------------------------------------------

def test_a_draft_never_outlives_the_sources_it_was_written_from():
    """After the researcher replaces the sources, the previous draft is stale.
    Shipping it would attach verified-looking citations to sections that are no
    longer in evidence."""
    nxt = decide(research_attempts=2, confidence=CONFIDENT, hits=[{"a": 1}],
                 write_attempts=1, drafted_from=1, status="audited", fault="")
    assert nxt == W, "stale draft was allowed to ship"


def test_gives_up_when_the_drafting_budget_is_gone():
    out = supervisor.supervise({"research_attempts": 2, "confidence": CONFIDENT,
                                "hits": [{"a": 1}],
                                "write_attempts": roles.MAX_WRITE_ATTEMPTS,
                                "drafted_from": 1, "status": "audited"})
    assert out["next_agent"] == D
    assert out["status"] == "unverified"
    assert out["citations"] == []


def test_refuses_rather_than_shipping_unverifiable_citations():
    out = supervisor.supervise({"research_attempts": roles.MAX_RESEARCH_ATTEMPTS,
                                "confidence": CONFIDENT, "hits": [{"a": 1}],
                                "write_attempts": roles.MAX_WRITE_ATTEMPTS,
                                "drafted_from": roles.MAX_RESEARCH_ATTEMPTS,
                                "status": "audited", "fault": "writer",
                                "citations": []})
    assert out["status"] == "unverified"
    assert out["abstain_reason"] == "citations_unverifiable"
    assert "withheld" in out["answer"]


# --- budgets and outages ---------------------------------------------------

@pytest.mark.parametrize("status", ["generation_failed", "retrieval_only"])
def test_an_outage_is_not_routed_around(status):
    assert decide(status=status, research_attempts=1, hits=[{"a": 1}],
                  confidence=CONFIDENT) == D


def test_writer_retries_are_capped():
    assert decide(research_attempts=1, confidence=CONFIDENT, hits=[{"a": 1}],
                  write_attempts=roles.MAX_WRITE_ATTEMPTS, drafted_from=1,
                  status="audited", fault="writer", citations=[]) == D


def test_researcher_retries_are_capped():
    assert decide(research_attempts=roles.MAX_RESEARCH_ATTEMPTS,
                  confidence=CONFIDENT, hits=[{"a": 1}], write_attempts=1,
                  drafted_from=roles.MAX_RESEARCH_ATTEMPTS, status="audited",
                  fault="researcher", citations=[]) == D


# --- human review ----------------------------------------------------------

def test_marginal_answers_go_to_a_human():
    out = supervisor.supervise({"research_attempts": 1, "confidence": 0.25,
                                "hits": [{"a": 1}], "write_attempts": 1,
                                "drafted_from": 1, "status": "audited",
                                "citations": [{"source": 1}]})
    assert out["needs_review"] is True
    assert supervisor.route({**out}) == supervisor.REVIEW


def test_confident_answers_ship_unreviewed():
    out = supervisor.supervise({"research_attempts": 1, "confidence": CONFIDENT,
                                "hits": [{"a": 1}], "write_attempts": 1,
                                "drafted_from": 1, "status": "audited",
                                "citations": [{"source": 1}]})
    assert out["needs_review"] is False
    assert supervisor.route({**out}) == D


def test_an_abstention_is_never_sent_for_review():
    """There is nothing for a reviewer to approve."""
    out = supervisor.supervise({"research_attempts": roles.MAX_RESEARCH_ATTEMPTS,
                                "confidence": 0.05, "hits": [{"a": 1}],
                                "needs_review": True})
    assert out["needs_review"] is False


# --- the auditor -----------------------------------------------------------

def _hit(text):
    return {"chunk_id": "c1", "doc_id": "d1", "score": 1.0, "text": text,
            "citation": "29 CFR 1910.1", "heading": "H", "char_start": 0,
            "char_end": len(text), "source_url": ""}


def test_auditor_passes_a_quote_it_can_find():
    text = "Employers shall provide eyewash facilities for immediate emergency use."
    out = roles.auditor({"status": "drafted", "hits": [_hit(text)],
                         "citations": [{"source": 1,
                                        "quote": "provide eyewash facilities"}]})
    assert out["fault"] == ""
    assert len(out["citations"]) == 1


def test_auditor_blames_the_writer_for_a_quote_that_is_not_there():
    out = roles.auditor({"status": "drafted", "hits": [_hit("Some other rule text.")],
                         "citations": [{"source": 1, "quote": "invented language"}]})
    assert out["fault"] == "writer"
    assert "invented language" in out["audit_note"]


def test_auditor_blames_the_researcher_when_nothing_was_cited():
    out = roles.auditor({"status": "drafted", "hits": [_hit("Some rule text.")],
                         "citations": []})
    assert out["fault"] == "researcher"


def test_auditor_stands_down_on_an_outage():
    assert roles.auditor({"status": "generation_failed", "hits": [], "citations": []}) == {}


# --- end to end ------------------------------------------------------------

def _fake_roles(monkeypatch, audit):
    """Fake specialists so the loops can be driven without a model."""
    calls = []

    def researcher(state):
        n = int(state.get("research_attempts") or 0) + 1
        calls.append("researcher")
        out = {"research_attempts": n, "hits": [_hit("text")], "confidence": CONFIDENT,
               "search_query": "q{}".format(n), "fault": "", "audit_note": ""}
        if n > 1:
            out.update({"answer": "", "citations": []})
        return out

    def writer(state):
        calls.append("writer")
        return {"write_attempts": int(state.get("write_attempts") or 0) + 1,
                "drafted_from": int(state.get("research_attempts") or 0),
                "answer": "draft", "citations": [{"source": 1, "quote": "q"}],
                "status": "drafted"}

    def auditor(state):
        calls.append("auditor")
        return {**audit, "status": "audited"}

    monkeypatch.setattr(roles, "researcher", researcher)
    monkeypatch.setattr(roles, "writer", writer)
    monkeypatch.setattr(roles, "auditor", auditor)
    return calls


def _run(monkeypatch, audit):
    calls = _fake_roles(monkeypatch, audit)
    graph = build_agent_graph(with_review=False)
    out = graph.invoke({"query": "how long can waste sit on site", "strategy": "structured"})
    return out, calls


def test_happy_path_runs_each_specialist_once(monkeypatch):
    out, calls = _run(monkeypatch, {"fault": "", "citations": [{"source": 1}]})
    assert calls == ["researcher", "writer", "auditor"]
    assert out["status"] == "answered"
    assert out["route_history"] == [R, W, A, D]


def test_a_writer_fault_loops_back_to_the_writer_and_terminates(monkeypatch):
    out, calls = _run(monkeypatch, {"fault": "writer", "citations": [],
                                    "audit_note": "bad quote"})
    assert calls.count("writer") == roles.MAX_WRITE_ATTEMPTS
    assert calls.count("researcher") == 1, "the sources were never the problem"
    assert out["status"] == "unverified"


def test_a_researcher_fault_re_searches_then_redrafts(monkeypatch):
    out, calls = _run(monkeypatch, {"fault": "researcher", "citations": [],
                                    "audit_note": "sources do not support it"})
    assert calls.count("researcher") == roles.MAX_RESEARCH_ATTEMPTS
    assert "writer" in calls[calls.index("researcher", 1):], "never redrafted on new sources"
    assert out["status"] == "unverified"
    assert out["citations"] == []


# --- structure -------------------------------------------------------------

def test_graph_compiles_with_and_without_review():
    assert build_agent_graph(with_review=True) is not None
    assert build_agent_graph(with_review=False, checkpointer=None) is not None


def test_specialists_report_back_rather_than_calling_each_other():
    """The topology is the claim. Every specialist's only outgoing edge is to
    the supervisor - that is what makes routing a decision, not an edge."""
    m = build_agent_graph(with_review=True).get_graph().draw_mermaid()
    for node in ("researcher", "writer", "auditor"):
        assert "{} --> supervisor".format(node) in m, node
    for a, b in [("researcher", "writer"), ("writer", "auditor"),
                 ("auditor", "writer"), ("auditor", "researcher")]:
        assert "{} --> {}".format(a, b) not in m, "{}->{} bypasses the supervisor".format(a, b)
    assert "review" in m, "HITL gate missing"


# --- reformulation: measured, guarded, and off -----------------------------
#
# scripts/measure_reformulation.py sampled 36 rewrites over the judged set:
# 0 rescued, 0 improved, and 4 that talked an out-of-scope question above the
# abstention threshold. These tests pin down both the guard and the default.

def test_reformulation_is_off_by_default():
    """The default is the finding. If this flips, the 11% harmful rate is back."""
    import importlib

    from cfr import config as cfg
    importlib.reload(cfg)
    assert cfg.ENABLE_REFORMULATION is False


def test_no_second_research_pass_when_reformulation_is_disabled(monkeypatch):
    """Re-running the identical query returns identical results. Spending an
    attempt on that is latency for nothing."""
    monkeypatch.setattr("cfr.config.ENABLE_REFORMULATION", False)
    out = supervisor.supervise({"research_attempts": 1, "confidence": 0.05,
                                "hits": [{"a": 1}]})
    assert out["next_agent"] == D
    assert out["status"] == "abstained"


def test_disabled_reformulation_returns_nothing(monkeypatch):
    monkeypatch.setattr("cfr.config.ENABLE_REFORMULATION", False)
    monkeypatch.setattr(roles, "_reformulate", lambda *a: "some rewrite")
    assert roles.reformulate("original question", []) == ""


def test_guard_rejects_a_rewrite_that_changed_the_question(monkeypatch):
    """The measured failure: "reverse a linked list" -> "40 CFR", scoring 0.80
    against a corpus that cannot answer it."""
    monkeypatch.setattr("cfr.config.ENABLE_REFORMULATION", True)
    monkeypatch.setattr("cfr.config.REFORMULATION_MIN_SIMILARITY", 0.70)
    monkeypatch.setattr(roles, "_reformulate", lambda *a: "40 CFR")
    monkeypatch.setattr(roles, "faithfulness", lambda a, b: 0.47)
    assert roles.reformulate("Write a Python function that reverses a linked list", []) == ""


def test_guard_accepts_a_faithful_rewrite(monkeypatch):
    monkeypatch.setattr("cfr.config.ENABLE_REFORMULATION", True)
    monkeypatch.setattr("cfr.config.REFORMULATION_MIN_SIMILARITY", 0.70)
    monkeypatch.setattr(roles, "_reformulate",
                        lambda *a: "hazardous waste accumulation time limits")
    monkeypatch.setattr(roles, "faithfulness", lambda a, b: 0.88)
    assert roles.reformulate("how long can waste sit on site", []) == \
        "hazardous waste accumulation time limits"


def test_guard_rejects_an_echo(monkeypatch):
    monkeypatch.setattr("cfr.config.ENABLE_REFORMULATION", True)
    monkeypatch.setattr(roles, "_reformulate", lambda q, t: q)
    assert roles.reformulate("same question", []) == ""


def test_research_budget_collapses_to_one_when_disabled(monkeypatch):
    monkeypatch.setattr("cfr.config.ENABLE_REFORMULATION", False)
    assert roles.max_research_attempts() == 1
    monkeypatch.setattr("cfr.config.ENABLE_REFORMULATION", True)
    assert roles.max_research_attempts() == roles.MAX_RESEARCH_ATTEMPTS
