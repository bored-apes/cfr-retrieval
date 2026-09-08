"""Graph nodes.

Each node is a pure-ish function of state. They reuse the same retrieval,
reranking and verification code as the hand-rolled pipeline in `cfr.answer` —
this is a port of the control flow, not a reimplementation of the logic, so any
measured difference between the two is attributable to the orchestration rather
than to different maths.
"""

from __future__ import annotations

import json
import time
from typing import Any, Dict, List

import numpy as np

from .. import answer as answer_mod
from .. import config, db, embed
from ..search import dense, fusion, lexical
from ..search import rerank as rerank_mod
from ..search.pipeline import doc_id_of
from .state import CFRState

MAX_GENERATE_ATTEMPTS = 2
# Answers whose confidence lands in this band above the abstention threshold are
# routed to a human before release. Comfortably-confident answers ship straight
# through; the band is where a reviewer's time is actually worth spending.
REVIEW_BAND = 0.15


def _conn():
    conn = db.connect()
    db.init(conn)
    return conn


# --------------------------------------------------------------------------
# 1. retrieve
# --------------------------------------------------------------------------

def retrieve(state: CFRState) -> Dict[str, Any]:
    """Lexical + dense in parallel, fused by RRF. Optimises recall, not precision."""
    conn = _conn()
    query = state["query"]
    strategy = state.get("strategy") or "structured"
    t: Dict[str, float] = {}

    t0 = time.perf_counter()
    lex = lexical.search(conn, query, strategy, config.CANDIDATES_PER_RETRIEVER)
    t["lexical"] = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    qvec = embed.embed_query(query)
    den = dense.search(conn, query, strategy, config.CANDIDATES_PER_RETRIEVER, qvec=qvec)
    t["dense"] = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    fused = fusion.rrf([lex, den], k=config.RRF_K)
    t["fusion"] = (time.perf_counter() - t0) * 1000

    lex_rank = {cid: i + 1 for i, (cid, _) in enumerate(lex)}
    den_rank = {cid: i + 1 for i, (cid, _) in enumerate(den)}

    shortlist = fused[: config.RERANK_TOP_N]
    marks = ",".join("?" * len(shortlist))
    rows = conn.execute(
        """SELECT c.chunk_id, c.doc_id, c.text, c.heading_path, c.char_start, c.char_end,
                  d.citation, d.heading, d.source_url
           FROM chunks c JOIN documents d USING (doc_id)
           WHERE c.chunk_id IN ({})""".format(marks),
        [cid for cid, _ in shortlist],
    ).fetchall()
    meta = {r["chunk_id"]: r for r in rows}

    hits: List[Dict[str, Any]] = []
    for chunk_id, score in shortlist:
        r = meta.get(chunk_id)
        if r is None:
            continue
        hits.append({
            "chunk_id": chunk_id, "doc_id": r["doc_id"], "text": r["text"],
            "heading": r["heading"], "citation": r["citation"],
            "char_start": int(r["char_start"]), "char_end": int(r["char_end"]),
            "source_url": r["source_url"] or "",
            "score": float(score), "rerank_score": None, "rerank_logit": None,
            "lexical_rank": lex_rank.get(chunk_id), "dense_rank": den_rank.get(chunk_id),
        })

    return {
        "hits": hits,
        "qvec": [float(x) for x in np.asarray(qvec).ravel()],
        "candidate_count": len(fused),
        "timings_ms": t,
        "generate_attempts": 0,
        "verify_feedback": "",
    }


# --------------------------------------------------------------------------
# 2. rerank
# --------------------------------------------------------------------------

def rerank(state: CFRState) -> Dict[str, Any]:
    """Cross-encoder over the shortlist.

    Its ranking contribution was measured as statistically indistinguishable
    from zero; it stays because it is the only stage producing a score with
    absolute meaning, which is what the gate below needs.
    """
    hits = state["hits"]
    if not hits:
        return {"confidence": 0.0}

    t0 = time.perf_counter()
    logits = rerank_mod.rerank_logits(state["query"], [h["text"] for h in hits])
    elapsed = (time.perf_counter() - t0) * 1000

    for h, logit in zip(hits, logits):
        h["rerank_logit"] = logit
        h["rerank_score"] = rerank_mod.to_confidence(logit)
    hits.sort(key=lambda h: -(h["rerank_score"] or 0.0))
    hits = hits[: config.FINAL_TOP_K]

    timings = dict(state.get("timings_ms") or {})
    timings["rerank"] = elapsed
    return {"hits": hits, "confidence": hits[0]["rerank_score"], "timings_ms": timings}


# --------------------------------------------------------------------------
# 3. gate  (conditional edge reads this)
# --------------------------------------------------------------------------

def gate(state: CFRState) -> Dict[str, Any]:
    """Decide whether the question is answerable from what was retrieved."""
    hits = state.get("hits") or []
    if not hits:
        return {"abstained": True, "abstain_reason": "no_results", "confidence": 0.0}

    conf = state.get("confidence") or 0.0
    if conf < config.ABSTAIN_THRESHOLD:
        return {"abstained": True, "abstain_reason": "low_confidence"}
    return {
        "abstained": False,
        "abstain_reason": "",
        # Marginal answers get a human before release; confident ones do not.
        "needs_review": conf < config.ABSTAIN_THRESHOLD + REVIEW_BAND,
    }


def route_after_gate(state: CFRState) -> str:
    return "abstain" if state.get("abstained") else "generate"


# --------------------------------------------------------------------------
# 4. abstain (terminal)
# --------------------------------------------------------------------------

def abstain(state: CFRState) -> Dict[str, Any]:
    reason = state.get("abstain_reason", "low_confidence")
    conf = state.get("confidence") or 0.0
    msg = {
        "no_results": "Nothing in the indexed parts of the CFR matched this question.",
        "low_confidence": (
            "The closest sections scored {:.2f}, below the {:.2f} confidence threshold, "
            "so this is left unanswered rather than guessed. The nearest matches are "
            "shown below."
        ).format(conf, config.ABSTAIN_THRESHOLD),
    }.get(reason, "Not answered.")
    return {"answer": msg, "citations": [], "status": "abstained"}


# --------------------------------------------------------------------------
# 5. generate
# --------------------------------------------------------------------------

def generate(state: CFRState) -> Dict[str, Any]:
    """Schema-constrained generation over the shortlist.

    On a retry, the prompt carries the quotes that failed verification so the
    model corrects rather than repeating itself.
    """
    from ..search.pipeline import Hit

    provider = answer_mod._provider()
    attempts = int(state.get("generate_attempts") or 0) + 1
    if provider is None:
        return {"status": "retrieval_only", "generate_attempts": attempts,
                "answer": "No generation provider configured; retrieval only.",
                "citations": []}

    hits = [Hit(chunk_id=h["chunk_id"], doc_id=h["doc_id"], score=h["score"],
                text=h["text"], heading_path="", citation=h["citation"],
                heading=h["heading"], char_start=h["char_start"],
                char_end=h["char_end"], source_url=h["source_url"])
            for h in state["hits"]]

    prompt = answer_mod.build_prompt(state["query"], hits)
    feedback = state.get("verify_feedback") or ""
    if feedback:
        prompt += (
            "\n\nIMPORTANT — your previous answer was rejected by verification:\n"
            + feedback
            + "\nEvery \"quote\" must appear VERBATIM in the source you cite. "
              "Copy the text exactly."
        )

    try:
        raw = (answer_mod._call_gemini(prompt) if provider == "gemini"
               else answer_mod._call_groq(prompt))
        parsed = json.loads(raw)
    except Exception as exc:  # noqa: BLE001
        return {"status": "generation_failed", "generate_attempts": attempts,
                "answer": "The answer service failed ({}).".format(type(exc).__name__),
                "citations": [], "error": str(exc)[:200]}

    return {
        "answer": parsed.get("answer", ""),
        "citations": parsed.get("citations", []),
        "generate_attempts": attempts,
        "status": "generated",
    }


# --------------------------------------------------------------------------
# 6. verify  (conditional edge reads this)
# --------------------------------------------------------------------------

def verify(state: CFRState) -> Dict[str, Any]:
    """Match every quoted span back into the section it claims to come from."""
    from ..search.pipeline import Hit

    if state.get("status") in ("generation_failed", "retrieval_only"):
        return {}

    hits = [Hit(chunk_id=h["chunk_id"], doc_id=h["doc_id"], score=h["score"],
                text=h["text"], heading_path="", citation=h["citation"],
                heading=h["heading"], char_start=h["char_start"],
                char_end=h["char_end"], source_url=h["source_url"])
            for h in state["hits"]]

    raw_cites = state.get("citations") or []
    verified, dropped = answer_mod.verify_citations(raw_cites, hits)

    feedback = ""
    if dropped and not verified:
        bad = [str(c.get("quote", ""))[:90] for c in raw_cites][:3]
        feedback = "These quotes were not found in the sources you cited: " + " | ".join(bad)

    return {"citations": verified, "citations_dropped": dropped,
            "verify_feedback": feedback}


def route_after_verify(state: CFRState) -> str:
    """Retry once if verification rejected everything; otherwise proceed.

    This is the loop the hand-rolled pipeline does not have — there, a fully
    unverifiable answer is simply served with its citations stripped.
    """
    if state.get("status") in ("generation_failed", "retrieval_only"):
        return "done"
    verified = state.get("citations") or []
    attempts = int(state.get("generate_attempts") or 0)
    if not verified and state.get("verify_feedback") and attempts < MAX_GENERATE_ATTEMPTS:
        return "retry"
    return "review" if state.get("needs_review") else "done"


# --------------------------------------------------------------------------
# 7. human review (interrupt)
# --------------------------------------------------------------------------

def review(state: CFRState) -> Dict[str, Any]:
    """Pause for a human when confidence is marginal.

    `interrupt` suspends the graph and persists state to the checkpointer; the
    caller resumes with Command(resume=...) once a person has decided.
    """
    from langgraph.types import interrupt

    decision = interrupt({
        "reason": "confidence in review band",
        "query": state["query"],
        "confidence": state.get("confidence"),
        "answer": state.get("answer", "")[:400],
        "citations": len(state.get("citations") or []),
        "options": ["approve", "reject"],
    })

    if isinstance(decision, dict):
        verdict = decision.get("decision", "approve")
        note = decision.get("note", "")
    else:
        verdict = str(decision or "approve")
        note = ""

    if verdict == "reject":
        return {"review_decision": "reject", "reviewer_note": note,
                "status": "rejected_by_reviewer",
                "answer": "A reviewer withheld this answer." + (" " + note if note else ""),
                "citations": []}
    return {"review_decision": "approve", "reviewer_note": note, "status": "answered"}


# --------------------------------------------------------------------------
# 8. finalise
# --------------------------------------------------------------------------

def finalise(state: CFRState) -> Dict[str, Any]:
    if state.get("status") in ("abstained", "generation_failed", "retrieval_only",
                               "rejected_by_reviewer"):
        return {}
    return {"status": "answered"}
