"""The three specialists.

Each owns one concern and reports back to the supervisor rather than calling the
next stage itself. That is the difference between this and the linear graph in
`cfr.graph`: here the control flow is a decision, not an edge.

  Researcher  finds candidate sections; may rewrite the query and try again
  Writer      drafts a grounded answer under a schema
  Auditor     verifies every quote and diagnoses WHOSE fault a failure was
"""

from __future__ import annotations

import json
import time
from typing import Any, Dict, List

from .. import answer as answer_mod
from .. import config, db, embed
from ..search import dense, fusion, lexical
from ..search import rerank as rerank_mod
from .state import AgentState

MAX_RESEARCH_ATTEMPTS = 2
MAX_WRITE_ATTEMPTS = 2
REVIEW_BAND = 0.15


def _conn():
    conn = db.connect()
    db.init(conn)
    return conn


def _hit_objects(hits: List[Dict[str, Any]]):
    from ..search.pipeline import Hit

    return [Hit(chunk_id=h["chunk_id"], doc_id=h["doc_id"], score=h["score"],
                text=h["text"], heading_path="", citation=h["citation"],
                heading=h["heading"], char_start=h["char_start"],
                char_end=h["char_end"], source_url=h["source_url"])
            for h in hits]


# ==========================================================================
# Researcher
# ==========================================================================

REFORMULATE_PROMPT = """You rewrite plain-English questions into the vocabulary US federal regulations actually use.

Regulations say "accumulate", not "store"; "90 days", not "how long"; they name forms and section numbers. A previous search using the user's own wording returned nothing confident.

Rewrite the question using the terms the regulation itself would use. Keep it a search query, not a sentence. Return ONLY JSON: {"query": "..."}"""


def _reformulate(original: str, tried: List[str]) -> str:
    """Ask the model for regulatory vocabulary. Returns "" if unavailable.

    Unguarded: this is the raw model call, used by the measurement script.
    Production goes through `reformulate`, which will not accept a rewrite that
    has stopped meaning what the user asked.
    """
    provider = answer_mod._provider()
    if provider is None:
        return ""
    prompt = "User question: {}\nAlready tried: {}".format(original, tried or "nothing")
    try:
        raw = (answer_mod._call_gemini_with_system(REFORMULATE_PROMPT, prompt)
               if provider == "gemini"
               else answer_mod._call_groq_with_system(REFORMULATE_PROMPT, prompt))
        return str(json.loads(raw).get("query", "")).strip()
    except Exception:  # noqa: BLE001 - reformulation is best-effort
        return ""


def faithfulness(original: str, rewritten: str) -> float:
    """Cosine of the original against the rewrite, in the retriever's own space."""
    import numpy as np

    a = embed.embed_query(original).ravel()
    b = embed.embed_query(rewritten).ravel()
    return float(np.dot(a, b))


def reformulate(original: str, tried: List[str]) -> str:
    """Guarded reformulation. Returns "" when the rewrite must not be used.

    The guard exists because the unguarded version was measured defeating the
    abstention gate on 11% of draws - rewriting "reverse a linked list" into
    "40 CFR" and scoring 0.80 against a corpus that cannot answer it. A rewrite
    that no longer means what the user asked is not a better query, it is a
    different question.
    """
    if not config.ENABLE_REFORMULATION:
        return ""
    rewritten = _reformulate(original, tried)
    if not rewritten or rewritten == original:
        return ""
    if faithfulness(original, rewritten) < config.REFORMULATION_MIN_SIMILARITY:
        return ""
    return rewritten


def max_research_attempts() -> int:
    """With reformulation disabled, a second research pass runs the identical
    query and returns the identical results. One attempt is the whole budget."""
    return MAX_RESEARCH_ATTEMPTS if config.ENABLE_REFORMULATION else 1


def researcher(state: AgentState) -> Dict[str, Any]:
    """Retrieve and rerank; on a low-confidence second pass, rewrite the query.

    The vocabulary gap is the problem this whole system exists to close, so it
    is the one place where a model call earns its cost at retrieval time.
    """
    conn = _conn()
    attempts = int(state.get("research_attempts") or 0)
    variants = list(state.get("query_variants") or [])
    strategy = state.get("strategy") or "structured"

    if attempts == 0:
        search_query = state["query"]
    else:
        rewritten = reformulate(state["query"], variants)
        search_query = rewritten or state["query"]
    variants.append(search_query)

    t0 = time.perf_counter()
    lex = lexical.search(conn, search_query, strategy, config.CANDIDATES_PER_RETRIEVER)
    qvec = embed.embed_query(search_query)
    den = dense.search(conn, search_query, strategy, config.CANDIDATES_PER_RETRIEVER, qvec=qvec)
    fused = fusion.rrf([lex, den], k=config.RRF_K)

    lex_rank = {cid: i + 1 for i, (cid, _) in enumerate(lex)}
    den_rank = {cid: i + 1 for i, (cid, _) in enumerate(den)}
    shortlist = fused[: config.RERANK_TOP_N]

    marks = ",".join("?" * len(shortlist))
    rows = conn.execute(
        """SELECT c.chunk_id, c.doc_id, c.text, c.char_start, c.char_end,
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
            "source_url": r["source_url"] or "", "score": float(score),
            "rerank_score": None, "lexical_rank": lex_rank.get(chunk_id),
            "dense_rank": den_rank.get(chunk_id),
        })

    confidence = 0.0
    if hits:
        logits = rerank_mod.rerank_logits(search_query, [h["text"] for h in hits])
        for h, lg in zip(hits, logits):
            h["rerank_score"] = rerank_mod.to_confidence(lg)
        hits.sort(key=lambda h: -(h["rerank_score"] or 0.0))
        hits = hits[: config.FINAL_TOP_K]
        confidence = hits[0]["rerank_score"]

    timings = dict(state.get("timings_ms") or {})
    timings["research_{}".format(attempts + 1)] = (time.perf_counter() - t0) * 1000

    # Keep whichever attempt scored best rather than the most recent one - a
    # reformulation is a guess, and it is allowed to be a worse one. The query
    # travels with its results so the UI reports what actually produced them.
    if state.get("hits") and confidence < (state.get("confidence") or 0.0):
        hits = state["hits"]
        confidence = state["confidence"]
        search_query = state.get("search_query") or search_query

    out = {
        "hits": hits, "confidence": confidence, "candidate_count": len(fused),
        "search_query": search_query, "query_variants": variants,
        "research_attempts": attempts + 1, "timings_ms": timings,
        "fault": "", "audit_note": "",
    }
    if attempts:
        # Any existing draft was written against sources that no longer stand.
        # Leaving it in state is how a stale answer ships.
        out["answer"] = ""
        out["citations"] = []
    return out


# ==========================================================================
# Writer
# ==========================================================================

def writer(state: AgentState) -> Dict[str, Any]:
    """Draft a grounded answer. On a retry, the auditor's note is in the prompt."""
    provider = answer_mod._provider()
    attempts = int(state.get("write_attempts") or 0) + 1
    if provider is None:
        return {"status": "retrieval_only", "write_attempts": attempts,
                "answer": "No generation provider configured; retrieval only.",
                "citations": [], "drafted_from": int(state.get("research_attempts") or 0)}

    hits = _hit_objects(state["hits"])
    prompt = answer_mod.build_prompt(state["query"], hits)
    note = state.get("audit_note") or ""
    if note:
        prompt += ("\n\nIMPORTANT — an auditor rejected your previous answer:\n" + note
                   + "\nEvery \"quote\" must appear VERBATIM in the source you cite.")

    t0 = time.perf_counter()
    try:
        raw = (answer_mod._call_gemini(prompt) if provider == "gemini"
               else answer_mod._call_groq(prompt))
        parsed = json.loads(raw)
    except Exception as exc:  # noqa: BLE001
        return {"status": "generation_failed", "write_attempts": attempts,
                "answer": "The answer service failed ({}).".format(type(exc).__name__),
                "citations": [], "error": str(exc)[:200]}

    timings = dict(state.get("timings_ms") or {})
    timings["write_{}".format(attempts)] = (time.perf_counter() - t0) * 1000

    return {"answer": parsed.get("answer", ""),
            "citations": parsed.get("citations", []),
            "sufficient": parsed.get("sufficient", True),
            "write_attempts": attempts, "timings_ms": timings,
            # Ties the draft to the sources it was written from; the supervisor
            # refuses to ship one whose sources have since been replaced.
            "drafted_from": int(state.get("research_attempts") or 0),
            "status": "drafted"}


# ==========================================================================
# Auditor
# ==========================================================================

def auditor(state: AgentState) -> Dict[str, Any]:
    """Verify every quote, then attribute the failure.

    The attribution is the point. A quote that is not in its cited section is
    the writer's problem and a retry can fix it. Sources that genuinely do not
    answer the question are the researcher's problem, and re-prompting the
    writer would just produce a more confident fabrication.
    """
    if state.get("status") in ("generation_failed", "retrieval_only"):
        return {}

    hits = _hit_objects(state["hits"])
    raw_cites = state.get("citations") or []
    verified, dropped = answer_mod.verify_citations(raw_cites, hits)

    if verified:
        return {"citations": verified, "citations_dropped": dropped,
                "fault": "", "audit_note": "", "status": "audited"}

    if raw_cites:
        bad = [str(c.get("quote", ""))[:80] for c in raw_cites][:3]
        return {
            "citations": [], "citations_dropped": dropped, "fault": "writer",
            "audit_note": "These quotes are not present in the sections you cited: "
                          + " | ".join(bad),
            "status": "audited",
        }

    # No citations attempted at all - the writer had nothing to work with.
    return {"citations": [], "citations_dropped": dropped, "fault": "researcher",
            "audit_note": "The retrieved sections do not support an answer.",
            "status": "audited"}
