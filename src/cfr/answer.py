"""Generation, abstention, and citation verification.

Asking a model for "[3]" and rendering a footnote is theatre: the model emitted
a token, not a promise. So the model is also required to quote the supporting
sentence verbatim, and that quote is matched back into the chunk it claims to
come from. A match yields character offsets that resolve against the original
document and drive the highlight. A miss means the citation was invented, and
it is dropped and counted.
"""

from __future__ import annotations

import datetime as dt
import difflib
import hashlib
import json
import re
import sqlite3
import unicodedata
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import config
from .search import Hit, Retriever, SearchResult

_WS = re.compile(r"\s+")
_QUOTES = str.maketrans({
    "‘": "'", "’": "'", "“": '"', "”": '"',
    "–": "-", "—": "-", " ": " ", "§": "S",
})


# ---------------------------------------------------------------------------
# quote verification
# ---------------------------------------------------------------------------

def _normalize(text: str) -> Tuple[str, List[int]]:
    """Fold whitespace and smart punctuation, keeping a map back to the original.

    offset_map[i] is the index in `text` of normalized character i, so a match
    found in normalized space can be reported as real character offsets.
    """
    out_chars: List[str] = []
    offset_map: List[int] = []
    prev_space = True  # suppress leading whitespace
    for i, ch in enumerate(unicodedata.normalize("NFKC", text)):
        c = ch.translate(_QUOTES)
        if c.isspace():
            if prev_space:
                continue
            out_chars.append(" ")
            offset_map.append(i)
            prev_space = True
        else:
            out_chars.append(c.lower())
            offset_map.append(i)
            prev_space = False
    while out_chars and out_chars[-1] == " ":
        out_chars.pop()
        offset_map.pop()
    return "".join(out_chars), offset_map


def locate_quote(quote: str, haystack: str, threshold: float = None) -> Optional[Tuple[int, int, float]]:
    """Find `quote` inside `haystack`, returning (start, end, similarity).

    Models routinely alter a hyphen, collapse a line break, or restyle a quote
    mark. Requiring a byte-exact match would flag that as a hallucination, so an
    exact pass is followed by a fuzzy one.
    """
    threshold = config.QUOTE_MATCH_THRESHOLD if threshold is None else threshold
    if not quote or not quote.strip():
        return None

    n_quote, _ = _normalize(quote)
    n_hay, hay_map = _normalize(haystack)
    if len(n_quote) < 12 or not n_hay:
        return None

    pos = n_hay.find(n_quote)
    if pos != -1:
        return (hay_map[pos], hay_map[min(pos + len(n_quote) - 1, len(hay_map) - 1)] + 1, 1.0)

    # Fuzzy: slide a window the size of the quote and keep the best block match.
    matcher = difflib.SequenceMatcher(None, n_hay, n_quote, autojunk=False)
    block = matcher.find_longest_match(0, len(n_hay), 0, len(n_quote))
    if block.size < max(12, int(0.5 * len(n_quote))):
        return None

    start = max(0, block.a - block.b)
    end = min(len(n_hay), start + len(n_quote))
    ratio = difflib.SequenceMatcher(None, n_hay[start:end], n_quote, autojunk=False).ratio()
    if ratio < threshold:
        return None
    return (hay_map[start], hay_map[min(end - 1, len(hay_map) - 1)] + 1, ratio)


def verify_citations(
    citations: Sequence[Dict],
    hits: Sequence[Hit],
) -> Tuple[List[Dict], int]:
    """Resolve each cited quote to document offsets, dropping the ones that miss.

    Returns (verified, dropped_count). `dropped` is a real operational metric -
    publish it.
    """
    verified: List[Dict] = []
    dropped = 0
    for cite in citations:
        try:
            idx = int(cite.get("source", 0)) - 1
        except (TypeError, ValueError):
            dropped += 1
            continue
        if not (0 <= idx < len(hits)):
            dropped += 1
            continue

        hit = hits[idx]
        found = locate_quote(str(cite.get("quote", "")), hit.text)
        if found is None:
            dropped += 1
            continue

        rel_start, rel_end, ratio = found
        verified.append({
            "source": idx + 1,
            "chunk_id": hit.chunk_id,
            "doc_id": hit.doc_id,
            "citation": hit.citation,
            "heading": hit.heading,
            "source_url": hit.source_url,
            "quote": hit.text[rel_start:rel_end],
            # Offsets into documents.text, which is what the source view renders.
            "doc_char_start": hit.char_start + rel_start,
            "doc_char_end": hit.char_start + rel_end,
            "match_ratio": round(ratio, 4),
        })
    return verified, dropped


# ---------------------------------------------------------------------------
# abstention
# ---------------------------------------------------------------------------

def should_abstain(result: SearchResult) -> Tuple[bool, str, float]:
    """Three independent signals; any one of them stops the answer."""
    if not result.hits:
        return True, "no_results", 0.0

    # Only the cross-encoder produces a score with absolute meaning. RRF scores
    # are built from rank positions (1/(k+rank)), so the top hit sits near 0.032
    # whether it is a perfect match or pizza dough - thresholding on them makes
    # the system refuse every query. Measured on this corpus: in-scope 0.0320 vs
    # out-of-scope 0.0249, against 0.911 vs 0.062 with the reranker on.
    #
    # This is the reranker's real contribution here. Its ranking gain was inside
    # the noise (+0.016 nDCG@10, CI spans zero) but without it there is no
    # calibrated signal to abstain on at all.
    if result.hits[0].rerank_score is None:
        return False, "no_confidence_signal", result.hits[0].score

    scores = [h.rerank_score for h in result.hits]
    top = scores[0]

    if top < config.ABSTAIN_THRESHOLD:
        return True, "low_confidence", top

    # Ambiguity: several *different* sections tied together, and only in the
    # marginal band just above the threshold.
    #
    # An earlier version compared top/median as a ratio and fired constantly on
    # good results: temperature-scaled confidences compress relevant hits into a
    # narrow band near 0.9, so that ratio is always about 1.0. Eight chunks of
    # the one section that answers the question is the strongest possible
    # outcome, not an ambiguous one - so require both a marginal score and a
    # spread across distinct sections.
    if (config.AMBIGUITY_ENABLED and len(scores) >= 4
            and top < config.ABSTAIN_THRESHOLD + config.AMBIGUITY_BAND):
        shortlist = scores[:5]
        median = sorted(shortlist)[len(shortlist) // 2]
        distinct_docs = len({h.doc_id for h in result.hits[:5]})
        if (top - median) < config.AMBIGUITY_SPREAD and distinct_docs >= config.AMBIGUITY_MIN_DOCS:
            return True, "ambiguous", top

    return False, "", top


# ---------------------------------------------------------------------------
# generation
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You answer questions about US federal regulations using ONLY the numbered sources provided.

Rules:
- Use only the sources. If they do not contain the answer, set "sufficient" to false.
- Cite with [n] inline, matching the source numbers.
- For every [n] you use, add a citation entry whose "quote" is copied VERBATIM from that source. Do not paraphrase inside "quote".
- Quote the shortest span that supports the claim, at least 12 characters.
- Regulations are precise. Preserve numbers, deadlines and conditions exactly.
- Do not give legal advice or add requirements that are not in the sources.

Return ONLY JSON of this shape:
{"answer": "...", "sufficient": true, "citations": [{"source": 1, "quote": "..."}]}"""


def build_prompt(query: str, hits: Sequence[Hit]) -> str:
    blocks = []
    for i, hit in enumerate(hits, start=1):
        blocks.append("[{}] {} - {}\n{}".format(i, hit.citation, hit.heading, hit.text))
    return "Sources:\n\n{}\n\nQuestion: {}".format("\n\n---\n\n".join(blocks), query)


def _call_gemini(prompt: str, timeout: float = 45.0) -> str:
    import httpx

    url = ("https://generativelanguage.googleapis.com/v1beta/models/"
           "{}:generateContent".format(config.GEMINI_MODEL))
    payload = {
        "systemInstruction": {"parts": [{"text": SYSTEM_PROMPT}]},
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.0, "responseMimeType": "application/json"},
    }
    with httpx.Client(timeout=timeout) as client:
        r = client.post(url, json=payload, headers={"x-goog-api-key": config.GEMINI_API_KEY})
        r.raise_for_status()
        data = r.json()
    return data["candidates"][0]["content"]["parts"][0]["text"]


def _call_groq(prompt: str, timeout: float = 45.0) -> str:
    import httpx

    with httpx.Client(timeout=timeout) as client:
        r = client.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": "Bearer {}".format(config.GROQ_API_KEY)},
            json={
                "model": config.GROQ_MODEL,
                "temperature": 0.0,
                "response_format": {"type": "json_object"},
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
            },
        )
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]


def _provider() -> Optional[str]:
    if config.LLM_PROVIDER == "gemini" and config.GEMINI_API_KEY:
        return "gemini"
    if config.LLM_PROVIDER == "groq" and config.GROQ_API_KEY:
        return "groq"
    return None


# ---------------------------------------------------------------------------
# budget + cache
# ---------------------------------------------------------------------------

def _today() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d")


def budget_state(conn: sqlite3.Connection) -> Dict[str, int]:
    row = conn.execute("SELECT calls FROM budget WHERE day = ?", (_today(),)).fetchone()
    used = int(row["calls"]) if row else 0
    return {"used": used, "limit": config.DAILY_GENERATION_BUDGET,
            "remaining": max(0, config.DAILY_GENERATION_BUDGET - used)}


def _spend(conn: sqlite3.Connection) -> bool:
    """Reserve one generation call. False when today's ceiling is reached."""
    day = _today()
    conn.execute("INSERT OR IGNORE INTO budget (day, calls) VALUES (?, 0)", (day,))
    cur = conn.execute(
        "UPDATE budget SET calls = calls + 1 WHERE day = ? AND calls < ?",
        (day, config.DAILY_GENERATION_BUDGET),
    )
    conn.commit()
    return cur.rowcount > 0


def _cache_get(conn: sqlite3.Connection, query: str, qvec: Optional[np.ndarray]) -> Optional[Dict]:
    key = hashlib.sha256(query.strip().lower().encode("utf-8")).hexdigest()
    row = conn.execute("SELECT payload FROM answer_cache WHERE query_hash = ?", (key,)).fetchone()
    if row:
        payload = json.loads(row["payload"])
        payload["cached"] = "exact"
        return payload

    if qvec is None:
        return None
    # Semantic hit: public demos ask the same question in many phrasings, and
    # serving those from cache is what keeps the generation budget alive.
    rows = conn.execute("SELECT payload, qvec FROM answer_cache WHERE qvec IS NOT NULL").fetchall()
    if not rows:
        return None
    q = qvec / max(float(np.linalg.norm(qvec)), 1e-12)
    best, best_sim = None, 0.0
    for r in rows:
        v = np.frombuffer(r["qvec"], dtype=np.float32)
        if v.shape != q.shape:
            continue
        sim = float(v @ q / max(float(np.linalg.norm(v)), 1e-12))
        if sim > best_sim:
            best, best_sim = r, sim
    if best is not None and best_sim >= config.SEMANTIC_CACHE_THRESHOLD:
        payload = json.loads(best["payload"])
        payload["cached"] = "semantic"
        payload["cache_similarity"] = round(best_sim, 4)
        return payload
    return None


def _cache_put(conn: sqlite3.Connection, query: str, qvec: Optional[np.ndarray], payload: Dict) -> None:
    key = hashlib.sha256(query.strip().lower().encode("utf-8")).hexdigest()
    conn.execute(
        "INSERT OR REPLACE INTO answer_cache (query_hash, query, qvec, payload, created_at) VALUES (?,?,?,?,?)",
        (key, query, None if qvec is None else np.asarray(qvec, dtype=np.float32).tobytes(),
         json.dumps(payload), dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")),
    )
    conn.commit()


# ---------------------------------------------------------------------------
# the public entry point
# ---------------------------------------------------------------------------

def answer(
    conn: sqlite3.Connection,
    retriever: Retriever,
    query: str,
    qvec: Optional[np.ndarray] = None,
) -> Dict:
    result = retriever.search(query, qvec=qvec)
    hits = [h.to_dict() for h in result.hits]
    base = {
        "query": query,
        "hits": hits,
        "timings_ms": {k: round(v, 1) for k, v in result.timings_ms.items()},
        "candidate_count": result.candidate_count,
        "config": retriever.cfg.describe(),
        "budget": budget_state(conn),
        "cached": None,
    }

    abstain, reason, confidence = should_abstain(result)
    base["confidence"] = round(confidence, 4)
    base["threshold"] = config.ABSTAIN_THRESHOLD

    if abstain:
        base["status"] = "abstained"
        base["reason"] = reason
        base["answer"] = {
            "no_results": "Nothing in the indexed parts of the CFR matched this question.",
            "low_confidence": (
                "The closest sections scored {:.2f}, below the {:.2f} confidence "
                "threshold, so this is left unanswered rather than guessed. The "
                "nearest matches are shown below."
            ).format(confidence, config.ABSTAIN_THRESHOLD),
            "ambiguous": (
                "Several sections matched about equally well, which usually means "
                "the question is ambiguous. Narrowing it will give a better answer."
            ),
        }.get(reason, "Not answered.")
        base["citations"] = []
        return base

    cached = _cache_get(conn, query, qvec)
    if cached:
        cached["hits"] = hits
        cached["budget"] = budget_state(conn)
        return cached

    provider = _provider()
    if provider is None:
        base["status"] = "retrieval_only"
        base["answer"] = (
            "No generation provider is configured, so this is retrieval only. "
            "Set GEMINI_API_KEY or GROQ_API_KEY to enable written answers - the "
            "ranked sections below are unaffected."
        )
        base["citations"] = []
        return base

    if not _spend(conn):
        # The designed degraded state: retrieval is most of the value and keeps
        # working; only the written answer goes away.
        base["status"] = "budget_exhausted"
        base["answer"] = (
            "Today's free generation budget is used up. Ranked sections are still "
            "below - they are produced by the retrieval pipeline, which costs nothing "
            "to run. Written answers resume at 00:00 UTC."
        )
        base["citations"] = []
        return base

    try:
        raw = _call_gemini(build_prompt(query, result.hits)) if provider == "gemini" \
            else _call_groq(build_prompt(query, result.hits))
        parsed = json.loads(raw)
    except Exception as exc:  # noqa: BLE001 - any provider failure degrades the same way
        base["status"] = "generation_failed"
        base["answer"] = "The answer service failed ({}). Ranked sections are below.".format(
            type(exc).__name__)
        base["citations"] = []
        return base

    verified, dropped = verify_citations(parsed.get("citations", []), result.hits)

    base["status"] = "answered" if parsed.get("sufficient", True) else "insufficient"
    base["answer"] = parsed.get("answer", "")
    base["citations"] = verified
    base["citations_dropped"] = dropped
    if dropped:
        base["citation_warning"] = (
            "{} citation(s) quoted text that is not in the cited source and were "
            "removed.".format(dropped)
        )
    _cache_put(conn, query, qvec, base)
    return base
