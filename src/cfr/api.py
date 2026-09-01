"""HTTP API and static hosting.

Deliberate shape: /api/search never costs anything and never rate-limits hard,
because retrieval is the part of this system that is free to run. /api/ask is
where the budget and the tighter limit live.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict, Optional

from fastapi import Body, FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from . import answer as answer_mod
from . import chunk as chunk_mod
from . import __version__, config, db, embed
from .ratelimit import TokenBucket
from .search import RetrievalConfig, Retriever

_conn: Optional[sqlite3.Connection] = None
_write_lock = threading.Lock()
_search_bucket = TokenBucket(capacity=60, refill_per_sec=1.0)
_ask_bucket = TokenBucket(capacity=8, refill_per_sec=0.05)  # ~3/min sustained


def conn() -> sqlite3.Connection:
    global _conn
    if _conn is None:
        _conn = db.connect()
        db.init(_conn)
    return _conn


def _retry_after(wait: float) -> float:
    """json.dumps turns inf into `Infinity`, which JSON.parse rejects.

    A non-refilling bucket has no finite retry-after, so report a long but
    representable one rather than emitting invalid JSON to the browser.
    """
    if wait == float("inf") or wait != wait:
        return 86400.0
    return round(wait, 1)


def _client_key(request: Request) -> str:
    fwd = request.headers.get("x-forwarded-for")
    if fwd:
        return fwd.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def _warm() -> None:
    c = conn()
    stats = db.stats(c)
    if not stats["documents"]:
        print("! database is empty - run `cfr build` first")
        return
    # Pay the model load and the vector-matrix load now, not on the first
    # user request.
    try:
        from .search import dense, rerank

        embed.embed_query("warmup")
        rerank.warm()
        for s in ("structured", "contextual"):
            dense.load(c, s)
        print("models warm; {} documents indexed".format(stats["documents"]))
    except Exception as exc:  # noqa: BLE001
        print("! warmup failed ({}); first request will be slow".format(exc))


@asynccontextmanager
async def lifespan(_app: FastAPI):
    _warm()
    yield
    global _conn
    if _conn is not None:
        _conn.close()
        _conn = None


app = FastAPI(title="CFR Retrieval", version=__version__,
              docs_url="/api/docs", lifespan=lifespan)


def _retriever(strategy: str = "structured") -> Retriever:
    if strategy not in chunk_mod.STRATEGIES:
        raise HTTPException(400, "unknown strategy")
    return Retriever(conn(), RetrievalConfig(strategy=strategy))


# ---------------------------------------------------------------------------
# retrieval
# ---------------------------------------------------------------------------

@app.post("/api/search")
def api_search(request: Request, payload: Dict = Body(...)) -> JSONResponse:
    query = (payload.get("query") or "").strip()
    if not query:
        raise HTTPException(400, "query is required")
    if len(query) > 500:
        raise HTTPException(413, "query must be 500 characters or fewer")

    ok, wait = _search_bucket.allow(_client_key(request))
    if not ok:
        return JSONResponse({"error": "rate_limited", "retry_after_s": _retry_after(wait)}, 429)

    cfg = RetrievalConfig(
        strategy=payload.get("strategy", "structured"),
        use_lexical=payload.get("lexical", True),
        use_dense=payload.get("dense", True),
        use_rerank=payload.get("rerank", True),
        final_k=min(int(payload.get("k", 8)), 20),
    )
    result = Retriever(conn(), cfg).search(query)
    abstain, reason, confidence = answer_mod.should_abstain(result)
    return JSONResponse({
        "query": query,
        "config": cfg.describe(),
        "candidate_count": result.candidate_count,
        "timings_ms": {k: round(v, 1) for k, v in result.timings_ms.items()},
        "confidence": round(confidence, 4),
        "threshold": config.ABSTAIN_THRESHOLD,
        "abstain": abstain,
        "abstain_reason": reason,
        "hits": [h.to_dict() for h in result.hits],
    })


@app.post("/api/ask")
def api_ask(request: Request, payload: Dict = Body(...)) -> JSONResponse:
    query = (payload.get("query") or "").strip()
    if not query:
        raise HTTPException(400, "query is required")
    if len(query) > 500:
        raise HTTPException(413, "query must be 500 characters or fewer")

    ok, wait = _ask_bucket.allow(_client_key(request))
    if not ok:
        return JSONResponse({"error": "rate_limited", "retry_after_s": _retry_after(wait)}, 429)

    qvec = embed.embed_query(query)
    with _write_lock:
        result = answer_mod.answer(conn(), _retriever(payload.get("strategy", "structured")),
                                  query, qvec=qvec)
    return JSONResponse(result)


@app.get("/api/document/{doc_id:path}")
def api_document(doc_id: str) -> JSONResponse:
    row = conn().execute("SELECT * FROM documents WHERE doc_id = ?", (doc_id,)).fetchone()
    if row is None:
        raise HTTPException(404, "no such document")
    return JSONResponse({
        "doc_id": row["doc_id"],
        "citation": row["citation"],
        "heading": row["heading"],
        "text": row["text"],
        "source_url": row["source_url"],
        "length": len(row["text"]),
    })


@app.get("/api/stats")
def api_stats() -> JSONResponse:
    s = db.stats(conn())
    s["budget"] = answer_mod.budget_state(conn())
    s["generation_enabled"] = answer_mod._provider() is not None
    s["abstain_threshold"] = config.ABSTAIN_THRESHOLD
    s["embed_model"] = config.EMBED_MODEL
    s["rerank_model"] = config.RERANK_MODEL
    return JSONResponse(s)


@app.get("/api/health")
def api_health() -> JSONResponse:
    try:
        n = conn().execute("SELECT COUNT(*) FROM documents").fetchone()[0]
        return JSONResponse({"ok": True, "documents": int(n)})
    except Exception as exc:  # noqa: BLE001
        return JSONResponse({"ok": False, "error": str(exc)}, 503)


# ---------------------------------------------------------------------------
# labelling
# ---------------------------------------------------------------------------

def _pool_path() -> Path:
    return config.EVAL_DIR / "pool.jsonl"


@app.get("/api/pool")
def api_pool(limit: int = 50, offset: int = 0) -> JSONResponse:
    path = _pool_path()
    if not path.exists():
        return JSONResponse({"items": [], "total": 0, "judged": 0,
                             "message": "run `cfr pool` first"})
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    judged = {
        (r["query_id"], r["doc_id"])
        for r in conn().execute("SELECT query_id, doc_id FROM qrels")
    }
    pending = [r for r in rows if (r["query_id"], r["doc_id"]) not in judged]
    return JSONResponse({
        "items": pending[offset: offset + limit],
        "total": len(rows),
        "judged": len(rows) - len(pending),
        "remaining": len(pending),
    })


@app.post("/api/judge")
def api_judge(payload: Dict = Body(...)) -> JSONResponse:
    query_id = payload.get("query_id")
    doc_id = payload.get("doc_id")
    grade = payload.get("grade")
    if not query_id or not doc_id or grade is None:
        raise HTTPException(400, "query_id, doc_id and grade are required")
    try:
        grade = int(grade)
    except (TypeError, ValueError):
        raise HTTPException(400, "grade must be an integer")
    if grade not in (0, 1, 2, 3):
        raise HTTPException(400, "grade must be 0, 1, 2 or 3")

    import datetime as dt

    with _write_lock:
        conn().execute(
            "INSERT OR REPLACE INTO qrels (query_id, doc_id, grade, judged_at, judged_by) "
            "VALUES (?,?,?,?,'human')",
            (query_id, doc_id, grade,
             dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")),
        )
        conn().commit()
    total = conn().execute("SELECT COUNT(*) FROM qrels").fetchone()[0]
    return JSONResponse({"ok": True, "judgements": int(total)})


@app.get("/api/eval")
def api_eval() -> JSONResponse:
    """Serve the last ablation run, so the site can publish its own numbers."""
    path = config.EVAL_DIR / "ablation.json"
    if not path.exists():
        return JSONResponse({"available": False,
                             "message": "run `cfr eval --json-out evaldata/ablation.json`"})
    return JSONResponse({"available": True, "rows": json.loads(path.read_text(encoding="utf-8"))})


# ---------------------------------------------------------------------------
# static
# ---------------------------------------------------------------------------

def _asset_version() -> str:
    """Cache key derived from the asset files themselves.

    StaticFiles serves css/js with a far-future cache, so without this a deploy
    leaves every returning visitor on the previous JavaScript against the new
    API - which fails silently and looks like the feature simply not working.
    """
    stamp = 0.0
    for name in ("app.js", "style.css"):
        f = config.WEB_DIR / name
        if f.exists():
            stamp = max(stamp, f.stat().st_mtime)
    return "{}-{}".format(__version__, int(stamp))


def _page(name: str) -> HTMLResponse:
    html = (config.WEB_DIR / name).read_text(encoding="utf-8")
    return HTMLResponse(
        html.replace("{{V}}", _asset_version()),
        # The shell must never be cached, or it pins the asset version with it.
        headers={"Cache-Control": "no-store, must-revalidate"},
    )


@app.get("/")
def index() -> HTMLResponse:
    return _page("index.html")


@app.get("/label")
def label() -> HTMLResponse:
    return _page("label.html")


if config.WEB_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(config.WEB_DIR)), name="static")
