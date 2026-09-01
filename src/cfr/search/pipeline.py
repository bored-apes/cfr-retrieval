"""The retrieval pipeline: a funnel that runs wide, then narrow.

Stage one (lexical + dense + fusion) optimises recall - anything it misses is
unrecoverable downstream, so it retrieves generously. Stage two (cross-encoder)
optimises precision over that shortlist.

Every knob is on RetrievalConfig rather than hard-coded, because the ablation
in cfr.eval.run works by instantiating this class with different configs. If a
setting cannot be varied here, it cannot be measured.
"""

from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .. import config
from . import dense, fusion, lexical, rerank as rerank_mod


@dataclass
class RetrievalConfig:
    name: str = "hybrid+rerank"
    strategy: str = "structured"
    use_lexical: bool = True
    use_dense: bool = True
    use_rerank: bool = True
    candidates: int = config.CANDIDATES_PER_RETRIEVER
    rerank_top_n: int = config.RERANK_TOP_N
    final_k: int = config.FINAL_TOP_K
    rrf_k: int = config.RRF_K

    def describe(self) -> str:
        bits = []
        if self.use_lexical:
            bits.append("bm25")
        if self.use_dense:
            bits.append("dense")
        joined = "+".join(bits) or "none"
        if self.use_lexical and self.use_dense:
            joined = "hybrid(rrf)"
        return "{} / {} / rerank={}".format(self.strategy, joined, self.use_rerank)


@dataclass
class Hit:
    chunk_id: str
    doc_id: str
    score: float
    text: str
    heading_path: str
    citation: str
    heading: str
    char_start: int
    char_end: int
    source_url: str = ""
    rerank_score: Optional[float] = None   # temperature-scaled, (0, 1)
    rerank_logit: Optional[float] = None   # raw model output, for debugging
    lexical_rank: Optional[int] = None
    dense_rank: Optional[int] = None

    def to_dict(self) -> Dict:
        return {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "score": round(self.score, 6),
            "rerank_score": None if self.rerank_score is None else round(self.rerank_score, 6),
            "rerank_logit": None if self.rerank_logit is None else round(self.rerank_logit, 4),
            "text": self.text,
            "heading": self.heading,
            "heading_path": self.heading_path,
            "citation": self.citation,
            "char_start": self.char_start,
            "char_end": self.char_end,
            "source_url": self.source_url,
            "lexical_rank": self.lexical_rank,
            "dense_rank": self.dense_rank,
        }


def doc_id_of(chunk_id: str) -> str:
    """Chunk ids are "{doc_id}#{strategy}:{ordinal}" - recover the document."""
    return chunk_id.rsplit("#", 1)[0]


@dataclass
class SearchResult:
    query: str
    hits: List[Hit]
    timings_ms: Dict[str, float] = field(default_factory=dict)
    candidate_count: int = 0
    config_name: str = ""
    # Fused candidate chunk ids, before reranking. Recall is a property of the
    # recall stage, so it has to be measured here rather than on the shortlist.
    candidates: List[str] = field(default_factory=list)

    @property
    def top_score(self) -> float:
        if not self.hits:
            return 0.0
        h = self.hits[0]
        return h.rerank_score if h.rerank_score is not None else h.score


class Retriever:
    def __init__(self, conn: sqlite3.Connection, cfg: Optional[RetrievalConfig] = None):
        self.conn = conn
        self.cfg = cfg or RetrievalConfig()

    # -- internals ----------------------------------------------------------

    def _hydrate(self, chunk_ids: Sequence[str]) -> Dict[str, sqlite3.Row]:
        if not chunk_ids:
            return {}
        marks = ",".join("?" * len(chunk_ids))
        rows = self.conn.execute(
            """SELECT c.chunk_id, c.doc_id, c.text, c.heading_path, c.char_start, c.char_end,
                      d.citation, d.heading, d.source_url
               FROM chunks c JOIN documents d USING (doc_id)
               WHERE c.chunk_id IN ({})""".format(marks),
            list(chunk_ids),
        ).fetchall()
        return {r["chunk_id"]: r for r in rows}

    # -- public -------------------------------------------------------------

    def search(self, query: str, qvec: Optional[np.ndarray] = None) -> SearchResult:
        cfg = self.cfg
        timings: Dict[str, float] = {}
        rankings: List[List[Tuple[str, float]]] = []
        lex_rank: Dict[str, int] = {}
        den_rank: Dict[str, int] = {}

        if cfg.use_lexical:
            t0 = time.perf_counter()
            lex = lexical.search(self.conn, query, cfg.strategy, cfg.candidates)
            timings["lexical"] = (time.perf_counter() - t0) * 1000
            lex_rank = {cid: i + 1 for i, (cid, _) in enumerate(lex)}
            rankings.append(lex)

        if cfg.use_dense:
            t0 = time.perf_counter()
            den = dense.search(self.conn, query, cfg.strategy, cfg.candidates, qvec=qvec)
            timings["dense"] = (time.perf_counter() - t0) * 1000
            den_rank = {cid: i + 1 for i, (cid, _) in enumerate(den)}
            rankings.append(den)

        if not rankings:
            raise ValueError("RetrievalConfig disables every retriever")

        if len(rankings) == 1:
            fused = list(rankings[0])
        else:
            t0 = time.perf_counter()
            fused = fusion.rrf(rankings, k=cfg.rrf_k)
            timings["fusion"] = (time.perf_counter() - t0) * 1000

        candidate_count = len(fused)
        shortlist = fused[: cfg.rerank_top_n] if cfg.use_rerank else fused[: cfg.final_k]
        meta = self._hydrate([cid for cid, _ in shortlist])

        hits: List[Hit] = []
        for chunk_id, score in shortlist:
            row = meta.get(chunk_id)
            if row is None:
                continue
            hits.append(
                Hit(
                    chunk_id=chunk_id,
                    doc_id=row["doc_id"],
                    score=float(score),
                    text=row["text"],
                    heading_path=row["heading_path"],
                    citation=row["citation"],
                    heading=row["heading"],
                    char_start=int(row["char_start"]),
                    char_end=int(row["char_end"]),
                    source_url=row["source_url"] or "",
                    lexical_rank=lex_rank.get(chunk_id),
                    dense_rank=den_rank.get(chunk_id),
                )
            )

        if cfg.use_rerank and hits:
            t0 = time.perf_counter()
            logits = rerank_mod.rerank_logits(query, [h.text for h in hits])
            timings["rerank"] = (time.perf_counter() - t0) * 1000
            for hit, logit in zip(hits, logits):
                hit.rerank_logit = logit
                hit.rerank_score = rerank_mod.to_confidence(logit)
            hits.sort(key=lambda h: -(h.rerank_score or 0.0))
            hits = hits[: cfg.final_k]

        timings["total"] = sum(v for k, v in timings.items() if k != "total")
        return SearchResult(
            query=query,
            hits=hits,
            timings_ms=timings,
            candidate_count=candidate_count,
            config_name=cfg.name,
            candidates=[cid for cid, _ in fused],
        )
