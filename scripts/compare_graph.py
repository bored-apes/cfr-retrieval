"""Hand-rolled pipeline vs LangGraph port, on identical queries.

Both paths call the same retrieval, reranking and verification code, so any
difference is orchestration overhead rather than different maths. The question
this answers is narrow and worth answering before adopting a framework: what
does the abstraction cost, and does it change the output?

Run: python scripts/compare_graph.py
"""

from __future__ import annotations

import statistics
import sys
import time
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cfr import answer as answer_mod  # noqa: E402
from cfr import db  # noqa: E402
from cfr.graph import build_graph  # noqa: E402
from cfr.search import RetrievalConfig, Retriever  # noqa: E402

QUERIES = [
    "How long can a large quantity generator keep hazardous waste on site?",
    "What do I do if a drum of waste starts leaking?",
    "When can the FDA stop my clinical trial?",
    "What paperwork has to travel with a shipment of hazardous waste?",
    "What safety gear does an employer have to pay for?",
    "What is the best hydration ratio for pizza dough?",
]


def main() -> int:
    conn = db.connect()
    db.init(conn)
    graph = build_graph(with_review=False)
    retriever = Retriever(conn, RetrievalConfig(strategy="structured"))

    print("warming models...", flush=True)
    conn.execute("DELETE FROM answer_cache")
    conn.commit()
    retriever.search("warmup")

    rows = []
    for q in QUERIES:
        conn.execute("DELETE FROM answer_cache")
        conn.commit()
        t0 = time.perf_counter()
        hand = answer_mod.answer(conn, retriever, q)
        t_hand = (time.perf_counter() - t0) * 1000

        conn.execute("DELETE FROM answer_cache")
        conn.commit()
        t0 = time.perf_counter()
        gout = graph.invoke({"query": q, "strategy": "structured"},
                            config={"configurable": {"thread_id": str(uuid.uuid4())}})
        t_graph = (time.perf_counter() - t0) * 1000

        same_status = (hand["status"] == "answered") == (gout.get("status") == "answered")
        same_top = (
            (hand["hits"][0]["doc_id"] if hand.get("hits") else None)
            == (gout["hits"][0]["doc_id"] if gout.get("hits") else None)
        )
        rows.append({
            "q": q, "hand_ms": t_hand, "graph_ms": t_graph,
            "hand_status": hand["status"], "graph_status": gout.get("status"),
            "hand_cites": len(hand.get("citations") or []),
            "graph_cites": len(gout.get("citations") or []),
            "attempts": gout.get("generate_attempts"),
            "same_status": same_status, "same_top": same_top,
        })
        print("  {:<50} hand {:>6.0f}ms   graph {:>6.0f}ms   top-hit match: {}".format(
            q[:48], t_hand, t_graph, "yes" if same_top else "NO"), flush=True)

    print("\n" + "=" * 74)
    hand_ms = sorted(r["hand_ms"] for r in rows)
    graph_ms = sorted(r["graph_ms"] for r in rows)
    med = lambda xs: statistics.median(xs)  # noqa: E731
    print("p50 latency      hand-rolled {:.0f} ms   graph {:.0f} ms   overhead {:+.0f} ms".format(
        med(hand_ms), med(graph_ms), med(graph_ms) - med(hand_ms)))
    print("top hit agrees   {}/{}".format(sum(r["same_top"] for r in rows), len(rows)))
    print("status agrees    {}/{}".format(sum(r["same_status"] for r in rows), len(rows)))
    print("citations        hand {} total, graph {} total".format(
        sum(r["hand_cites"] for r in rows), sum(r["graph_cites"] for r in rows)))
    retries = [r for r in rows if (r["attempts"] or 0) > 1]
    print("graph retries    {} of {} queries needed a second generation".format(
        len(retries), len(rows)))
    print("=" * 74)
    print("\nBoth paths share the retrieval, reranking and verification code, so the")
    print("latency delta is orchestration only. Anything else would be a bug.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
