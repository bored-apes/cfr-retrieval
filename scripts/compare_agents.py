"""Linear graph vs supervisor-routed agents, on identical queries.

All three orchestrations - the hand-rolled pipeline, the linear graph and this
one - call the same retrieval, reranking and verification code. So a difference
in output is a difference in *control flow*, which is the only thing worth
paying an abstraction for.

Two questions this answers:

  1. What does the supervisor cost? Extra hops are free; extra model calls are
     not. The researcher's reformulation is a real LLM round-trip and it only
     fires when the first pass scored below threshold.
  2. Does it recover anything the linear graph refuses? p11 in the eval set
     (eyewash stations) is the documented case where no configuration retrieved
     § 1910.151 from the user's own wording. If reformulation is worth its
     latency, that is where it shows up.

Run: python scripts/compare_agents.py
"""

from __future__ import annotations

import statistics
import sys
import time
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cfr import db  # noqa: E402
from cfr.agents import build_agent_graph  # noqa: E402
from cfr.graph import build_graph  # noqa: E402

QUERIES = [
    # Answerable from the corpus, plain wording.
    "How long can a large quantity generator keep hazardous waste on site?",
    "What do I do if a drum of waste starts leaking?",
    "When can the FDA stop my clinical trial?",
    "What paperwork has to travel with a shipment of hazardous waste?",
    "What safety gear does an employer have to pay for?",
    # The documented retrieval failure: the corpus covers this, the user's
    # wording does not reach it.
    "Where do I need to put an emergency eye wash station?",
    # Out of scope. Both paths must refuse.
    "What is the best hydration ratio for pizza dough?",
]


def _fresh(conn):
    conn.execute("DELETE FROM answer_cache")
    conn.commit()


def _run(graph, q, conn):
    _fresh(conn)
    t0 = time.perf_counter()
    out = graph.invoke({"query": q, "strategy": "structured"},
                       config={"configurable": {"thread_id": str(uuid.uuid4())}})
    return out, (time.perf_counter() - t0) * 1000


def main() -> int:
    conn = db.connect()
    db.init(conn)
    linear = build_graph(with_review=False)
    agents = build_agent_graph(with_review=False)

    print("warming models...", flush=True)
    _run(linear, "warmup", conn)

    rows = []
    for q in QUERIES:
        lin, t_lin = _run(linear, q, conn)
        agt, t_agt = _run(agents, q, conn)

        rewrote = (agt.get("search_query") or q) != q
        rows.append({
            "q": q,
            "lin_ms": t_lin, "agt_ms": t_agt,
            "lin_status": lin.get("status"), "agt_status": agt.get("status"),
            "lin_cites": len(lin.get("citations") or []),
            "agt_cites": len(agt.get("citations") or []),
            "lin_conf": lin.get("confidence") or 0.0,
            "agt_conf": agt.get("confidence") or 0.0,
            "rewrote": rewrote,
            "rewritten_to": agt.get("search_query") if rewrote else "",
            "hops": len(agt.get("route_history") or []),
        })
        print("  {:<52} linear {:>6.0f}ms {:<10} agents {:>6.0f}ms {:<10} {} hops{}".format(
            q[:50], t_lin, rows[-1]["lin_status"] or "?",
            t_agt, rows[-1]["agt_status"] or "?", rows[-1]["hops"],
            "  [reformulated]" if rewrote else ""), flush=True)

    print("\n" + "=" * 78)
    med = statistics.median
    print("p50 latency        linear {:.0f} ms    agents {:.0f} ms    overhead {:+.0f} ms".format(
        med([r["lin_ms"] for r in rows]), med([r["agt_ms"] for r in rows]),
        med([r["agt_ms"] for r in rows]) - med([r["lin_ms"] for r in rows])))

    quiet = [r for r in rows if not r["rewrote"]]
    if quiet:
        print("  ...on queries where the supervisor added no model call: {:+.0f} ms".format(
            med([r["agt_ms"] for r in quiet]) - med([r["lin_ms"] for r in quiet])))

    agree = sum(r["lin_status"] == r["agt_status"] for r in rows)
    print("outcome agreement  {}/{}".format(agree, len(rows)))
    for r in rows:
        if r["lin_status"] != r["agt_status"]:
            print("  DIVERGED  {}\n            linear={} (conf {:.2f})  agents={} (conf {:.2f})".format(
                r["q"], r["lin_status"], r["lin_conf"], r["agt_status"], r["agt_conf"]))
            if r["rewrote"]:
                print('            reformulated to: "{}"'.format(r["rewritten_to"]))

    rw = [r for r in rows if r["rewrote"]]
    print("reformulations     {} of {} queries triggered a second research pass".format(
        len(rw), len(rows)))
    recovered = [r for r in rw if r["agt_status"] == "answered" and r["lin_status"] != "answered"]
    print("recovered          {} query(s) the linear graph refused".format(len(recovered)))
    for r in recovered:
        print('  + "{}"\n      -> "{}"  (conf {:.2f} vs {:.2f})'.format(
            r["q"], r["rewritten_to"], r["agt_conf"], r["lin_conf"]))

    print("citations          linear {} total, agents {} total".format(
        sum(r["lin_cites"] for r in rows), sum(r["agt_cites"] for r in rows)))
    print("=" * 78)
    print("\nThe supervisor is free when it routes straight through; the cost is the")
    print("researcher's reformulation, which is one extra LLM call and only fires")
    print("below the confidence threshold. Whether that trade is worth it is the")
    print("'recovered' line - not the hop count.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
