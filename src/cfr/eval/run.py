"""The ablation runner: the thing this project exists to produce.

Every row is one architectural decision with a measured cost and benefit. The
table is deliberately shipped empty of numbers - they come from your own
judgements, and inventing plausible ones is the only thing that would actually
sink the project's credibility.
"""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from .. import config, db
from ..search import RetrievalConfig, Retriever, doc_id_of
from . import metrics as M

# Each config isolates exactly one change from the row above it.
ABLATION: List[RetrievalConfig] = [
    RetrievalConfig(name="bm25 only", strategy="structured",
                    use_lexical=True, use_dense=False, use_rerank=False, final_k=10),
    RetrievalConfig(name="dense only", strategy="structured",
                    use_lexical=False, use_dense=True, use_rerank=False, final_k=10),
    RetrievalConfig(name="hybrid (rrf)", strategy="structured",
                    use_lexical=True, use_dense=True, use_rerank=False, final_k=10),
    RetrievalConfig(name="hybrid + rerank", strategy="structured",
                    use_lexical=True, use_dense=True, use_rerank=True, final_k=10),
    RetrievalConfig(name="+ contextual chunks", strategy="contextual",
                    use_lexical=True, use_dense=True, use_rerank=True, final_k=10),
]

RECALL_K = 100
NDCG_K = 10


def load_queries(path: Optional[str] = None) -> List[Dict]:
    p = Path(path) if path else (config.EVAL_DIR / "queries.jsonl")
    if not p.exists():
        raise SystemExit("no query file at {}".format(p))
    rows = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def load_qrels(conn: sqlite3.Connection) -> Dict[str, Dict[str, int]]:
    out: Dict[str, Dict[str, int]] = {}
    for r in conn.execute("SELECT query_id, doc_id, grade FROM qrels"):
        out.setdefault(r["query_id"], {})[r["doc_id"]] = int(r["grade"])
    return out


def _query_vectors(queries: Sequence[Dict]) -> Dict[str, "object"]:
    """Embed every query once and reuse across configs.

    Without this the dense stage is re-encoded for each of five configs, which
    dominates wall-clock time and tells you nothing extra.
    """
    from .. import embed

    return {q["query_id"]: embed.embed_query(q["query"]) for q in queries}


def run_config(
    conn: sqlite3.Connection,
    cfg: RetrievalConfig,
    queries: Sequence[Dict],
    qvecs: Dict,
) -> Dict[str, Dict]:
    retriever = Retriever(conn, cfg)
    out: Dict[str, Dict] = {}
    for q in queries:
        t0 = time.perf_counter()
        res = retriever.search(q["query"], qvec=qvecs.get(q["query_id"]))
        elapsed = (time.perf_counter() - t0) * 1000
        out[q["query_id"]] = {
            "ranked": M.dedupe_docs([h.doc_id for h in res.hits]),
            "candidates": M.dedupe_docs([doc_id_of(c) for c in res.candidates]),
            "top_score": res.top_score,
            "latency_ms": elapsed,
            "timings": res.timings_ms,
        }
    return out


def score_config(
    results: Dict[str, Dict],
    qrels: Dict[str, Dict[str, int]],
    queries: Sequence[Dict],
) -> Dict:
    per_query: Dict[str, Dict[str, Optional[float]]] = {}
    for q in queries:
        qid = q["query_id"]
        rel = qrels.get(qid, {})
        r = results.get(qid)
        if r is None:
            continue
        per_query[qid] = {
            "recall": M.recall_at_k(r["candidates"], rel, RECALL_K),
            "ndcg": M.ndcg_at_k(r["ranked"], rel, NDCG_K),
            "mrr": M.mrr_at_k(r["ranked"], rel, NDCG_K),
        }

    lat = sorted(r["latency_ms"] for r in results.values())
    def pct(p: float) -> float:
        if not lat:
            return float("nan")
        return lat[min(len(lat) - 1, int(p * len(lat)))]

    judged = [qid for qid, m in per_query.items() if m["ndcg"] is not None]
    return {
        "per_query": per_query,
        "judged": len(judged),
        "p50_ms": pct(0.50),
        "p95_ms": pct(0.95),
    }


def _fmt(mean: float, lo: float, hi: float) -> str:
    if mean != mean:  # NaN
        return "     --     "
    return "{:.3f} ±{:.3f}".format(mean, max(hi - mean, mean - lo))


def main(args) -> int:
    conn = db.connect()
    db.init(conn)
    queries = load_queries(getattr(args, "queries", None))
    qrels = load_qrels(conn)

    # A query judged entirely 0 is out of scope, not answerable. Testing the
    # dict for truthiness counts those as answerable and inflates the reported
    # denominator, even though every retrieval metric correctly returns None.
    answerable = [
        q for q in queries
        if any(g > 0 for g in qrels.get(q["query_id"], {}).values())
    ]
    if not answerable:
        print("No relevance judgements found.\n")
        print("The ablation cannot score anything until you build the answer key:")
        print("  1. cfr pool          # gather candidates from every config")
        print("  2. cfr serve         # then open /label and judge them")
        print("  3. cfr eval          # come back here")
        print("\nThat labelling pass is the differentiator. It is meant to be your work,")
        print("not the tool's - a generated answer key only measures the generator.")
        return 1

    wanted = getattr(args, "configs", None)
    configs = [c for c in ABLATION if not wanted or c.name in wanted]

    print("Embedding {} queries once for reuse...".format(len(queries)))
    qvecs = _query_vectors(queries)

    rows = []
    last_results: Dict[str, Dict] = {}
    baseline_ndcg: Optional[List[Optional[float]]] = None
    prev_ndcg: Optional[List[Optional[float]]] = None
    prev_p50 = 0.0
    order = [q["query_id"] for q in queries]

    for cfg in configs:
        print("  running: {:<22} {}".format(cfg.name, cfg.describe()), flush=True)
        results = run_config(conn, cfg, queries, qvecs)
        scored = score_config(results, qrels, queries)
        pq = scored["per_query"]

        recalls = [pq[q].get("recall") for q in order if q in pq]
        ndcgs = [pq[q].get("ndcg") for q in order if q in pq]
        mrrs = [pq[q].get("mrr") for q in order if q in pq]

        iters = getattr(args, "bootstrap", 1000)
        row = {
            "name": cfg.name,
            "describe": cfg.describe(),
            "judged": scored["judged"],
            "recall": M.bootstrap_ci([v for v in recalls if v is not None], iters),
            "ndcg": M.bootstrap_ci([v for v in ndcgs if v is not None], iters),
            "mrr": M.bootstrap_ci([v for v in mrrs if v is not None], iters),
            "p50_ms": scored["p50_ms"],
            "p95_ms": scored["p95_ms"],
            "per_query": pq,
        }
        last_results = results
        if baseline_ndcg is None:
            baseline_ndcg = ndcgs
            row["delta"] = None
        else:
            row["delta"] = M.paired_bootstrap(baseline_ndcg, ndcgs, iters)
        # Each config isolates one change from the row above it, so the
        # incremental comparison is the one that answers "did this stage earn
        # its cost". Comparing everything to the first row hides a stage that
        # added latency and nothing else.
        if prev_ndcg is not None:
            row["delta_prev"] = M.paired_bootstrap(prev_ndcg, ndcgs, iters)
            row["p50_delta"] = row["p50_ms"] - prev_p50
        else:
            row["delta_prev"] = None
            row["p50_delta"] = None
        prev_ndcg, prev_p50 = ndcgs, row["p50_ms"]
        rows.append(row)

    _print_table(rows, len(answerable), len(queries))

    if getattr(args, "show_failures", 0):
        _print_failures(conn, rows[-1], last_results, qrels, queries, args.show_failures)

    if getattr(args, "json_out", None):
        Path(args.json_out).write_text(json.dumps(rows, indent=2, default=list), encoding="utf-8")
        print("\nwrote {}".format(args.json_out))
    if getattr(args, "markdown_out", None):
        Path(args.markdown_out).write_text(_markdown(rows, len(answerable)), encoding="utf-8")
        print("wrote {}".format(args.markdown_out))
    return 0


def _print_table(rows: List[Dict], n_answerable: int, n_total: int) -> None:
    print("\n{} judged queries of {} total\n".format(n_answerable, n_total))
    head = "{:<22} {:^14} {:^14} {:^14} {:>8} {:>8}".format(
        "configuration", "recall@100", "nDCG@10", "MRR@10", "p50 ms", "p95 ms")
    print(head)
    print("-" * len(head))
    for r in rows:
        print("{:<22} {} {} {} {:>8.0f} {:>8.0f}".format(
            r["name"], _fmt(*r["recall"]), _fmt(*r["ndcg"]), _fmt(*r["mrr"]),
            r["p50_ms"], r["p95_ms"]))
    print("-" * len(head))
    print("± is a 95% percentile bootstrap CI over queries.\n")

    print("Paired comparison against '{}' (nDCG@10):".format(rows[0]["name"]))
    for r in rows[1:]:
        d = r["delta"]
        if not d or d["delta"] != d["delta"]:
            continue
        verdict = "significant" if M.significant(d["lo"], d["hi"]) else "NOT significant"
        print("  {:<22} {:+.3f}  [{:+.3f}, {:+.3f}]  {}".format(
            r["name"], d["delta"], d["lo"], d["hi"], verdict))
    print("\nIncremental - each row vs the row above (nDCG@10, and p50 cost):")
    for r in rows[1:]:
        d = r.get("delta_prev")
        if not d or d["delta"] != d["delta"]:
            continue
        verdict = "EARNS IT" if M.significant(d["lo"], d["hi"]) and d["delta"] > 0 else "no measurable gain"
        print("  {:<22} {:+.3f}  [{:+.3f}, {:+.3f}]  {:+7.0f} ms   {}".format(
            r["name"], d["delta"], d["lo"], d["hi"], r.get("p50_delta") or 0.0, verdict))

    if n_answerable < 100:
        print("\n  Note: with {} judged queries these intervals are wide. Differences whose".format(n_answerable))
        print("  interval spans zero are noise - report them as ties, not improvements.")


def _markdown(rows: List[Dict], n_answerable: int) -> str:
    out = ["# Retrieval ablation", "",
           "{} judged queries. Intervals are 95% percentile bootstrap over queries.".format(n_answerable),
           "",
           "| Configuration | Recall@100 | nDCG@10 | MRR@10 | p50 ms | p95 ms |",
           "|---|---|---|---|---|---|"]
    for r in rows:
        out.append("| {} | {} | {} | {} | {:.0f} | {:.0f} |".format(
            r["name"],
            _fmt(*r["recall"]).strip(), _fmt(*r["ndcg"]).strip(), _fmt(*r["mrr"]).strip(),
            r["p50_ms"], r["p95_ms"]))
    out += ["", "## Paired comparison vs `{}`".format(rows[0]["name"]), "",
            "| Configuration | ΔnDCG@10 | 95% CI | Verdict |", "|---|---|---|---|"]
    for r in rows[1:]:
        d = r["delta"]
        if not d or d["delta"] != d["delta"]:
            continue
        out.append("| {} | {:+.3f} | [{:+.3f}, {:+.3f}] | {} |".format(
            r["name"], d["delta"], d["lo"], d["hi"],
            "significant" if M.significant(d["lo"], d["hi"]) else "tie (CI spans 0)"))

    out += ["", "## Incremental - each row vs the row above", "",
            "The comparison that decides whether a stage is worth keeping.", "",
            "| Configuration | ΔnDCG@10 | 95% CI | Δp50 | Verdict |", "|---|---|---|---|---|"]
    for r in rows[1:]:
        d = r.get("delta_prev")
        if not d or d["delta"] != d["delta"]:
            continue
        out.append("| {} | {:+.3f} | [{:+.3f}, {:+.3f}] | {:+.0f} ms | {} |".format(
            r["name"], d["delta"], d["lo"], d["hi"], r.get("p50_delta") or 0.0,
            "earns it" if (M.significant(d["lo"], d["hi"]) and d["delta"] > 0)
            else "no measurable gain"))
    return "\n".join(out) + "\n"


# --------------------------------------------------------------------------
# abstention calibration
# --------------------------------------------------------------------------

def calibrate(args) -> int:
    """Sweep the abstention threshold and print the coverage/accuracy trade-off.

    Calls the real `should_abstain` rather than re-implementing a threshold
    comparison. An earlier version compared top_score to tau directly, so it
    reported 0% false abstention while the deployed system was refusing
    answerable queries via the ambiguity rule - the sweep was measuring
    something the app does not do.
    """
    from .. import answer as answer_mod
    from .. import config as cfg_mod

    conn = db.connect()
    db.init(conn)
    queries = load_queries(getattr(args, "queries", None))
    qrels = load_qrels(conn)
    if not qrels:
        print("No judgements yet - calibration needs the answer key. See `cfr pool`.")
        return 1

    cfg = RetrievalConfig(name="calibration", strategy=args.strategy,
                          use_lexical=True, use_dense=True, use_rerank=True, final_k=10)
    qvecs = _query_vectors(queries)
    retriever = Retriever(conn, cfg)

    print("running {} queries...".format(len(queries)), flush=True)
    records = []
    for q in queries:
        res = retriever.search(q["query"], qvec=qvecs.get(q["query_id"]))
        rel = qrels.get(q["query_id"], {})
        records.append({
            "result": res,
            "answerable": any(g > 0 for g in rel.values()),
            "top_ok": bool(res.hits) and rel.get(res.hits[0].doc_id, 0) > 0,
        })

    n_answerable = sum(1 for r in records if r["answerable"])
    n_oos = len(records) - n_answerable

    want_ambiguity = getattr(args, "ambiguity", None)
    modes = [False, True] if want_ambiguity is None else [bool(want_ambiguity)]
    original_tau = cfg_mod.ABSTAIN_THRESHOLD
    original_amb = cfg_mod.AMBIGUITY_ENABLED
    best_overall = None

    try:
        for amb in modes:
            cfg_mod.AMBIGUITY_ENABLED = amb
            print("\n{} answerable, {} out-of-scope   [ambiguity rule {}]".format(
                n_answerable, n_oos, "ON" if amb else "OFF"))
            head = "{:>6} {:>10} {:>12} {:>14} {:>16}".format(
                "tau", "coverage", "accuracy", "false abstain", "correct refusals")
            print(head)
            print("-" * len(head))
            best = None

            for i in range(0, 21):
                tau = i / 20.0
                cfg_mod.ABSTAIN_THRESHOLD = tau
                answered, ans_answerable, false_ab, correct_ref = [], [], 0, 0
                for rec in records:
                    abstain, _reason, _score = answer_mod.should_abstain(rec["result"])
                    if abstain:
                        if rec["answerable"]:
                            false_ab += 1
                        else:
                            correct_ref += 1
                    else:
                        answered.append(rec)
                        if rec["answerable"]:
                            ans_answerable.append(rec)

                coverage = len(answered) / len(records) if records else 0.0
                accuracy = (sum(1 for r in ans_answerable if r["top_ok"]) / len(ans_answerable)
                            if ans_answerable else float("nan"))
                false_rate = (false_ab / n_answerable) if n_answerable else float("nan")
                refuse_rate = (correct_ref / n_oos) if n_oos else float("nan")

                print("{:>6.2f} {:>9.0%} {:>11} {:>13} {:>15}".format(
                    tau, coverage,
                    "--" if accuracy != accuracy else "{:.0%}".format(accuracy),
                    "--" if false_rate != false_rate else "{:.0%}".format(false_rate),
                    "--" if refuse_rate != refuse_rate else "{:.0%}".format(refuse_rate)))

                target = getattr(args, "target_accuracy", 0.90)
                max_false = getattr(args, "max_false_abstain", 0.05)
                if (accuracy == accuracy and accuracy >= target
                        and false_rate == false_rate and false_rate <= max_false
                        and refuse_rate == refuse_rate):
                    cand = (refuse_rate, -false_rate, coverage, tau, accuracy, amb)
                    if best is None or cand > best:
                        best = cand
            print("-" * len(head))

            if best:
                refuse_rate, neg_false, coverage, tau, accuracy, _ = best
                print("  best tau {:.2f}: refuses {:.0%} of out-of-scope, "
                      "{:.0%} accuracy over {:.0%} coverage, {:.0%} false abstain".format(
                          tau, refuse_rate, accuracy, coverage, -neg_false))
                if best_overall is None or best[:3] > best_overall[:3]:
                    best_overall = best
            else:
                print("  no threshold met the target within the false-abstention ceiling")
    finally:
        cfg_mod.ABSTAIN_THRESHOLD = original_tau
        cfg_mod.AMBIGUITY_ENABLED = original_amb

    if best_overall:
        refuse_rate, neg_false, coverage, tau, accuracy, amb = best_overall
        print("\nAdopt: CFR_ABSTAIN_THRESHOLD={:.2f}  CFR_AMBIGUITY={}".format(
            tau, "1" if amb else "0"))
        print("  refuses {:.0%} of out-of-scope | {:.0%} accuracy | {:.0%} coverage | "
              "{:.0%} false abstain".format(refuse_rate, accuracy, coverage, -neg_false))
    return 0
