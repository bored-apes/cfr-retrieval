"""Does the researcher's query reformulation earn its LLM call?

The supervisor's routing is free - it is a few dict lookups. The one place the
multi-agent variant spends real money is the researcher's second pass, where a
below-threshold first attempt buys an LLM rewrite of the question into
regulatory vocabulary. That is a claim, and the eval set can test it.

Each of the 60 judged queries is retrieved and reranked. Any query scoring below
the abstention threshold triggers a rewrite, and the rewrite is then scored
against the graded judgements two ways - with and without the faithfulness
guard - using the same rewrite for both, so the guard's effect is isolated from
the model's nondeterminism.

Outcomes, in descending order of how much they matter:

  HARMFUL    a question that must be refused was rewritten into one the corpus
             answers, crossing the threshold. This defeats the abstention gate,
             which is the safety property the whole system is built around.
  rescued    below threshold -> above it, on a query that really is answerable
  improved   nDCG@10 went up
  no change  an LLM call that bought nothing

The rewrite is an LLM call, so it is sampled repeatedly - a failure that shows
up once in nine is a curiosity; a failure rate over thirty draws is a finding.

Run: python scripts/measure_reformulation.py [repeats]
"""

from __future__ import annotations

import json
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cfr import config, db, embed  # noqa: E402
from cfr.agents import roles  # noqa: E402
from cfr.eval import metrics  # noqa: E402
from cfr.search import dense, fusion, lexical  # noqa: E402
from cfr.search import rerank as rerank_mod  # noqa: E402
from cfr.search.pipeline import doc_id_of  # noqa: E402

STRATEGY = "structured"
OUT = ROOT / "evaldata" / "reformulation.json"


def load():
    queries = [json.loads(x) for x in
               (ROOT / "evaldata/queries.jsonl").read_text().splitlines() if x.strip()]
    qrels = defaultdict(dict)
    for line in (ROOT / "evaldata/qrels.jsonl").read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            qrels[r["query_id"]][r["doc_id"]] = int(r["grade"])
    return queries, qrels


def answerable(rel) -> bool:
    """Out-of-scope queries are judged too - every row is graded 0. Presence of
    judgements is not evidence the corpus answers the question."""
    return bool(rel) and max(rel.values()) > 0


def retrieve(conn, q):
    """One retrieval + rerank pass. The same code the researcher runs."""
    lex = lexical.search(conn, q, STRATEGY, config.CANDIDATES_PER_RETRIEVER)
    den = dense.search(conn, q, STRATEGY, config.CANDIDATES_PER_RETRIEVER,
                       qvec=embed.embed_query(q))
    fused = fusion.rrf([lex, den], k=config.RRF_K)
    shortlist = fused[: config.RERANK_TOP_N]
    if not shortlist:
        return [], 0.0

    marks = ",".join("?" * len(shortlist))
    rows = conn.execute(
        "SELECT chunk_id, text FROM chunks WHERE chunk_id IN ({})".format(marks),
        [c for c, _ in shortlist]).fetchall()
    text = {r["chunk_id"]: r["text"] for r in rows}

    ids = [c for c, _ in shortlist if c in text]
    logits = rerank_mod.rerank_logits(q, [text[c] for c in ids])
    scored = sorted(zip(ids, logits), key=lambda p: -p[1])
    ranked = metrics.dedupe_docs([doc_id_of(c) for c, _ in scored])
    return ranked, rerank_mod.to_confidence(scored[0][1])


def classify(rel, conf1, conf2, ndcg1, ndcg2, tau):
    crossed = conf2 >= tau
    if crossed and not answerable(rel):
        return "HARMFUL"
    if crossed and ndcg2 > 0:
        return "rescued"
    if ndcg2 > ndcg1 + 1e-9:
        return "improved"
    return "no change"


def main(repeats: int = 1) -> int:
    queries, qrels = load()
    conn = db.connect()
    db.init(conn)
    tau = config.ABSTAIN_THRESHOLD

    print("retrieving over {} judged queries (tau = {:.2f})...\n".format(len(queries), tau),
          flush=True)

    rows = []
    for q in queries:
        ranked, conf = retrieve(conn, q["query"])
        rel = qrels.get(q["query_id"], {})
        rows.append({"q": q, "conf": conf, "rel": rel,
                     "ndcg": metrics.ndcg_at_k(ranked, rel, 10) or 0.0,
                     "answerable": answerable(rel)})

    triggered = [r for r in rows if r["conf"] < tau]
    n_ans = sum(r["answerable"] for r in triggered)
    print("first pass: {} of {} queries fell below tau and would trigger a rewrite".format(
        len(triggered), len(rows)))
    print("  answerable from the corpus  {}".format(n_ans))
    print("  genuinely out of scope      {}\n".format(len(triggered) - n_ans))
    missed = [r for r in rows if r["answerable"] and r["conf"] < tau]
    if not missed:
        print("Every answerable query already clears the threshold on the first pass,")
        print("so reformulation has nothing to rescue here - only something to break.\n")

    results, latencies = [], []
    for rep in range(repeats):
        if repeats > 1:
            print("--- draw {} of {} ---".format(rep + 1, repeats), flush=True)
        for r in triggered:
            q = r["q"]
            t0 = time.perf_counter()
            rewritten = roles._reformulate(q["query"], [q["query"]])
            latencies.append((time.perf_counter() - t0) * 1000)
            if not rewritten or rewritten == q["query"]:
                results.append({"qid": q["query_id"], "type": q["type"], "rep": rep,
                                "query": q["query"], "rewritten": "",
                                "conf": r["conf"], "conf2": r["conf"],
                                "ndcg": r["ndcg"], "ndcg2": r["ndcg"], "sim": 1.0,
                                "answerable": r["answerable"], "verdict": "no change"})
                continue

            ranked2, conf2 = retrieve(conn, rewritten)
            ndcg2 = metrics.ndcg_at_k(ranked2, r["rel"], 10) or 0.0
            sim = float(np.dot(embed.embed_query(q["query"]).ravel(),
                               embed.embed_query(rewritten).ravel()))
            verdict = classify(r["rel"], r["conf"], conf2, r["ndcg"], ndcg2, tau)

            results.append({"qid": q["query_id"], "type": q["type"], "rep": rep,
                            "query": q["query"], "rewritten": rewritten,
                            "conf": r["conf"], "conf2": conf2,
                            "ndcg": r["ndcg"], "ndcg2": ndcg2, "sim": sim,
                            "answerable": r["answerable"], "verdict": verdict})
            flag = "  <-- crossed tau" if verdict == "HARMFUL" else ""
            print("  [{:<9}] {:<12} conf {:.2f}->{:.2f}  nDCG {:.3f}->{:.3f}  sim {:.2f}{}".format(
                verdict, q["type"], r["conf"], conf2, r["ndcg"], ndcg2, sim, flag))
            if verdict == "HARMFUL" or repeats == 1:
                print("     was: {}\n     now: {}".format(q["query"][:72], rewritten[:72]),
                      flush=True)

    tally = lambda key: {v: sum(1 for x in results if x[key] == v)  # noqa: E731
                         for v in ("HARMFUL", "rescued", "improved", "no change")}
    raw = tally("verdict")

    print("\n" + "=" * 78)
    print("UNGUARDED")
    for k in ("HARMFUL", "rescued", "improved", "no change"):
        print("  {:<24} {}".format(k, raw[k]))
    if latencies:
        print("  cost                     {:.0f} ms p50, {} extra LLM calls".format(
            statistics.median(latencies), len(latencies)))
    n_rw = len(results)
    print("  harmful rate             {}/{} = {:.0%} of rewrites defeated the "
          "abstention gate".format(raw["HARMFUL"], n_rw, raw["HARMFUL"] / max(n_rw, 1)))

    # --- the guard --------------------------------------------------------
    # A rewrite is only allowed to stand if it still means roughly what the user
    # asked. Cosine similarity between the original and the rewrite, in the same
    # embedding space the retriever uses.
    harmful_sims = [x["sim"] for x in results if x["verdict"] == "HARMFUL"]
    safe_sims = [x["sim"] for x in results if x["verdict"] != "HARMFUL" and x["rewritten"]]
    print("\nfaithfulness (cosine of original vs rewrite, {}):".format(config.EMBED_MODEL
          if hasattr(config, "EMBED_MODEL") else "query embedding space"))
    if harmful_sims:
        print("  harmful rewrites   min {:.2f}  max {:.2f}".format(
            min(harmful_sims), max(harmful_sims)))
    if safe_sims:
        print("  other rewrites     min {:.2f}  max {:.2f}".format(
            min(safe_sims), max(safe_sims)))

    for thresh in (0.50, 0.60, 0.70, 0.75, 0.80):
        kept = [x for x in results if x["sim"] >= thresh or not x["rewritten"]]
        blocked_harm = sum(1 for x in results if x["verdict"] == "HARMFUL" and x["sim"] < thresh)
        lost = sum(1 for x in results if x["verdict"] in ("rescued", "improved")
                   and x["sim"] < thresh)
        print("  guard >= {:.2f}   blocks {}/{} harmful, costs {} good rewrite(s), "
              "{} rewrites survive".format(thresh, blocked_harm, len(harmful_sims), lost,
                                           len([x for x in kept if x["rewritten"]])))
    print("=" * 78)

    OUT.write_text(json.dumps({"tau": tau, "repeats": repeats, "results": results},
                              indent=2))
    print("\nwrote {}".format(OUT.relative_to(ROOT)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(int(sys.argv[1]) if len(sys.argv) > 1 else 1))
