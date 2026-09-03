"""Command line entry point. Run `cfr --help` after `pip install -e .`."""

from __future__ import annotations

import argparse
import json
import sys
from typing import List, Optional, Tuple

from . import chunk as chunk_mod
from . import config, db, embed, ingest


def _parts(spec: Optional[str]) -> Optional[List[Tuple[str, str]]]:
    """Parse "40:262,29:1910" into [("40","262"), ("29","1910")]."""
    if not spec:
        return None
    out = []
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            raise SystemExit("bad --parts entry {!r}; expected TITLE:PART".format(item))
        title, part = item.split(":", 1)
        out.append((title.strip(), part.strip()))
    return out


def cmd_ingest(args) -> int:
    conn = db.connect()
    db.init(conn)
    print("Ingesting from eCFR ({})".format(config.ECFR_DATE))
    result = ingest.ingest(conn, parts=_parts(args.parts), date=args.date)
    print("Stored {} sections.".format(result["documents"]))
    return 0


def cmd_chunk(args) -> int:
    conn = db.connect()
    db.init(conn)
    strategies = args.strategy or list(chunk_mod.STRATEGIES)
    for strategy in strategies:
        res = chunk_mod.build(conn, strategy=strategy)
        checked, bad = chunk_mod.verify_offsets(conn, strategy)
        status = "ok" if bad == 0 else "{} MISMATCHED".format(bad)
        print("  {:<12} {:>7} chunks   offsets: {}/{} {}".format(
            strategy, res["chunks"], checked - bad, checked, status))
        if bad:
            print("  ! offset drift will produce citations that highlight the wrong text.")
            return 1
    return 0


def cmd_embed(args) -> int:
    conn = db.connect()
    db.init(conn)
    for strategy in (args.strategy or ["structured"]):
        print("  embedding {} with {}".format(strategy, config.EMBED_MODEL))
        res = embed.build(conn, strategy=strategy, batch_size=args.batch_size,
                          resume=not getattr(args, "rebuild", False))
        note = "" if not res.get("skipped") else "  ({} already done)".format(res["skipped"])
        print("  {:<12} {:>7} vectors{}".format(strategy, res["vectors"], note))
    return 0


def cmd_build(args) -> int:
    """Ingest, chunk every strategy, embed the ones the ablation needs."""
    rc = cmd_ingest(args)
    if rc:
        return rc
    args.strategy = list(chunk_mod.STRATEGIES)
    rc = cmd_chunk(args)
    if rc:
        return rc
    args.strategy = args.embed_strategies or ["structured", "contextual"]
    return cmd_embed(args)


def cmd_stats(args) -> int:
    conn = db.connect()
    db.init(conn)
    s = db.stats(conn)
    print("documents      {:>9,}".format(s["documents"]))
    print("characters     {:>9,}".format(s["chars"]))
    print("judged queries {:>9,}  ({:,} judgements)".format(s["judged_queries"], s["qrels"]))
    print("\nstrategy       chunks    vectors")
    for row in s["strategies"]:
        print("  {:<12} {:>7,}  {:>9,}".format(row["strategy"], row["chunks"], row["vectors"]))
    return 0


def cmd_search(args) -> int:
    from .search import RetrievalConfig, Retriever

    conn = db.connect(readonly=True)
    cfg = RetrievalConfig(
        strategy=args.strategy,
        use_lexical=not args.no_lexical,
        use_dense=not args.no_dense,
        use_rerank=not args.no_rerank,
        final_k=args.k,
    )
    result = Retriever(conn, cfg).search(args.query)
    print('"{}"  [{}]'.format(args.query, cfg.describe()))
    print("{} candidates -> {} hits   {}".format(
        result.candidate_count,
        len(result.hits),
        "  ".join("{}={:.0f}ms".format(k, v) for k, v in sorted(result.timings_ms.items())),
    ))
    print()
    for i, hit in enumerate(result.hits, 1):
        score = hit.rerank_score if hit.rerank_score is not None else hit.score
        print("{:>2}. [{:.3f}] {}".format(i, score, hit.citation))
        print("    {}".format(hit.heading[:96]))
        body = " ".join(hit.text.split())
        print("    {}".format(body[:180] + ("..." if len(body) > 180 else "")))
        print("    chars {}-{}".format(hit.char_start, hit.char_end))
        print()
    return 0


def cmd_ask(args) -> int:
    from . import answer as answer_mod
    from .search import RetrievalConfig, Retriever

    conn = db.connect()
    db.init(conn)
    retriever = Retriever(conn, RetrievalConfig(strategy=args.strategy))
    payload = answer_mod.answer(conn, retriever, args.query)
    print(json.dumps(payload, indent=2)[:6000])
    return 0


def cmd_eval(args) -> int:
    from .eval import run as eval_run

    return eval_run.main(args)


def cmd_calibrate(args) -> int:
    from .eval import run as eval_run

    return eval_run.calibrate(args)


def cmd_pool(args) -> int:
    from .eval import pool as pool_mod

    return pool_mod.main(args)


def cmd_bench(args) -> int:
    from .eval import bench

    return bench.main(args)


def cmd_serve(args) -> int:
    import uvicorn

    uvicorn.run("cfr.api:app", host=args.host, port=args.port, reload=args.reload)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="cfr", description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)

    def add_parts(sp):
        sp.add_argument("--parts", help='e.g. "40:262,29:1910" (default: config.DEFAULT_PARTS)')
        sp.add_argument("--date", help="eCFR snapshot date, YYYY-MM-DD")

    sp = sub.add_parser("ingest", help="fetch CFR parts from eCFR")
    add_parts(sp)
    sp.set_defaults(func=cmd_ingest)

    sp = sub.add_parser("chunk", help="build chunks + FTS index")
    sp.add_argument("--strategy", action="append", choices=chunk_mod.STRATEGIES)
    sp.set_defaults(func=cmd_chunk)

    sp = sub.add_parser("embed", help="build dense vectors")
    sp.add_argument("--strategy", action="append", choices=chunk_mod.STRATEGIES)
    sp.add_argument("--batch-size", type=int, default=64)
    sp.add_argument("--rebuild", action="store_true", help="re-embed chunks that already have vectors")
    sp.set_defaults(func=cmd_embed)

    sp = sub.add_parser("build", help="ingest + chunk + embed, end to end")
    add_parts(sp)
    sp.add_argument("--batch-size", type=int, default=64)
    sp.add_argument("--embed-strategies", nargs="*", choices=chunk_mod.STRATEGIES)
    sp.set_defaults(func=cmd_build)

    sp = sub.add_parser("stats", help="what is in the database")
    sp.set_defaults(func=cmd_stats)

    sp = sub.add_parser("search", help="run one query")
    sp.add_argument("query")
    sp.add_argument("-k", type=int, default=8)
    sp.add_argument("--strategy", default="structured", choices=chunk_mod.STRATEGIES)
    sp.add_argument("--no-lexical", action="store_true")
    sp.add_argument("--no-dense", action="store_true")
    sp.add_argument("--no-rerank", action="store_true")
    sp.set_defaults(func=cmd_search)

    sp = sub.add_parser("ask", help="retrieve, then generate a cited answer")
    sp.add_argument("query")
    sp.add_argument("--strategy", default="structured", choices=chunk_mod.STRATEGIES)
    sp.set_defaults(func=cmd_ask)

    sp = sub.add_parser("eval", help="run the ablation over the labelled set")
    sp.add_argument("--configs", nargs="*", help="config names to run (default: all)")
    sp.add_argument("--queries", help="path to queries.jsonl")
    sp.add_argument("--bootstrap", type=int, default=1000)
    sp.add_argument("--json-out", help="write raw results here")
    sp.add_argument("--markdown-out", help="write the ablation table here")
    sp.add_argument("--show-failures", type=int, default=0, metavar="N",
                    help="print the N worst queries under the best config")
    sp.set_defaults(func=cmd_eval)

    sp = sub.add_parser("calibrate", help="sweep the abstention threshold")
    sp.add_argument("--queries", help="path to queries.jsonl")
    sp.add_argument("--strategy", default="structured", choices=chunk_mod.STRATEGIES)
    sp.add_argument("--target-accuracy", type=float, default=0.90)
    sp.add_argument("--max-false-abstain", type=float, default=0.05,
                    help="ceiling on answerable queries wrongly refused")
    sp.add_argument("--ambiguity", type=int, choices=(0, 1), default=None,
                    help="force the ambiguity rule on/off (default: sweep both)")
    sp.set_defaults(func=cmd_calibrate)

    sp = sub.add_parser("pool", help="build a labelling pool from multiple configs")
    sp.add_argument("--queries", help="path to queries.jsonl")
    sp.add_argument("--depth", type=int, default=20)
    sp.add_argument("--out", help="where to write pool.jsonl")
    sp.set_defaults(func=cmd_pool)

    sp = sub.add_parser("bench", help="latency per stage, across rerank depths")
    sp.add_argument("--queries", help="path to queries.jsonl")
    sp.add_argument("-n", type=int, default=20, help="how many queries to time")
    sp.add_argument("--strategy", default="structured", choices=chunk_mod.STRATEGIES)
    sp.add_argument("--depths", type=int, nargs="*", help="rerank_top_n values to try")
    sp.set_defaults(func=cmd_bench)

    sp = sub.add_parser("serve", help="run the web app")
    sp.add_argument("--host", default="127.0.0.1")
    sp.add_argument("--port", type=int, default=8000)
    sp.add_argument("--reload", action="store_true")
    sp.set_defaults(func=cmd_serve)

    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
