"""Labelling helper: dump candidates for judging, then apply grades.

Deliberately hides `found_by` and `best_rank` when dumping. Seeing which system
surfaced a candidate, or how highly, biases the judgement toward that system -
which is the exact failure pooling exists to prevent.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cfr import config, db  # noqa: E402


def load_pool():
    path = config.EVAL_DIR / "pool.jsonl"
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def judged_pairs(conn):
    return {(r["query_id"], r["doc_id"]) for r in conn.execute("SELECT query_id, doc_id FROM qrels")}


def cmd_dump(args):
    conn = db.connect()
    db.init(conn)
    done = judged_pairs(conn)
    pool = [r for r in load_pool() if (r["query_id"], r["doc_id"]) not in done]

    by_query = {}
    for r in pool:
        by_query.setdefault(r["query_id"], []).append(r)

    order = [q for q in dict.fromkeys(r["query_id"] for r in pool)]
    if args.only:
        order = [q for q in order if q in set(args.only)]
    order = order[: args.queries]

    for qid in order:
        items = by_query[qid]
        print("\n### {} [{}] {}".format(qid, items[0]["type"], items[0]["query"]))
        for r in sorted(items, key=lambda x: x["doc_id"]):
            snippet = r["snippet"][: args.chars].replace("\n", " ")
            print("- {} | {} | {}".format(r["doc_id"], r["citation"], r["heading"][:90]))
            print("    {}".format(snippet))
    print("\n<<< {} queries, {} pairs >>>".format(
        len(order), sum(len(by_query[q]) for q in order)))


def cmd_apply(args):
    conn = db.connect()
    db.init(conn)
    payload = json.loads(Path(args.file).read_text(encoding="utf-8"))
    now = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")

    rows = []
    for qid, grades in payload.items():
        for doc_id, grade in grades.items():
            grade = int(grade)
            if grade not in (0, 1, 2, 3):
                raise SystemExit("bad grade {} for {}/{}".format(grade, qid, doc_id))
            rows.append((qid, doc_id, grade, now, args.by, args.note))

    # Never overwrite a human judgement with a machine one.
    conn.executemany(
        """INSERT INTO qrels (query_id, doc_id, grade, judged_at, judged_by, note)
           VALUES (?,?,?,?,?,?)
           ON CONFLICT(query_id, doc_id) DO UPDATE SET
             grade=excluded.grade, judged_at=excluded.judged_at,
             judged_by=excluded.judged_by, note=excluded.note
           WHERE qrels.judged_by != 'human' OR excluded.judged_by = 'human'""",
        rows,
    )
    conn.commit()
    total = conn.execute("SELECT COUNT(*) FROM qrels").fetchone()[0]
    print("applied {} judgements ({}); {} total".format(len(rows), args.by, total))


def cmd_status(args):
    conn = db.connect()
    db.init(conn)
    pool = load_pool()
    done = judged_pairs(conn)
    remaining = [r for r in pool if (r["query_id"], r["doc_id"]) not in done]
    print("pool {} | judged {} | remaining {}".format(len(pool), len(done), len(remaining)))
    for r in conn.execute("SELECT judged_by, COUNT(*) n FROM qrels GROUP BY judged_by"):
        print("  by {}: {}".format(r["judged_by"], r["n"]))
    for r in conn.execute("SELECT grade, COUNT(*) n FROM qrels GROUP BY grade ORDER BY grade"):
        print("  grade {}: {}".format(r["grade"], r["n"]))
    qs = conn.execute(
        "SELECT COUNT(DISTINCT query_id) FROM qrels WHERE grade > 0").fetchone()[0]
    print("  queries with >=1 relevant doc: {}".format(qs))


def cmd_export(args):
    """Write qrels to a version-controlled file.

    The database is a rebuildable artifact and is gitignored; the answer key is
    not. Days of judgement should not live only inside a 112 MB binary that is
    excluded from the repository.
    """
    conn = db.connect()
    db.init(conn)
    rows = [
        dict(r) for r in conn.execute(
            "SELECT query_id, doc_id, grade, judged_at, judged_by, note "
            "FROM qrels ORDER BY query_id, doc_id"
        )
    ]
    out = Path(args.out or (config.EVAL_DIR / "qrels.jsonl"))
    out.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8"
    )
    print("exported {} judgements -> {}".format(len(rows), out))


def cmd_import(args):
    """Load qrels back into a freshly built database."""
    conn = db.connect()
    db.init(conn)
    path = Path(args.file or (config.EVAL_DIR / "qrels.jsonl"))
    if not path.exists():
        raise SystemExit("no qrels file at {}".format(path))
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    conn.executemany(
        """INSERT INTO qrels (query_id, doc_id, grade, judged_at, judged_by, note)
           VALUES (:query_id, :doc_id, :grade, :judged_at, :judged_by, :note)
           ON CONFLICT(query_id, doc_id) DO UPDATE SET
             grade=excluded.grade, judged_at=excluded.judged_at,
             judged_by=excluded.judged_by, note=excluded.note
           WHERE qrels.judged_by != 'human' OR excluded.judged_by = 'human'""",
        rows,
    )
    conn.commit()
    print("imported {} judgements from {}".format(len(rows), path))


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("dump")
    d.add_argument("--queries", type=int, default=6)
    d.add_argument("--chars", type=int, default=420)
    d.add_argument("--only", nargs="*")
    d.set_defaults(func=cmd_dump)

    a = sub.add_parser("apply")
    a.add_argument("file")
    a.add_argument("--by", default="machine:claude")
    a.add_argument("--note")
    a.set_defaults(func=cmd_apply)

    s = sub.add_parser("status")
    s.set_defaults(func=cmd_status)

    e = sub.add_parser("export", help="write qrels to evaldata/qrels.jsonl")
    e.add_argument("--out")
    e.set_defaults(func=cmd_export)

    i = sub.add_parser("import", help="load qrels from evaldata/qrels.jsonl")
    i.add_argument("file", nargs="?")
    i.set_defaults(func=cmd_import)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
