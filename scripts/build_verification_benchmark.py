"""Build the SQL verification benchmark from a BIRD-format question file.

For each (question, gold SQL):
    + the gold SQL                         label 1 (preserves intent)
    + 1 equivalent rewrite                 label 1
    + up to 3 semantic mutations           label 0 (intent violated)

Every rewrite is checked by execution against the database (it must return
exactly the gold result); every mutation records whether it *happens* to
return the same rows as the gold query on the current data (exec_equal) -
those are the errors execution-based checking cannot see.

Usage:
    python scripts/build_verification_benchmark.py                     # Semmai BIRD-style set
    python scripts/build_verification_benchmark.py --questions data/bird_style/handwritten.json \
           --out data/bird_style/verification_handwritten.json
    python scripts/build_verification_benchmark.py --bird-root data/official/bird/dev   # official BIRD
"""
from __future__ import annotations

import argparse
import json
import random
import sqlite3
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from semmai.mutations import MUTATIONS, REWRITES, Mutator          # noqa: E402
from semmai.schema import Schema                                     # noqa: E402
from semmai.sql_parser import UnsupportedSQL, sql_to_usig            # noqa: E402


def run(con, sql):
    rows = con.execute(sql).fetchall()
    return rows


def same_result(a, b, ordered):
    if ordered:
        return a == b
    key = lambda r: tuple(str(x) for x in r)
    return sorted(a, key=key) == sorted(b, key=key)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", default=str(ROOT / "data/bird_style/questions.json"))
    ap.add_argument("--out", default=str(ROOT / "data/bird_style/verification_pairs.json"))
    ap.add_argument("--data-root", default=str(ROOT / "data/bird_style"))
    ap.add_argument("--bird-root", default=None, help="official BIRD dev folder (dev.json + dev_databases/)")
    ap.add_argument("--dev-frac", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--max-mutations", type=int, default=3)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    if args.bird_root:
        broot = Path(args.bird_root)
        questions = json.loads((broot / "dev.json").read_text())
        db_path = lambda db: broot / "dev_databases" / db / f"{db}.sqlite"
        mk_schema = lambda db: Schema(db, db_path(db))
    else:
        questions = json.loads(Path(args.questions).read_text())
        mk_schema = lambda db: Schema.from_bird_style(args.data_root, db)

    schemas, cons = {}, {}
    usage = Counter()
    pairs, skipped, bad_rewrites = [], Counter(), 0
    dbs = sorted({q["db_id"] for q in questions})
    # question-level split, stratified by database
    split = {}
    for db in dbs:
        qs = [i for i, q in enumerate(questions) if q["db_id"] == db]
        rng.shuffle(qs)
        k = int(round(len(qs) * args.dev_frac))
        for j, i in enumerate(qs):
            split[i] = "dev" if j < k else "test"

    pid = 0
    for qi, q in enumerate(questions):
        db = q["db_id"]
        if db not in schemas:
            schemas[db] = mk_schema(db)
            cons[db] = sqlite3.connect(schemas[db].db_path)
        s, con = schemas[db], cons[db]
        gold_sql = q.get("SQL") or q.get("query")
        try:
            g = sql_to_usig(gold_sql, s)
        except UnsupportedSQL as e:
            skipped[f"unsupported gold: {str(e)[:40]}"] += 1
            continue
        gold_rows = run(con, gold_sql)
        ordered = bool(g.order_by)
        mut = Mutator(s, rng)
        base = {"question_id": q.get("question_id", qi), "db_id": db, "question": q["question"],
                "evidence": q.get("evidence", ""), "difficulty": q.get("difficulty", ""),
                "qtype": q.get("qtype", ""), "split": split[qi], "gold_sql": gold_sql}

        def add(sql, label, kind):
            nonlocal pid
            try:
                rows = run(con, sql)
            except sqlite3.Error as e:
                skipped[f"sqlite error in {kind}"] += 1
                return False
            eq = same_result(rows, gold_rows, ordered)
            pairs.append(dict(base, pair_id=pid, sql=sql, label=label, kind=kind, exec_equal=eq))
            pid += 1
            return True

        add(gold_sql, 1, "GOLD")
        rws = REWRITES[:]
        rng.shuffle(rws)
        for kind in rws:
            sql = mut.rewrite(g, kind)
            if not sql or sql == gold_sql:
                continue
            rows = run(con, sql)
            if not same_result(rows, gold_rows, ordered):
                bad_rewrites += 1
                continue
            add(sql, 1, f"REWRITE_{kind}")
            break
        kinds = sorted(MUTATIONS, key=lambda k: (usage[k], rng.random()))
        made = 0
        for kind in kinds:
            if made >= args.max_mutations:
                break
            sql = mut.mutate(g, kind)
            if not sql or sql == gold_sql:
                continue
            if add(sql, 0, kind):
                usage[kind] += 1
                made += 1

    Path(args.out).write_text(json.dumps(pairs, indent=1, ensure_ascii=False))
    print(f"wrote {len(pairs)} pairs -> {args.out}")
    print("labels:", Counter(p["label"] for p in pairs))
    print("kinds:", dict(Counter(p["kind"] for p in pairs)))
    print("split:", Counter((p["split"], p["label"]) for p in pairs))
    neg = [p for p in pairs if p["label"] == 0]
    print(f"mutations returning the SAME rows as gold: {sum(p['exec_equal'] for p in neg)}/{len(neg)}")
    print("rewrites rejected by execution check (dropped):", bad_rewrites, "| skipped:", dict(skipped))


if __name__ == "__main__":
    main()
