"""Command-line interface.

    python -m semmai verify --db hospital "Show patients older than 60" "SELECT name FROM patients WHERE age >= 60"
    python -m semmai verify --db retail --json "..." "..."
    python -m semmai intent --db travel "Flights from Chennai to Delhi under 5000"
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .schema import Schema
from .verifier import SemmaiVerifier
from .intent import IntentExtractor

DATA = Path(__file__).resolve().parents[1] / "data" / "bird_style"


def main(argv=None):
    ap = argparse.ArgumentParser(prog="semmai", description="Semantic verification for LLM-generated SQL")
    sub = ap.add_subparsers(dest="cmd", required=True)
    v = sub.add_parser("verify", help="verify that SQL preserves the question's intent")
    v.add_argument("question")
    v.add_argument("sql")
    i = sub.add_parser("intent", help="show the intent graph extracted from a question")
    i.add_argument("question")
    for p in (v, i):
        p.add_argument("--db", default="hospital", help="database id under data/bird_style (default: hospital)")
        p.add_argument("--sqlite", default=None, help="path to any other SQLite database")
        p.add_argument("--json", action="store_true", help="print JSON")
    a = ap.parse_args(argv)
    schema = Schema(Path(a.sqlite).stem, a.sqlite) if a.sqlite else Schema.from_bird_style(DATA, a.db)
    if a.cmd == "intent":
        g = IntentExtractor(schema).extract(a.question)
        print(json.dumps(g.to_dict(), indent=2, ensure_ascii=False))
        return 0
    r = SemmaiVerifier(schema).verify(a.question, a.sql)
    if a.json:
        print(json.dumps(r.to_dict(), indent=2, ensure_ascii=False))
    else:
        print(f"Decision : {r.decision}")
        print(f"SQL means: {r.action_text}")
        if r.similarity is not None:
            print(f"Similarity: {r.similarity:.2f}")
        for f in r.findings:
            print(f"  [{f.severity}] {f.kind}: {f.message}")
    return 0 if r.allowed else 1


if __name__ == "__main__":
    sys.exit(main())
