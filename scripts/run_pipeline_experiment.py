"""End-to-end agent experiment: what actually gets executed, with and without Semmai.

For each question of the generated-test and blind sets, an LLM (SimulatedLLM
with a given error rate, or a real model through Ollama with --ollama) writes
SQL, and three pipelines run it:

    baseline         Generate -> Execute
    semmai_block     Generate -> Verify -> Execute / Reject
    semmai_regen     Generate -> Verify -> (Reject -> regenerate, max 3) -> Execute

An execution is "wrong" when the executed query's meaning differs from the
user's request (the injected error), i.e. a silent wrong answer.

Usage:
    python scripts/run_pipeline_experiment.py
    python scripts/run_pipeline_experiment.py --ollama qwen2.5-coder:7b   # real LLM (needs Ollama)
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from semmai.generator import OllamaGenerator, SimulatedLLM      # noqa: E402
from semmai.schema import Schema                                 # noqa: E402
from semmai.verifier import SemmaiVerifier                       # noqa: E402

DATA = ROOT / "data" / "bird_style"
RES = ROOT / "results"


def same(a, b):
    k = lambda r: tuple(map(str, r))
    return sorted(a, key=k) == sorted(b, key=k)


def load_questions():
    pairs = json.loads((DATA / "verification_pairs.json").read_text())
    test_ids = {p["question_id"] for p in pairs if p["split"] == "test"}
    qs = [q for q in json.loads((DATA / "questions.json").read_text()) if q["question_id"] in test_ids]
    qs += json.loads((DATA / "blind_test.json").read_text())
    return qs


def simulated(error_rates, seeds=(0, 1, 2)):
    qs = load_questions()
    schemas, verifiers = {}, {}
    out = []
    for rate in error_rates:
        agg = {"error_rate": rate, "questions": 0, "baseline_wrong": 0, "block_wrong": 0, "block_rejected": 0,
               "block_correct": 0, "regen_wrong": 0, "regen_rejected": 0, "regen_correct": 0, "regen_attempts": 0,
               "block_false_rejections": 0}
        for seed in seeds:
            llm = SimulatedLLM(rate, seed=seed * 1000 + int(rate * 100))
            for q in qs:
                db = q["db_id"]
                if db not in schemas:
                    schemas[db] = Schema.from_bird_style(DATA, db)
                    verifiers[db] = SemmaiVerifier(schemas[db], use_similarity=False)
                s, v = schemas[db], verifiers[db]
                agg["questions"] += 1
                sql, err = llm.generate(q["question"], s, q["SQL"])
                wrong = err is not None
                agg["baseline_wrong"] += wrong
                ok = v.verify(q["question"], sql).allowed
                if ok:
                    agg["block_wrong"] += wrong
                    agg["block_correct"] += not wrong
                else:
                    agg["block_rejected"] += 1
                    agg["block_false_rejections"] += not wrong
                # regenerate loop (first attempt reused)
                attempts, final = 1, None
                while True:
                    if ok:
                        final = wrong
                        break
                    if attempts >= 3:
                        break
                    attempts += 1
                    sql, err = llm.generate(q["question"], s, q["SQL"])
                    wrong = err is not None
                    ok = v.verify(q["question"], sql).allowed
                agg["regen_attempts"] += attempts
                if final is None:
                    agg["regen_rejected"] += 1
                elif final:
                    agg["regen_wrong"] += 1
                else:
                    agg["regen_correct"] += 1
        n = agg["questions"]
        res = {"error_rate": rate, "n": n,
               "baseline": {"correct": 1 - agg["baseline_wrong"] / n, "wrong_executed": agg["baseline_wrong"] / n,
                            "rejected": 0.0},
               "semmai_block": {"correct": agg["block_correct"] / n, "wrong_executed": agg["block_wrong"] / n,
                                "rejected": agg["block_rejected"] / n,
                                "false_rejections": agg["block_false_rejections"] / n},
               "semmai_regen": {"correct": agg["regen_correct"] / n, "wrong_executed": agg["regen_wrong"] / n,
                                "rejected": agg["regen_rejected"] / n, "mean_attempts": agg["regen_attempts"] / n}}
        out.append(res)
        print(json.dumps(res))
    return out


def with_ollama(model):
    qs = load_questions()
    gen = OllamaGenerator(model)
    rows = []
    for q in qs:
        s = Schema.from_bird_style(DATA, q["db_id"])
        sql, _ = gen.generate(q["question"], s)
        con = sqlite3.connect(s.db_path)
        try:
            ok_exec = same(con.execute(sql).fetchall(), con.execute(q["SQL"]).fetchall())
        except sqlite3.Error:
            ok_exec = False
        v = SemmaiVerifier(s, use_similarity=False).verify(q["question"], sql)
        rows.append({"question": q["question"], "gold": q["SQL"], "generated": sql, "exec_match": ok_exec,
                     "semmai": v.decision, "findings": [f.message for f in v.findings]})
        print(v.decision, ok_exec, sql)
    (RES / f"ollama_{model.replace(':', '_')}.json").write_text(json.dumps(rows, indent=2))
    n = len(rows)
    wrong = [r for r in rows if not r["exec_match"]]
    print(f"LLM execution accuracy: {1 - len(wrong) / n:.1%}; wrong queries blocked by Semmai: "
          f"{sum(r['semmai'] == 'REJECT' for r in wrong)}/{len(wrong)}; correct queries allowed: "
          f"{sum(r['semmai'] == 'ALLOW' for r in rows if r['exec_match'])}/{n - len(wrong)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ollama", default=None, help="Ollama model name, e.g. qwen2.5-coder:7b")
    args = ap.parse_args()
    RES.mkdir(exist_ok=True)
    if args.ollama:
        with_ollama(args.ollama)
        return
    out = simulated([0.1, 0.2, 0.3, 0.4])
    (RES / "pipeline_results.json").write_text(json.dumps(out, indent=2))
    md = ["# End-to-end pipeline experiment (Simulated LLM, 3 seeds)", "",
          "| LLM error rate | Pipeline | Correct answers | Wrong answers executed ↓ | Rejected (no answer) |",
          "|---|---|---|---|---|"]
    for r in out:
        for k, name in (("baseline", "Generate → Execute"), ("semmai_block", "Generate → Verify → Execute/Reject"),
                        ("semmai_regen", "… + regenerate (≤3 attempts)")):
            x = r[k]
            md.append(f"| {r['error_rate']:.0%} | {name} | {x['correct']:.1%} | {x['wrong_executed']:.1%} | "
                      f"{x['rejected']:.1%} |")
    (RES / "pipeline_results.md").write_text("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
