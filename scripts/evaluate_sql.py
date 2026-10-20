"""Evaluate Semmai and baselines on the SQL verification benchmarks.

Methods
    syntax_permission   SQLite compiles the query and it only touches known tables
                        (what current agent frameworks check)
    execution_nonempty  query runs and returns at least one non-null row
    similarity_sql      char n-gram similarity(question, raw SQL) >= t
    similarity_text     similarity(question, verbalised SQL meaning) >= t
                        (the "semantic similarity" design from the proposal)
    semmai_proof        Semmai symbolic proof engine only
    semmai              Semmai proof engine + similarity gate (full system)
    exec_match_oracle   result equals the GOLD query's result - needs the gold
                        answer, so it is NOT deployable; shown as an upper
                        reference for execution-based evaluation

Thresholds t are tuned on the dev split of the generated benchmark only.

Usage:  python scripts/evaluate_sql.py
"""
from __future__ import annotations

import csv
import json
import sqlite3
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from semmai.metrics import best_threshold, scores               # noqa: E402
from semmai.schema import Schema                                 # noqa: E402
from semmai.similarity import Similarity                         # noqa: E402
from semmai.sql_parser import UnsupportedSQL, sql_to_usig        # noqa: E402
from semmai.verbalize import verbalize                           # noqa: E402
from semmai.verifier import SemmaiVerifier                       # noqa: E402

DATA = ROOT / "data" / "bird_style"
RES = ROOT / "results"
SETS = {"generated": DATA / "verification_pairs.json", "handwritten": DATA / "verification_handwritten.json",
        "blind": DATA / "verification_blind.json"}
METHODS = ["syntax_permission", "execution_nonempty", "similarity_sql", "similarity_text", "semmai_proof",
           "semmai", "exec_match_oracle"]
NAMES = {"syntax_permission": "Syntax + permission check", "execution_nonempty": "Execution sanity (non-empty)",
         "similarity_sql": "Similarity: question vs SQL", "similarity_text": "Similarity: question vs SQL meaning",
         "semmai_proof": "Semmai (proof engine only)", "semmai": "Semmai (proof + similarity)",
         "exec_match_oracle": "Execution-match oracle*"}


def main():
    RES.mkdir(exist_ok=True)
    sim = Similarity()
    print("similarity backend:", sim.backend)
    schemas, verifiers, cons = {}, {}, {}

    def ctx(db):
        if db not in schemas:
            schemas[db] = Schema.from_bird_style(DATA, db)
            verifiers[db] = SemmaiVerifier(schemas[db], similarity=sim, use_similarity=False)
            cons[db] = sqlite3.connect(schemas[db].db_path)
        return schemas[db], verifiers[db], cons[db]

    rows = []           # one row per (set, pair)
    for set_name, path in SETS.items():
        pairs = json.loads(path.read_text())
        for p in pairs:
            s, v, con = ctx(p["db_id"])
            # --- syntax / permission
            try:
                con.execute("EXPLAIN " + p["sql"]).fetchall()
                syntax_ok = True
            except sqlite3.Error:
                syntax_ok = False
            # --- execution sanity
            res = con.execute(p["sql"]).fetchall() if syntax_ok else []
            nonempty = bool(res) and not all(all(x in (None, 0) for x in r) for r in res)
            # --- similarity
            try:
                text = verbalize(sql_to_usig(p["sql"], s), s.col_types())
            except UnsupportedSQL:
                text = p["sql"]
            sim_sql = sim.score(p["question"], p["sql"])
            sim_text = sim.score(p["question"], text)
            # --- Semmai proof
            t0 = time.perf_counter()
            verdict = v.verify(p["question"], p["sql"])
            ms = (time.perf_counter() - t0) * 1000
            rows.append({"set": set_name, "pair_id": p["pair_id"], "db_id": p["db_id"], "split": p["split"],
                         "kind": p["kind"], "label": p["label"], "difficulty": p.get("difficulty", ""),
                         "qtype": p.get("qtype", ""), "exec_equal": p["exec_equal"],
                         "question": p["question"], "sql": p["sql"],
                         "syntax_ok": syntax_ok, "nonempty": nonempty, "sim_sql": sim_sql, "sim_text": sim_text,
                         "proof_allow": verdict.allowed, "latency_ms": ms,
                         "findings": "; ".join(f"{f.kind}: {f.message}" for f in verdict.critical())})

    # ---------------------------------------------------- tune thresholds on dev
    dev = [r for r in rows if r["set"] == "generated" and r["split"] == "dev"]
    t_sql = best_threshold([r["sim_sql"] for r in dev], [r["label"] for r in dev])
    t_text = best_threshold([r["sim_text"] for r in dev], [r["label"] for r in dev])
    # full Semmai: similarity gate on top of the proof; choose the gate that maximises
    # dev F1 among actions the proof engine accepted (0 = gate disabled)
    acc_dev = [r for r in dev if r["proof_allow"]]
    cands = [0.0] + sorted({round(r["sim_text"], 4) for r in acc_dev})
    best_gate, best_f1 = 0.0, -1
    for t in cands:
        f1 = scores([r["label"] for r in dev], [r["proof_allow"] and r["sim_text"] >= t for r in dev])["f1"]
        if f1 > best_f1 + 1e-12:
            best_f1, best_gate = f1, t
    thresholds = {"similarity_sql": t_sql, "similarity_text": t_text, "semmai_gate": best_gate}
    from sklearn.metrics import roc_auc_score
    auc = {}
    for nm in ("generated", "handwritten", "blind"):
        rs = [r for r in rows if r["set"] == nm and (nm != "generated" or r["split"] == "test")]
        auc[nm] = {k: roc_auc_score([r["label"] for r in rs], [r[k] for r in rs]) for k in ("sim_sql", "sim_text")}
    print("thresholds (tuned on dev):", thresholds)

    def decide(r, m):
        if m == "syntax_permission":
            return r["syntax_ok"]
        if m == "execution_nonempty":
            return r["syntax_ok"] and r["nonempty"]
        if m == "similarity_sql":
            return r["sim_sql"] >= t_sql
        if m == "similarity_text":
            return r["sim_text"] >= t_text
        if m == "semmai_proof":
            return r["proof_allow"]
        if m == "semmai":
            return r["proof_allow"] and r["sim_text"] >= best_gate
        if m == "exec_match_oracle":
            return r["exec_equal"]
        raise ValueError(m)

    for r in rows:
        for m in METHODS:
            r[f"allow_{m}"] = decide(r, m)

    # ---------------------------------------------------------- main tables
    evalsets = {"generated_test": [r for r in rows if r["set"] == "generated" and r["split"] == "test"],
                "handwritten": [r for r in rows if r["set"] == "handwritten"],
                "blind": [r for r in rows if r["set"] == "blind"]}
    summary = {"similarity_backend": sim.backend, "thresholds": thresholds, "similarity_roc_auc": auc, "sets": {}}
    for name, rs in evalsets.items():
        summary["sets"][name] = {
            "pairs": len(rs), "correct_actions": sum(r["label"] for r in rs),
            "mismatched_actions": sum(1 - r["label"] for r in rs),
            "mismatches_with_same_execution_result": sum(1 for r in rs if r["label"] == 0 and r["exec_equal"]),
            "methods": {m: scores([r["label"] for r in rs], [r[f"allow_{m}"] for r in rs]) for m in METHODS}}
    lat = [r["latency_ms"] for r in rows]
    summary["latency_ms"] = {"mean": statistics.mean(lat), "median": statistics.median(lat),
                             "p95": sorted(lat)[int(0.95 * len(lat))]}

    # per error type (detection rate) and per rewrite type (acceptance rate)
    per_type = defaultdict(dict)
    test_like = evalsets["generated_test"] + evalsets["handwritten"] + evalsets["blind"]
    kinds = sorted({r["kind"] for r in test_like})
    for k in kinds:
        rs = [r for r in test_like if r["kind"] == k]
        for m in METHODS:
            if rs[0]["label"] == 0:
                per_type[k][m] = sum(not r[f"allow_{m}"] for r in rs) / len(rs)   # detection rate
            else:
                per_type[k][m] = sum(r[f"allow_{m}"] for r in rs) / len(rs)       # acceptance rate
        per_type[k]["n"] = len(rs)
        per_type[k]["label"] = rs[0]["label"]
    summary["per_type"] = per_type
    # per difficulty (generated test)
    per_diff = {}
    for d in ("simple", "moderate", "challenging"):
        rs = [r for r in evalsets["generated_test"] if r["difficulty"] == d]
        if rs:
            per_diff[d] = {m: scores([r["label"] for r in rs], [r[f"allow_{m}"] for r in rs]) for m in
                           ("similarity_text", "semmai")}
            per_diff[d]["n"] = len(rs)
    summary["per_difficulty"] = per_diff

    (RES / "sql_results.json").write_text(json.dumps(summary, indent=2))
    with open(RES / "sql_predictions.csv", "w", newline="", encoding="utf-8") as f:
        keys = ["set", "pair_id", "db_id", "split", "kind", "label", "difficulty", "exec_equal", "question", "sql",
                "sim_text", "findings"] + [f"allow_{m}" for m in METHODS]
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({**r, "sim_text": round(r["sim_text"], 4)})
    with open(RES / "similarity_scores.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["set", "label", "kind", "sim_text", "sim_sql"])
        for r in rows:
            w.writerow([r["set"], r["label"], r["kind"], round(r["sim_text"], 4), round(r["sim_sql"], 4)])

    # ------------------------------------------------------------ markdown
    md = ["# SQL verification results", "",
          f"Similarity backend: `{sim.backend}`. Thresholds tuned on the dev split only: "
          + ", ".join(f"{k} = {v:.3f}" for k, v in thresholds.items()) + " (balanced accuracy objective).",
          "", "ROC-AUC of the similarity score for separating correct from mismatched actions: "
          + "; ".join(f"{k}: question-vs-SQL {v['sim_sql']:.3f}, question-vs-meaning {v['sim_text']:.3f}"
                      for k, v in auc.items()) + " (0.5 = chance).", ""]
    for name, info in summary["sets"].items():
        md += [f"## {name} ({info['pairs']} pairs: {info['correct_actions']} correct, "
               f"{info['mismatched_actions']} mismatched; {info['mismatches_with_same_execution_result']} "
               f"mismatches return the same rows as the gold query)", "",
               "| Method | Accuracy | Precision | Recall | F1 | FAR ↓ | FRR ↓ |", "|---|---|---|---|---|---|---|"]
        for m in METHODS:
            sc = info["methods"][m]
            md.append(f"| {NAMES[m]} | {sc['accuracy']:.1%} | {sc['precision']:.1%} | {sc['recall']:.1%} | "
                      f"{sc['f1']:.1%} | {sc['far']:.1%} | {sc['frr']:.1%} |")
        md.append("")
    md += ["\\* needs the gold query's result, so it cannot be used at run time.", "",
           "## Detection rate per error type (all test sets)", "",
           "| Error type | n | Syntax | Exec non-empty | Similarity | Semmai | Exec-match oracle* |",
           "|---|---|---|---|---|---|---|"]
    for k in kinds:
        if per_type[k]["label"] == 0:
            p = per_type[k]
            md.append(f"| {k} | {p['n']} | {p['syntax_permission']:.0%} | {p['execution_nonempty']:.0%} | "
                      f"{p['similarity_text']:.0%} | {p['semmai']:.0%} | {p['exec_match_oracle']:.0%} |")
    md += ["", "## Acceptance rate of meaning-preserving rewrites", "", "| Rewrite | n | Similarity | Semmai |",
           "|---|---|---|---|"]
    for k in kinds:
        if per_type[k]["label"] == 1:
            p = per_type[k]
            md.append(f"| {k} | {p['n']} | {p['similarity_text']:.0%} | {p['semmai']:.0%} |")
    md += ["", f"Mean verification latency: {summary['latency_ms']['mean']:.2f} ms "
               f"(median {summary['latency_ms']['median']:.2f} ms, p95 {summary['latency_ms']['p95']:.2f} ms)."]
    (RES / "sql_results.md").write_text("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
