"""Tamil evaluation.

1. Intent understanding - does Semmai's Tamil parser recover the exact meaning
   (same intent graph as the gold SQL)?  dev, dev2 and the frozen blind set.
2. Verification of SQL against Tamil requests - gold SQL + meaning-preserving
   rewrites (must ALLOW) and semantic mutations (must REJECT), checked against
   the intent graph Semmai built from the *Tamil* question.
3. Translation mistakes (the project story) - for every Tamil question and
   every applicable realistic translator mistake ("after" -> "from",
   dropped "except", ...): the SQL produced from the mistaken translation is
   checked by Semmai. How many are caught, and how many would have silently
   returned a different answer?
4. End-to-end chatbot - AI translator makes a mistake with probability p;
   compare Generate->Execute with Semmai (block + regenerate).

Usage: python scripts/evaluate_tamil.py
"""
from __future__ import annotations

import json
import random
import sqlite3
import sys
from collections import Counter, defaultdict
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from semmai.chat import MISTAKES, ChatPipeline, apply_mistake                 # noqa: E402
from semmai.intent import IntentExtractor                                     # noqa: E402
from semmai.metrics import scores                                             # noqa: E402
from semmai.mutations import MUTATIONS, REWRITES, Mutator                      # noqa: E402
from semmai.proof import prove                                                # noqa: E402
from semmai.schema import Schema                                              # noqa: E402
from semmai.similarity import Similarity                                      # noqa: E402
from semmai.sql_parser import render_sql, sql_to_usig                         # noqa: E402
from semmai.tamil import TamilParser, render_english                          # noqa: E402
from semmai.verbalize import verbalize                                        # noqa: E402

DATA = ROOT / "data"
RES = ROOT / "results"
SETS = {"dev": DATA / "tamil/dev.json", "dev2": DATA / "tamil/dev2.json", "blind": DATA / "tamil/blind_test.json"}

_ctx = {}


def ctx(db):
    if db not in _ctx:
        s = Schema.from_bird_style(DATA / "bird_style", db)
        _ctx[db] = {"schema": s, "tp": TamilParser(s), "ie": IntentExtractor(s), "pipe": ChatPipeline(s),
                    "con": sqlite3.connect(s.db_path)}
    return _ctx[db]


def tamil_intent(c, q):
    fr = c["tp"].parse(q)
    en = render_english(fr)
    g = c["ie"].extract(en)
    if fr.op == "delete":
        g.action, g.projections = "delete", []
    return fr, en, g


def rows_of(con, sql):
    s = sql.strip()
    if s.lower().startswith("delete"):
        s = "SELECT * FROM " + s[len("delete from "):]
    return con.execute(s).fetchall()


def same(a, b, ordered):
    if ordered:
        return a == b
    k = lambda r: tuple(map(str, r))
    return sorted(a, key=k) == sorted(b, key=k)


def main():
    RES.mkdir(exist_ok=True)
    sim = Similarity()
    out = {"intent": {}, "verification": {}, "mistakes": {}, "end_to_end": []}
    questions = {name: json.loads(p.read_text(encoding="utf-8")) for name, p in SETS.items()}

    # ---------------------------------------------------------- 1. intent
    for name, qs in questions.items():
        ok, by_db, fails = 0, Counter(), []
        for x in qs:
            c = ctx(x["db_id"])
            _, en, g = tamil_intent(c, x["question"])
            crit = [f for f in prove(g, sql_to_usig(x["SQL"], c["schema"]), c["schema"]) if f.severity == "critical"]
            if not crit:
                ok += 1
                by_db[x["db_id"]] += 1
            else:
                fails.append({"question": x["question"], "english": en, "gold": x["SQL"],
                              "why": [f.message for f in crit]})
        out["intent"][name] = {"n": len(qs), "exact": ok, "accuracy": ok / len(qs), "failures": fails}
    # the blind set was first run before the post-blind fix (see git history)
    out["intent"]["blind"]["first_run_exact"] = 49
    out["intent"]["blind"]["first_run_accuracy"] = 49 / 50

    # --------------------------------------------- 2. verification benchmark
    rng = random.Random(21)
    rows = []
    for name, qs in questions.items():
        for x in qs:
            c = ctx(x["db_id"])
            s = c["schema"]
            _, en, intent = tamil_intent(c, x["question"])
            gold = sql_to_usig(x["SQL"], s)
            m = Mutator(s, rng)
            cands = [(x["SQL"], 1, "GOLD")]
            for kind in rng.sample(REWRITES, len(REWRITES)):
                if gold.action == "delete" and kind not in ("REORDER_CONDITIONS", "IN_SINGLE", "LOWERCASE"):
                    continue
                r = m.rewrite(gold, kind)
                if r and r != x["SQL"]:
                    try:
                        if same(rows_of(c["con"], r), rows_of(c["con"], x["SQL"]), bool(gold.order_by)):
                            cands.append((r, 1, "REWRITE_" + kind))
                            break
                    except sqlite3.Error:
                        pass
            made = 0
            for kind in rng.sample(MUTATIONS, len(MUTATIONS)):
                if made >= 3:
                    break
                bad = m.mutate(gold, kind)
                if bad and bad != x["SQL"]:
                    if gold.action == "delete":
                        bad = bad.replace("SELECT * FROM", "DELETE FROM")
                    cands.append((bad, 0, kind))
                    made += 1
            for sql, label, kind in cands:
                try:
                    action = sql_to_usig(sql, s)
                    allow = not [f for f in prove(intent, action, s) if f.severity == "critical"]
                    text = verbalize(action, s.col_types())
                except Exception:  # noqa: BLE001
                    allow, text = False, sql
                rows.append({"set": name, "label": label, "kind": kind, "allow_semmai": allow,
                             "sim_tamil": sim.score(x["question"], text), "sim_english": sim.score(en, text),
                             "syntax_ok": True})
    # similarity thresholds tuned on dev, evaluated on blind
    from semmai.metrics import best_threshold
    dev = [r for r in rows if r["set"] in ("dev", "dev2")]
    t_ta = best_threshold([r["sim_tamil"] for r in dev], [r["label"] for r in dev])
    t_en = best_threshold([r["sim_english"] for r in dev], [r["label"] for r in dev])
    for subset in ("blind", "all"):
        rs = [r for r in rows if subset == "all" or r["set"] == subset]
        lab = [r["label"] for r in rs]
        out["verification"][subset] = {
            "pairs": len(rs), "correct": sum(lab), "mismatched": len(lab) - sum(lab),
            "methods": {"syntax_permission": scores(lab, [True] * len(rs)),
                        "similarity_tamil_vs_sql": scores(lab, [r["sim_tamil"] >= t_ta for r in rs]),
                        "similarity_translation_vs_sql": scores(lab, [r["sim_english"] >= t_en for r in rs]),
                        "semmai": scores(lab, [r["allow_semmai"] for r in rs])}}
    out["verification"]["thresholds"] = {"tamil": t_ta, "english": t_en}
    per_kind = defaultdict(list)
    for r in rows:
        per_kind[r["kind"]].append(r)
    out["verification"]["per_kind"] = {k: {"n": len(v), "label": v[0]["label"],
                                           "semmai": (sum(r["allow_semmai"] for r in v) if v[0]["label"] == 1 else
                                                      sum(not r["allow_semmai"] for r in v)) / len(v)}
                                       for k, v in sorted(per_kind.items())}

    # ------------------------------------------------ 3. translation mistakes
    mk = defaultdict(lambda: {"n": 0, "caught": 0, "silent_wrong": 0, "harmless": 0})
    for name, qs in questions.items():
        for x in qs:
            c = ctx(x["db_id"])
            s = c["schema"]
            fr, en, intent = tamil_intent(c, x["question"])
            ref = render_sql(intent, s)
            ref_rows = rows_of(c["con"], ref)
            for kind in MISTAKES:
                bad_fr, got = apply_mistake(fr, random.Random(1), preferred=kind)
                if got != kind:
                    continue
                bad_en = render_english(bad_fr)
                bad_sql = render_sql(c["ie"].extract(bad_en), s)
                if fr.op == "delete":
                    bad_sql = bad_sql.replace("SELECT name FROM", "DELETE FROM")
                action = sql_to_usig(bad_sql, s)
                caught = bool([f for f in prove(intent, action, s) if f.severity == "critical"])
                diff = not same(rows_of(c["con"], bad_sql), ref_rows, bool(intent.order_by))
                d = mk[kind]
                d["n"] += 1
                d["caught"] += caught
                d["silent_wrong"] += diff
                d["harmless"] += (not diff)
    tot = {"n": sum(d["n"] for d in mk.values()), "caught": sum(d["caught"] for d in mk.values()),
           "silent_wrong": sum(d["silent_wrong"] for d in mk.values())}
    out["mistakes"] = {"per_kind": dict(mk), "total": tot}

    # ------------------------------------------------------ 4. end-to-end
    allq = [x for qs in questions.values() for x in qs]
    for p in (0.2, 0.4):
        agg = Counter()
        for seed in range(3):
            rng = random.Random(100 + seed)
            for x in allq:
                c = ctx(x["db_id"])
                s = c["schema"]
                fr, en, intent = tamil_intent(c, x["question"])
                ref_rows = rows_of(c["con"], render_sql(intent, s))

                def ai_sql():
                    f2, kind = (apply_mistake(fr, rng) if rng.random() < p else (fr, None))
                    q = render_sql(c["ie"].extract(render_english(f2)), s)
                    return q.replace("SELECT name FROM", "DELETE FROM") if fr.op == "delete" else q

                agg["n"] += 1
                sql = ai_sql()
                right = same(rows_of(c["con"], sql), ref_rows, bool(intent.order_by))
                agg["base_correct"] += right
                agg["base_wrong"] += not right
                # Semmai: verify, regenerate up to 3 attempts, then repair from the intent graph
                for attempt in range(3):
                    ok = not [f for f in prove(intent, sql_to_usig(sql, s), s) if f.severity == "critical"]
                    if ok:
                        break
                    agg["blocked"] += 1
                    sql = ai_sql()
                if not ok:
                    sql = render_sql(intent, s)
                    agg["repaired"] += 1
                right = same(rows_of(c["con"], sql), ref_rows, bool(intent.order_by))
                agg["semmai_correct"] += right
                agg["semmai_wrong"] += not right
        n = agg["n"]
        out["end_to_end"].append({"p": p, "runs": n, "without_semmai": {"correct": agg["base_correct"] / n,
                                                                       "wrong": agg["base_wrong"] / n},
                                  "with_semmai": {"correct": agg["semmai_correct"] / n, "wrong": agg["semmai_wrong"] / n,
                                                  "blocked_attempts": agg["blocked"], "repaired": agg["repaired"]}})

    (RES / "tamil_results.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")

    # ------------------------------------------------------------ markdown
    I = out["intent"]
    md = ["# Tamil results", "",
          "## 1. Understanding Tamil requests (exact intent-graph match with the gold SQL)", "",
          "| Set | Questions | Exact meaning |", "|---|---|---|",
          f"| Development (dev) | {I['dev']['n']} | {I['dev']['accuracy']:.1%} |",
          f"| Development (dev2: mixed Tamil-English, number words, ₹) | {I['dev2']['n']} | {I['dev2']['accuracy']:.1%} |",
          f"| **Blind test, first run** (frozen in git before the parser existed) | 50 | **{I['blind']['first_run_accuracy']:.1%}** |",
          f"| Blind test after one post-hoc fix | {I['blind']['n']} | {I['blind']['accuracy']:.1%} |", "",
          "## 2. Verifying SQL against a Tamil request", "",
          f"Gold SQL and meaning-preserving rewrites must be allowed; semantic mutations must be rejected. "
          f"Thresholds for the similarity baselines were tuned on the development sets.", ""]
    for subset in ("blind", "all"):
        V = out["verification"][subset]
        md += [f"**{'Blind set' if subset == 'blind' else 'All Tamil sets'}** - {V['pairs']} SQL candidates "
               f"({V['correct']} correct, {V['mismatched']} wrong)", "",
               "| Method | Accuracy | F1 | Wrong SQL let through (FAR) | Correct SQL blocked (FRR) |", "|---|---|---|---|---|"]
        names = {"syntax_permission": "Syntax + permission check", "similarity_tamil_vs_sql":
                 "Similarity: Tamil question vs SQL meaning", "similarity_translation_vs_sql":
                 "Similarity: English translation vs SQL meaning", "semmai": "**Semmai**"}
        for m, nm in names.items():
            sc = V["methods"][m]
            md.append(f"| {nm} | {sc['accuracy']:.1%} | {sc['f1']:.1%} | {sc['far']:.1%} | {sc['frr']:.1%} |")
        md.append("")
    M = out["mistakes"]
    md += ["## 3. Translation mistakes (the project story)", "",
           "Every Tamil question x every applicable realistic translator mistake. *Silently wrong* = the mistaken "
           "SQL returns a different answer than the user asked for.", "",
           "| Mistake | Cases | Caught by Semmai | Would silently return a wrong answer |", "|---|---|---|---|"]
    for kind, d in sorted(M["per_kind"].items(), key=lambda kv: -kv[1]["n"]):
        md.append(f"| {kind.replace('_', ' ').lower()} | {d['n']} | {d['caught'] / d['n']:.0%} | {d['silent_wrong'] / d['n']:.0%} |")
    md.append(f"| **All** | {M['total']['n']} | **{M['total']['caught'] / M['total']['n']:.1%}** | "
              f"{M['total']['silent_wrong'] / M['total']['n']:.0%} |")
    md += ["", "## 4. End-to-end Tamil chatbot (3 seeds over all 126 questions)", "",
           "| AI mistake rate | Pipeline | Correct answers | Wrong answers executed |", "|---|---|---|---|"]
    for e in out["end_to_end"]:
        md.append(f"| {e['p']:.0%} | Translate → SQL → Execute | {e['without_semmai']['correct']:.1%} | {e['without_semmai']['wrong']:.1%} |")
        md.append(f"| {e['p']:.0%} | + Semmai (verify, regenerate, repair) | **{e['with_semmai']['correct']:.1%}** | "
                  f"**{e['with_semmai']['wrong']:.1%}** |")
    (RES / "tamil_results.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
