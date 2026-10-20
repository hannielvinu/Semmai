"""Evaluate multilingual tool-action verification (MASSIVE format).

Benchmark: for every TEST utterance the agent's tool call is either
    correct            gold action, or the gold action re-serialised
                       (slot order / letter case)                       label 1
    semantically wrong INTENT_SWAP, SLOT_VALUE_DRIFT, SLOT_DROP,
                       SLOT_ADD, SLOT_TYPE_SWAP                        label 0

Methods
    schema_validation  well-formed call, known tool, known slot names
    similarity         char n-gram similarity(utterance, verbalised call) >= t
    semmai_intent_only Semmai intent check only            (ablation)
    semmai_slots_only  Semmai grounding/coverage/type only (ablation)
    semmai             full Semmai tool verifier

Thresholds are tuned on cross-validated TRAIN data (held-out templates).

Usage: python scripts/evaluate_tools.py [--data data/massive_style/massive_style.jsonl]
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.model_selection import GroupKFold, cross_val_predict

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from semmai.metrics import best_threshold, scores                     # noqa: E402
from semmai.similarity import Similarity                              # noqa: E402
from semmai.tools import ToolVerifier, load_massive, verbalize_action  # noqa: E402

RES = ROOT / "results"
CONFUSABLE = {"alarm_set": "alarm_remove", "alarm_remove": "alarm_set", "iot_hue_lightoff": "iot_hue_lighton",
              "iot_hue_lighton": "iot_hue_lightoff", "calendar_set": "alarm_set", "weather_query": "news_query",
              "news_query": "weather_query", "qa_currency": "news_query", "play_music": "news_query",
              "takeaway_order": "transport_ticket", "transport_ticket": "takeaway_order",
              "email_sendemail": "calendar_set"}
MUTS = ["INTENT_SWAP", "SLOT_VALUE_DRIFT", "SLOT_DROP", "SLOT_ADD", "SLOT_TYPE_SWAP"]
METHODS = ["schema_validation", "similarity", "semmai_intent_only", "semmai_slots_only", "semmai_no_numeric",
           "semmai"]
NAMES = {"schema_validation": "Schema validation", "similarity": "Similarity (utterance vs call)",
         "semmai_intent_only": "Semmai: intent check only", "semmai_slots_only": "Semmai: slot checks only",
         "semmai_no_numeric": "Semmai without numeric-coverage rule", "semmai": "Semmai (full)"}


def build_pairs(records, values_by_loc_type, intent_slots, rng):
    pairs = []
    usage = defaultdict(int)
    for r in records:
        gold = {"intent": r["intent"], "slots": dict(r["slots"])}
        base = {"id": r["id"], "locale": r["locale"], "utt": r["utt"]}
        pairs.append(dict(base, action=gold, label=1, kind="GOLD"))
        rw = {"intent": gold["intent"], "slots": dict(reversed(list(gold["slots"].items())))}
        if rw["slots"]:
            k = rng.choice(list(rw["slots"]))
            rw["slots"][k] = rw["slots"][k].upper() if rng.random() < 0.5 else rw["slots"][k].title()
        pairs.append(dict(base, action=rw, label=1, kind="REWRITE_FORMAT"))
        made = 0
        for kind in sorted(MUTS, key=lambda k: (usage[k], rng.random())):
            if made >= 2:
                break
            a = {"intent": gold["intent"], "slots": dict(gold["slots"])}
            if kind == "INTENT_SWAP":
                a["intent"] = CONFUSABLE[gold["intent"]] if rng.random() < 0.7 else rng.choice(
                    [i for i in intent_slots if i != gold["intent"]])
            elif kind == "SLOT_VALUE_DRIFT":
                if not a["slots"]:
                    continue
                k = rng.choice(list(a["slots"]))
                opts = [v for v in values_by_loc_type[(r["locale"], k)] if v.casefold() not in r["utt"].casefold()]
                if not opts:
                    continue
                a["slots"][k] = rng.choice(opts)
            elif kind == "SLOT_DROP":
                if not a["slots"]:
                    continue
                del a["slots"][rng.choice(list(a["slots"]))]
            elif kind == "SLOT_ADD":
                free = [t for t in intent_slots[gold["intent"]] if t not in a["slots"]]
                if not free:
                    free = [t for t in ("date", "time", "place_name") if t not in a["slots"]]
                t = rng.choice(free)
                opts = [v for v in values_by_loc_type[(r["locale"], t)] if v.casefold() not in r["utt"].casefold()]
                if not opts:
                    continue
                a["slots"][t] = rng.choice(opts)
            elif kind == "SLOT_TYPE_SWAP":
                if not a["slots"]:
                    continue
                k = rng.choice(list(a["slots"]))
                others = [t for t in intent_slots[gold["intent"]] if t != k and t not in a["slots"]] or \
                         [t for t in ("date", "time", "place_name", "person") if t != k and t not in a["slots"]]
                a["slots"][rng.choice(others)] = a["slots"].pop(k)
            pairs.append(dict(base, action=a, label=0, kind=kind))
            usage[kind] += 1
            made += 1
    return pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data/massive_style/massive_style.jsonl"))
    args = ap.parse_args()
    rng = random.Random(5)
    recs = load_massive(args.data)
    train = [r for r in recs if r["partition"] == "train"]
    test = [r for r in recs if r["partition"] == "test"]
    values = defaultdict(set)
    intent_slots = defaultdict(set)
    for r in recs:
        for k, v in r["slots"].items():
            values[(r["locale"], k)].add(v)
            intent_slots[r["intent"]].add(k)
    values = {k: sorted(v) for k, v in values.items()}
    global_slots = {k for s in intent_slots.values() for k in s}

    # ---------------- dev: cross-validated intent probabilities on train (grouped by template)
    groups = [f"{r['locale']}-{r['template']}" for r in train]
    vtmp = ToolVerifier().fit(train)
    cv_probs = cross_val_predict(vtmp.clf, [r["utt"] for r in train], [r["intent"] for r in train],
                                 cv=GroupKFold(n_splits=3), groups=[r["template"] for r in train],
                                 method="predict_proba")
    classes = sorted({r["intent"] for r in train})
    dev_pairs = build_pairs(train, values, intent_slots, random.Random(11))
    prob_of = {r["id"]: dict(zip(classes, p)) for r, p in zip(train, cv_probs)}
    sim = Similarity(backend="tfidf")
    # tune intent threshold on dev
    best_t, best_v = 0.35, -1
    for t in np.linspace(0.05, 0.95, 19):
        v = ToolVerifier(intent_threshold=t)
        v.clf, v.intents = vtmp.clf, vtmp.intents
        v.allowed_slots, v.gazetteer, v._gaz = vtmp.allowed_slots, vtmp.gazetteer, vtmp._gaz
        al = [v.verify(p["utt"], p["action"], prob_of[p["id"]]).allowed for p in dev_pairs]
        sc = scores([p["label"] for p in dev_pairs], al)
        bal = (sc["recall"] + 1 - sc["frr"]) / 2
        if bal > best_v + 1e-9:
            best_v, best_t = bal, float(t)
    sim_dev = [sim.score(p["utt"], verbalize_action(p["action"])) for p in dev_pairs]
    t_sim = best_threshold(sim_dev, [p["label"] for p in dev_pairs])
    print(f"intent threshold (dev) = {best_t:.2f}; similarity threshold (dev) = {t_sim:.3f}")

    # ---------------- final verifier trained on all train data
    ver = ToolVerifier(intent_threshold=best_t).fit(train)
    intent_acc = defaultdict(lambda: [0, 0])
    for r in test:
        ok = ver.predict_intent(r["utt"]) == r["intent"]
        intent_acc[r["locale"]][0] += ok
        intent_acc[r["locale"]][1] += 1
        intent_acc["all"][0] += ok
        intent_acc["all"][1] += 1
    pairs = build_pairs(test, values, intent_slots, rng)
    rows = []
    for p in pairs:
        probs = ver.intent_probs(p["utt"])
        verdict = ver.verify(p["utt"], p["action"], probs)
        kinds = {f.kind for f in verdict.findings}
        a = p["action"]
        row = {**p, "action": json.dumps(a, ensure_ascii=False),
               "findings": "; ".join(f.message for f in verdict.findings),
               "allow_schema_validation": a.get("intent") in intent_slots and all(k in global_slots for k in a["slots"]),
               "allow_similarity": sim.score(p["utt"], verbalize_action(a)) >= t_sim,
               "allow_semmai_intent_only": not (kinds & {"INTENT_MISMATCH", "UNKNOWN_INTENT"}),
               "allow_semmai_slots_only": not (kinds - {"INTENT_MISMATCH", "UNKNOWN_INTENT"}),
               "allow_semmai_no_numeric": not (kinds - {"NUMBER_DROPPED"}),
               "allow_semmai": verdict.allowed}
        rows.append(row)

    seen_vals = {v for r in train for v in r["slots"].values()}
    labels = [r["label"] for r in rows]
    summary = {"thresholds": {"intent": best_t, "similarity": t_sim},
               "data": {"train_utterances": len(train), "test_utterances": len(test), "pairs": len(rows),
                        "correct": sum(labels), "mismatched": len(labels) - sum(labels),
                        "unseen_slot_value_test_utterances": sum(
                            1 for r in test if any(v not in seen_vals for v in r["slots"].values()))},
               "intent_accuracy": {k: v[0] / v[1] for k, v in intent_acc.items()},
               "overall": {m: scores(labels, [r[f"allow_{m}"] for r in rows]) for m in METHODS},
               "per_locale": {}, "per_type": {}}
    for loc in sorted({r["locale"] for r in rows}):
        rs = [r for r in rows if r["locale"] == loc]
        summary["per_locale"][loc] = {m: scores([r["label"] for r in rs], [r[f"allow_{m}"] for r in rs])
                                      for m in METHODS}
    for k in ["GOLD", "REWRITE_FORMAT"] + MUTS:
        rs = [r for r in rows if r["kind"] == k]
        lab = rs[0]["label"]
        summary["per_type"][k] = {"n": len(rs), "label": lab, **{
            m: (sum(r[f"allow_{m}"] for r in rs) if lab == 1 else sum(not r[f"allow_{m}"] for r in rs)) / len(rs)
            for m in METHODS}}
    RES.mkdir(exist_ok=True)
    (RES / "tools_results.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    with open(RES / "tools_predictions.csv", "w", newline="", encoding="utf-8") as f:
        keys = ["id", "locale", "utt", "action", "label", "kind", "findings"] + [f"allow_{m}" for m in METHODS]
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    d = summary["data"]
    md = ["# Multilingual tool-action verification results", "",
          f"{d['train_utterances']} training utterances, {d['test_utterances']} test utterances "
          f"(held-out templates; {d['unseen_slot_value_test_utterances']} contain slot values never seen in training) "
          f"-> {d['pairs']} action pairs ({d['correct']} correct, {d['mismatched']} mismatched). "
          f"Thresholds tuned on cross-validated training data: intent {best_t:.2f}, similarity {t_sim:.3f}.", "",
          "| Method | Accuracy | Precision | Recall | F1 | FAR ↓ | FRR ↓ |", "|---|---|---|---|---|---|---|"]
    for m in METHODS:
        sc = summary["overall"][m]
        md.append(f"| {NAMES[m]} | {sc['accuracy']:.1%} | {sc['precision']:.1%} | {sc['recall']:.1%} | {sc['f1']:.1%} "
                  f"| {sc['far']:.1%} | {sc['frr']:.1%} |")
    md += ["", "## Per language (Semmai full)", "",
           "| Locale | Intent model accuracy | Accuracy | F1 | FAR ↓ | FRR ↓ | Similarity F1 |", "|---|---|---|---|---|---|---|"]
    for loc, ms in summary["per_locale"].items():
        sc = ms["semmai"]
        md.append(f"| {loc} | {summary['intent_accuracy'][loc]:.1%} | {sc['accuracy']:.1%} | {sc['f1']:.1%} | "
                  f"{sc['far']:.1%} | {sc['frr']:.1%} | {ms['similarity']['f1']:.1%} |")
    md += ["", "## Per error type (detection rate; acceptance rate for correct calls)", "",
           "| Type | n | Schema | Similarity | Intent only | Slots only | No numeric rule | Semmai |",
           "|---|---|---|---|---|---|---|---|"]
    for k, v in summary["per_type"].items():
        md.append(f"| {k} | {v['n']} | {v['schema_validation']:.0%} | {v['similarity']:.0%} | "
                  f"{v['semmai_intent_only']:.0%} | {v['semmai_slots_only']:.0%} | {v['semmai_no_numeric']:.0%} | "
                  f"{v['semmai']:.0%} |")
    (RES / "tools_results.md").write_text("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
