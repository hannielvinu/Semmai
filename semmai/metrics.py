"""Verification metrics.

Positive class = "semantic mismatch" (the action must be REJECTED).

    accuracy   = correct decisions / all decisions
    precision  = rejected mismatches / all rejections
    recall     = rejected mismatches / all mismatches      (detection rate)
    F1         = harmonic mean of precision and recall
    FAR        = accepted mismatches / all mismatches      (false acceptance rate:
                 wrong actions that would be executed - the safety metric)
    FRR        = rejected correct actions / all correct actions
                 (false rejection rate: the usability cost)
"""
from __future__ import annotations


def confusion(labels, allowed):
    """labels: 1 = action preserves intent, 0 = mismatch. allowed: verifier decision."""
    tp = sum(1 for y, a in zip(labels, allowed) if y == 0 and not a)    # mismatch rejected
    fn = sum(1 for y, a in zip(labels, allowed) if y == 0 and a)        # mismatch accepted
    fp = sum(1 for y, a in zip(labels, allowed) if y == 1 and not a)    # correct rejected
    tn = sum(1 for y, a in zip(labels, allowed) if y == 1 and a)        # correct accepted
    return tp, fp, fn, tn


def scores(labels, allowed) -> dict:
    tp, fp, fn, tn = confusion(labels, allowed)
    n = tp + fp + fn + tn
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"n": n, "accuracy": (tp + tn) / n if n else 0.0, "precision": prec, "recall": rec, "f1": f1,
            "far": fn / (tp + fn) if tp + fn else 0.0, "frr": fp / (fp + tn) if fp + tn else 0.0,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def best_threshold(sims, labels, objective="balanced"):
    """Choose t so that ALLOW = sim >= t maximises the objective on dev data."""
    cands = sorted(set(round(s, 4) for s in sims))
    best_t, best = 0.0, -1.0
    for t in [0.0] + cands + [1.01]:
        sc = scores(labels, [s >= t for s in sims])
        if objective == "balanced":
            v = (sc["recall"] + (1 - sc["frr"])) / 2
        else:
            v = sc[objective] if objective != "acc" else sc["accuracy"]
        if v > best:
            best, best_t = v, t
    return best_t
