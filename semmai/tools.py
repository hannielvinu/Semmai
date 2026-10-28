"""Semantic verification of tool / API actions (MASSIVE-style intents + slots).

An agent turns a user utterance (any language) into a tool call:

    "நாளை காலை 7 மணி க்கு அலாரம் வை"  ->  {"intent": "alarm_set",
                                            "slots": {"date": "நாளை", "time": "காலை 7 மணி"}}

``ToolVerifier`` checks that call against the utterance before it runs:

    INTENT       a multilingual intent model (character n-gram TF-IDF +
                 logistic regression, trained on the train split) must agree
                 that the utterance asks for the proposed intent
    GROUNDING    every slot value must actually occur in the utterance
                 (catches hallucinated or drifted values: 7 am -> 8 am)
    SLOT SCHEMA  slot types must be valid for the intent
    COVERAGE     slot values that a gazetteer (learned from training
                 annotations) finds in the utterance must not be dropped
    TYPE         a value must sit in the right slot (time not in date)
"""
from __future__ import annotations

import json
import math
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from .proof import Finding
from .textfeat import char_wb_ngrams

SLOT_RE = re.compile(r"\[\s*([\w_]+)\s*:\s*([^\]]+?)\s*\]")


def norm(s: str) -> str:
    s = unicodedata.normalize("NFKC", s).casefold()
    s = re.sub(r"[\"'’`,.;:!?¿¡()\[\]]", " ", s)
    return " " + re.sub(r"\s+", " ", s).strip() + " "


def parse_annot(annot: str) -> dict:
    """'wake me up at [time : 7 am]' -> {'time': '7 am'}  (first value per slot type)."""
    slots = {}
    for k, v in SLOT_RE.findall(annot):
        slots.setdefault(k, v.strip())
    return slots


def load_massive(path: str | Path, locales=None) -> list:
    """Load MASSIVE-format jsonl: one file (Semmai's set) or a folder of per-locale
    files (official MASSIVE 1.1: data/<locale>.jsonl)."""
    path = Path(path)
    if path.is_dir():
        out = []
        for f in sorted(path.glob("*.jsonl")):
            if locales and f.stem not in locales:
                continue
            out.extend(load_massive(f, locales))
        return out
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if locales and r["locale"] not in locales:
            continue
        r["slots"] = parse_annot(r.get("annot_utt", ""))
        r.setdefault("template", int(r["id"]) % 3 if str(r["id"]).isdigit() else hash(r["id"]) % 3)
        out.append(r)
    return out


@dataclass
class ToolVerdict:
    decision: str
    findings: list = field(default_factory=list)
    intent_probs: dict = field(default_factory=dict)

    @property
    def allowed(self):
        return self.decision == "ALLOW"


class ToolVerifier:
    def __init__(self, intent_threshold: float = 0.35, numeric_coverage: bool = True):
        self.intent_threshold = intent_threshold
        self.numeric_coverage = numeric_coverage
        self.clf = None                  # scikit-learn pipeline (training) ...
        self._plain = None               # ... or exported weights (plain-Python inference)
        self.allowed_slots = defaultdict(set)
        self.gazetteer = {}          # normalised value -> slot type
        self.intents = []

    # ------------------------------------------------------------- training
    def fit(self, records: list) -> "ToolVerifier":
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        self.clf = make_pipeline(
            TfidfVectorizer(analyzer="char_wb", ngram_range=(1, 4), sublinear_tf=True, min_df=1),
            LogisticRegression(C=20, max_iter=3000))
        X = [r["utt"] for r in records]
        y = [r["intent"] for r in records]
        self.clf.fit(X, y)
        self.intents = list(self.clf.classes_)
        for r in records:
            for k, v in r["slots"].items():
                self.allowed_slots[r["intent"]].add(k)
                self.gazetteer[norm(v).strip()] = k
        # longest values first for greedy matching
        self._gaz = sorted(self.gazetteer.items(), key=lambda kv: -len(kv[0]))
        return self

    def intent_probs(self, utt: str) -> dict:
        if self._plain is not None:
            return self._plain_probs(utt)
        p = self.clf.predict_proba([utt])[0]
        return dict(zip(self.intents, (float(x) for x in p)))

    def predict_intent(self, utt: str) -> str:
        probs = self.intent_probs(utt)
        return max(probs, key=probs.get)

    # ------------------------------------------------- export / plain-Python load
    def export(self, path) -> None:
        """Save the trained model as JSON so it can run without scikit-learn."""
        vec, lr = self.clf.steps[0][1], self.clf.steps[1][1]
        vocab = sorted(vec.vocabulary_, key=vec.vocabulary_.get)
        idx = [vec.vocabulary_[t] for t in vocab]
        data = {"format": "semmai-intent-v1", "classes": self.intents,
                "intent_threshold": self.intent_threshold, "numeric_coverage": self.numeric_coverage,
                "vocab": vocab, "idf": [round(float(vec.idf_[i]), 5) for i in idx],
                "coef": [[round(float(lr.coef_[c][i]), 4) for i in idx] for c in range(len(self.intents))],
                "intercept": [round(float(b), 5) for b in lr.intercept_],
                "allowed_slots": {k: sorted(v) for k, v in self.allowed_slots.items()},
                "gazetteer": self.gazetteer}
        Path(path).write_text(json.dumps(data, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")

    @classmethod
    def load(cls, path) -> "ToolVerifier":
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        v = cls(intent_threshold=d["intent_threshold"], numeric_coverage=d["numeric_coverage"])
        v.intents = d["classes"]
        index = {t: i for i, t in enumerate(d["vocab"])}
        v._plain = {"index": index, "idf": d["idf"], "coef": d["coef"], "intercept": d["intercept"]}
        v.allowed_slots = defaultdict(set, {k: set(x) for k, x in d["allowed_slots"].items()})
        v.gazetteer = d["gazetteer"]
        v._gaz = sorted(v.gazetteer.items(), key=lambda kv: -len(kv[0]))
        return v

    def _plain_probs(self, utt: str) -> dict:
        P = self._plain
        counts = Counter(t for t in char_wb_ngrams(utt.lower(), 1, 4) if t in P["index"])
        feats = {P["index"][t]: (1.0 + math.log(c)) * P["idf"][P["index"][t]] for t, c in counts.items()}
        n = math.sqrt(sum(x * x for x in feats.values())) or 1.0
        z = [b + sum(w[i] * x / n for i, x in feats.items()) for w, b in zip(P["coef"], P["intercept"])]
        m = max(z)
        e = [math.exp(x - m) for x in z]
        s = sum(e)
        return {c: x / s for c, x in zip(self.intents, e)}

    # ------------------------------------------------------------ verify
    def detect_values(self, utt: str) -> list:
        text = norm(utt)
        found = []
        for v, typ in self._gaz:
            pat = f" {v} "
            idx = text.find(pat)
            if idx >= 0:
                found.append((v, typ))
                text = text[:idx] + " " + "#" * len(v) + " " + text[idx + len(pat):]
        return found

    def verify(self, utt: str, action: dict, probs: dict | None = None) -> ToolVerdict:
        F = []
        intent = action.get("intent")
        slots = action.get("slots", {}) or {}
        probs = probs if probs is not None else self.intent_probs(utt)
        top = max(probs, key=probs.get)
        p_prop = probs.get(intent, 0.0)
        if intent not in probs:
            F.append(Finding("UNKNOWN_INTENT", "critical", f"'{intent}' is not a known tool/intent"))
        elif intent != top and p_prop < self.intent_threshold:
            F.append(Finding("INTENT_MISMATCH", "critical",
                             f"utterance asks for '{top}' (p={probs[top]:.2f}), action performs '{intent}' "
                             f"(p={p_prop:.2f})"))
        text = norm(utt)
        for k, v in slots.items():
            if k not in self.allowed_slots.get(intent, set()):
                F.append(Finding("SLOT_NOT_ALLOWED", "critical", f"slot '{k}' is not valid for '{intent}'"))
            nv = norm(str(v)).strip()
            if not nv or f" {nv} " not in text:
                F.append(Finding("UNGROUNDED_VALUE", "critical",
                                 f"slot {k}='{v}' does not appear in the request (value drift / hallucination)"))
            else:
                gt = self.gazetteer.get(nv)
                if gt and gt != k:
                    F.append(Finding("SLOT_TYPE_MISMATCH", "critical", f"'{v}' is a {gt}, not a {k}"))
        given = {norm(str(v)).strip() for v in slots.values()}
        for v, typ in self.detect_values(utt):
            if v not in given and typ in self.allowed_slots.get(intent, set()):
                F.append(Finding("SLOT_MISSING", "critical", f"request mentions {typ} '{v.strip()}' but the action drops it"))
        if self.numeric_coverage:
            # every number in the request (times, amounts, counts) must be carried by some slot
            covered = " ".join(" " + str(v).casefold() + " " for v in slots.values())
            for num in set(re.findall(r"\d+(?:[:.]\d+)?", utt)):
                if not re.search(rf"(?<![\d:.]){re.escape(num)}(?![\d:.])", covered) and \
                        not any(f.kind == "SLOT_MISSING" and num in f.message for f in F):
                    F.append(Finding("NUMBER_DROPPED", "critical",
                                     f"the request contains '{num}' but no slot of the action carries it"))
        return ToolVerdict("REJECT" if any(f.severity == "critical" for f in F) else "ALLOW", F, probs)


def verbalize_action(action: dict) -> str:
    return action.get("intent", "").replace("_", " ") + " " + " ".join(
        f"{k.replace('_', ' ')} {v}" for k, v in (action.get("slots") or {}).items())
