"""Semmai verifier: the semantic safety layer between generation and execution.

    question ──► IntentExtractor ──► intent USIG ─┐
                                                   ├─► proof engine ─┐
    SQL ───────► SQL compiler ─────► action USIG ─┘                  ├─► ALLOW / REJECT
                     └──► verbalize ──► similarity(question, text) ──┘

Decision rule
    REJECT if the action cannot be compiled (fail-closed),
           if any critical proof obligation fails,
           or if neural similarity is below the threshold.
    ALLOW  otherwise.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

from .intent import IntentExtractor
from .proof import Finding, prove
from .schema import Schema
from .similarity import Similarity
from .sql_parser import UnsupportedSQL, sql_to_usig
from .usig import USIG
from .verbalize import verbalize


@dataclass
class Verdict:
    decision: str                          # ALLOW | REJECT
    findings: list = field(default_factory=list)
    similarity: Optional[float] = None
    intent: Optional[USIG] = None
    action: Optional[USIG] = None
    action_text: str = ""
    latency_ms: float = 0.0

    @property
    def allowed(self) -> bool:
        return self.decision == "ALLOW"

    def critical(self) -> list:
        return [f for f in self.findings if f.severity == "critical"]

    def to_dict(self) -> dict:
        return {"decision": self.decision, "similarity": self.similarity,
                "findings": [f.to_dict() for f in self.findings],
                "intent": self.intent.to_dict() if self.intent else None,
                "action": self.action.to_dict() if self.action else None,
                "action_text": self.action_text, "latency_ms": round(self.latency_ms, 2)}


class SemmaiVerifier:
    def __init__(self, schema: Schema, similarity: Optional[Similarity] = None,
                 sim_threshold: float = 0.0, use_proof: bool = True, use_similarity: bool = True):
        self.schema = schema
        self.extractor = IntentExtractor(schema)
        self.sim = similarity or (Similarity() if use_similarity else None)
        self.sim_threshold = sim_threshold
        self.use_proof = use_proof
        self.use_similarity = use_similarity
        self._types = schema.col_types()

    def verify(self, question: str, sql: str) -> Verdict:
        t0 = time.perf_counter()
        intent = self.extractor.extract(question)
        try:
            action = sql_to_usig(sql, self.schema)
        except UnsupportedSQL as e:
            return Verdict("REJECT", [Finding("UNVERIFIABLE", "critical", f"action could not be analysed: {e}")],
                           None, intent, None, "", (time.perf_counter() - t0) * 1000)
        text = verbalize(action, self._types)
        findings = prove(intent, action, self.schema) if self.use_proof else []
        sim = self.sim.score(question, text) if self.sim is not None else None
        decision = "ALLOW"
        if any(f.severity == "critical" for f in findings):
            decision = "REJECT"
        if self.use_similarity and sim is not None and sim < self.sim_threshold:
            findings.append(Finding("LOW_SIMILARITY", "critical",
                                    f"meaning of the action is not similar enough to the request "
                                    f"(score {sim:.2f} < {self.sim_threshold:.2f})"))
            decision = "REJECT"
        return Verdict(decision, findings, sim, intent, action, text, (time.perf_counter() - t0) * 1000)
