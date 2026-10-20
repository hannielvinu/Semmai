"""Agent execution pipeline.

    Without Semmai:  Generate -> Execute
    With Semmai:     Generate -> Verify -> Execute | Reject (-> regenerate, up to k attempts)
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from typing import Optional

from .schema import Schema
from .verifier import SemmaiVerifier


@dataclass
class PipelineResult:
    executed_sql: Optional[str]
    rows: Optional[list]
    attempts: int
    verdicts: list = field(default_factory=list)
    status: str = ""            # executed | rejected


class AgentPipeline:
    def __init__(self, schema: Schema, generator, verifier: Optional[SemmaiVerifier] = None,
                 max_attempts: int = 1):
        self.schema = schema
        self.gen = generator
        self.verifier = verifier
        self.max_attempts = max_attempts

    def run(self, question: str, gold_sql: Optional[str] = None) -> PipelineResult:
        verdicts = []
        for attempt in range(1, self.max_attempts + 1):
            sql, _ = self.gen.generate(question, self.schema, gold_sql)
            if self.verifier is not None:
                v = self.verifier.verify(question, sql)
                verdicts.append(v)
                if not v.allowed:
                    continue
            con = sqlite3.connect(self.schema.db_path)
            try:
                rows = con.execute(sql).fetchall()
            finally:
                con.close()
            return PipelineResult(sql, rows, attempt, verdicts, "executed")
        return PipelineResult(None, None, self.max_attempts, verdicts, "rejected")
