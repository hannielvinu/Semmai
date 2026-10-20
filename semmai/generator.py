"""SQL generators (the "LLM" in Generate -> Verify -> Execute).

Semmai is model-agnostic: anything that maps (question, schema) -> SQL can
be plugged in.

* ``OllamaGenerator``  - a real LLM, e.g. Qwen2.5-Coder served locally by
                          Ollama (``ollama pull qwen2.5-coder:7b``).
* ``SimulatedLLM``      - reproducible stand-in used in the experiments: it
                          returns the correct query, or - with a given error
                          rate - a query containing one realistic semantic
                          error (the same error taxonomy as the benchmark).
"""
from __future__ import annotations

import json
import random
import re
import urllib.request
from typing import Optional

from .mutations import MUTATIONS, Mutator
from .schema import Schema
from .sql_parser import sql_to_usig

# relative frequency of error types (value / operator / missing-constraint errors are most common)
ERROR_WEIGHTS = {"VALUE_DRIFT": 3, "OPERATOR_BOUNDARY": 3, "CONSTRAINT_DROP": 3, "DATE_DRIFT": 2,
                 "COLUMN_SWAP": 2, "CONSTRAINT_ADD": 2, "NEGATION_FLIP": 1, "OPERATOR_REVERSE": 1,
                 "AGGREGATION_SWAP": 1, "ORDER_FLIP": 1, "LIMIT_CHANGE": 1, "LOGIC_AND_TO_OR": 1,
                 "PROJECTION_SWAP": 1, "ENTITY_SWAP": 1, "GROUP_SWAP": 1}


class SimulatedLLM:
    name = "simulated"

    def __init__(self, error_rate: float = 0.25, seed: int = 0):
        self.error_rate = error_rate
        self.rng = random.Random(seed)

    def generate(self, question: str, schema: Schema, gold_sql: str) -> tuple:
        """Returns (sql, injected_error_or_None)."""
        if self.rng.random() >= self.error_rate:
            return gold_sql, None
        g = sql_to_usig(gold_sql, schema)
        mut = Mutator(schema, self.rng)
        kinds = list(ERROR_WEIGHTS)
        for _ in range(10):
            kind = self.rng.choices(kinds, weights=[ERROR_WEIGHTS[k] for k in kinds])[0]
            sql = mut.mutate(g, kind)
            if sql and sql != gold_sql:
                return sql, kind
        return gold_sql, None


PROMPT = """You are an expert SQLite developer. Given the database schema, write ONE SQLite query that answers the question.
Return only the SQL, no explanation.

Schema:
{schema}

Question: {question}
SQL:"""


class OllamaGenerator:
    """Real LLM backend through a local Ollama server (default model: Qwen2.5-Coder)."""

    def __init__(self, model: str = "qwen2.5-coder:7b", host: str = "http://localhost:11434",
                 temperature: float = 0.0):
        self.model, self.host, self.temperature = model, host.rstrip("/"), temperature
        self.name = f"ollama:{model}"

    def generate(self, question: str, schema: Schema, gold_sql: Optional[str] = None) -> tuple:
        body = json.dumps({"model": self.model, "prompt": PROMPT.format(schema=schema.describe(), question=question),
                           "stream": False, "options": {"temperature": self.temperature}}).encode()
        req = urllib.request.Request(f"{self.host}/api/generate", data=body,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=120) as r:
            text = json.loads(r.read())["response"]
        return extract_sql(text), None


def extract_sql(text: str) -> str:
    m = re.search(r"```(?:sql)?\s*(.+?)```", text, re.S | re.I)
    if m:
        text = m.group(1)
    m = re.search(r"(select\b.+)", text, re.S | re.I)
    sql = m.group(1) if m else text
    return sql.strip().split(";")[0].strip()
