"""Universal Semantic Intent Graph (USIG).

The USIG is Semmai's language-independent meaning representation. Both the
user's natural-language request and the generated SQL / tool action are
compiled into a USIG; the proof engine then checks the two graphs for
semantic equivalence.

Nodes of the graph:
    entities     - tables / domain objects the request is about
    projections  - what is returned:  (aggregate, column)
    constraints  - filters:          (column, operator, value)
    joins        - relationships between entities
    group_by / order_by / limit / distinct - result shaping

Canonical form
--------------
Many surface forms mean the same thing.  `canonicalize()` rewrites a USIG
so that equivalent meanings get identical representations:

    age >= 61          ->  age > 60          (integer columns are discrete)
    d <= '2020-12-31'  ->  d < '2021-01-01'  (dates are discrete days)
    x IN ('a')         ->  x = 'a'
    BETWEEN a AND b    ->  >= a  AND  <= b   (done by the parsers)
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import date, timedelta
from typing import Any, Optional

OPS = {"=", "!=", ">", ">=", "<", "<=", "IN", "NOT IN", "LIKE", "NOT LIKE", "IS NULL", "IS NOT NULL"}
NEGATE = {"=": "!=", "!=": "=", "IN": "NOT IN", "NOT IN": "IN", "LIKE": "NOT LIKE",
          "NOT LIKE": "LIKE", "IS NULL": "IS NOT NULL", "IS NOT NULL": "IS NULL",
          ">": "<=", "<=": ">", "<": ">=", ">=": "<"}
AGGS = {"count", "sum", "avg", "min", "max"}


@dataclass(frozen=True)
class Constraint:
    column: str            # "table.column" (lower-case)
    op: str                # one of OPS
    value: Any = None      # number | str | tuple(...) for IN
    source: str = ""       # text span that produced it (for explanations)

    def describe(self) -> str:
        v = self.value
        if isinstance(v, tuple):
            v = "(" + ", ".join(map(str, v)) + ")"
        return f"{self.column} {self.op} {v}" if self.op not in ("IS NULL", "IS NOT NULL") else f"{self.column} {self.op}"


@dataclass
class USIG:
    entities: set = field(default_factory=set)
    projections: list = field(default_factory=list)   # [(agg|None, col)]
    constraints: list = field(default_factory=list)   # [Constraint]
    joins: list = field(default_factory=list)         # [(colA, colB)]
    group_by: list = field(default_factory=list)      # [col]
    order_by: list = field(default_factory=list)      # [(agg|None, col, 'ASC'|'DESC')]
    limit: Optional[int] = None
    distinct: bool = False
    has_or: bool = False
    primary_entity: Optional[str] = None
    notes: list = field(default_factory=list)

    # ------------------------------------------------------------ helpers
    def to_dict(self) -> dict:
        return {
            "entities": sorted(self.entities),
            "primary_entity": self.primary_entity,
            "projections": [f"{a.upper()}({c})" if a else c for a, c in self.projections],
            "constraints": [c.describe() for c in self.constraints],
            "joins": [f"{a} = {b}" for a, b in self.joins],
            "group_by": list(self.group_by),
            "order_by": [f"{(a.upper() + '(' + c + ')') if a else c} {d}" for a, c, d in self.order_by],
            "limit": self.limit,
            "distinct": self.distinct,
            "has_or": self.has_or,
        }

    def copy(self) -> "USIG":
        return USIG(set(self.entities), list(self.projections), list(self.constraints), list(self.joins),
                    list(self.group_by), list(self.order_by), self.limit, self.distinct, self.has_or,
                    self.primary_entity, list(self.notes))

    def aggregates(self) -> set:
        return {a for a, _ in self.projections if a}


# ------------------------------------------------------------------ values
def parse_date(s: Any) -> Optional[date]:
    if not isinstance(s, str) or len(s) < 10:
        return None
    try:
        return date.fromisoformat(s[:10])
    except ValueError:
        return None


def to_number(v: Any) -> Optional[float]:
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        try:
            return float(v.replace(",", ""))
        except ValueError:
            return None
    return None


def values_equal(a: Any, b: Any) -> bool:
    if isinstance(a, tuple) or isinstance(b, tuple):
        ta = a if isinstance(a, tuple) else (a,)
        tb = b if isinstance(b, tuple) else (b,)
        return len(ta) == len(tb) and all(any(values_equal(x, y) for y in tb) for x in ta)
    na, nb = to_number(a), to_number(b)
    if na is not None and nb is not None:
        return abs(na - nb) <= 1e-6 * max(1.0, abs(na), abs(nb))
    if a is None or b is None:
        return a is b
    sa, sb = str(a).strip().lower(), str(b).strip().lower()
    return sa.strip("%") == sb.strip("%") if ("%" in sa or "%" in sb) else sa == sb


# ------------------------------------------------------------ canonical form
def canonicalize_constraint(c: Constraint, col_type: str) -> Constraint:
    op, v = c.op, c.value
    if op in ("IN", "NOT IN") and isinstance(v, tuple) and len(v) == 1:
        op, v = ("=" if op == "IN" else "!="), v[0]
    if op in ("IN", "NOT IN") and isinstance(v, tuple):
        v = tuple(sorted(v, key=lambda x: str(x).lower()))
    if col_type == "int":
        n = to_number(v)
        if n is not None and op in (">", ">=", "<", "<=", "=", "!="):
            if op in (">", "<") and n != int(n):                 # age > 60.5  ->  age > 60
                v = int(n // 1)
                if op == "<":
                    v = int(n // 1) + 1
            elif op == ">=":
                op, v = ">", int(-(-n // 1)) - 1                 # >= 61 -> > 60
            elif op == "<=":
                op, v = "<", int(n // 1) + 1                     # <= 60 -> < 61
            else:
                v = int(n) if n == int(n) else n
    elif col_type == "real":
        n = to_number(v)
        if n is not None and op in (">", ">=", "<", "<=", "=", "!="):
            v = float(n)
    elif col_type == "date":
        d = parse_date(v)
        if d is not None:
            if op == ">=":
                op, d = ">", d - timedelta(days=1)
            elif op == "<=":
                op, d = "<", d + timedelta(days=1)
            v = d.isoformat()
    elif col_type == "text" and isinstance(v, str) and op in ("=", "!="):
        v = v.strip()
    return replace(c, op=op, value=v)


def canonicalize(g: USIG, col_types: dict) -> USIG:
    out = g.copy()
    out.constraints = [canonicalize_constraint(c, col_types.get(c.column, "text")) for c in g.constraints]
    out.joins = [tuple(sorted(j)) for j in g.joins]
    return out
