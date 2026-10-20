"""Semantic proof engine: checks that the action graph preserves the intent graph.

Every check produces a ``Finding``.  A *critical* finding means the action
changes the meaning of the request (the proof fails); a *warning* is a
difference that does not change the answer set (e.g. an extra column in the
output) and is reported but does not block execution.

Checks (the "proof obligations"):
    ENTITY        intended entity is queried
    CONSTRAINT    each intended constraint is present with the same column,
                  operator and value                 (MISSING / OPERATOR /
                  VALUE_DRIFT / NEGATION_FLIP / COLUMN_MISMATCH)
    EXTRA         no constraint that the user did not ask for
    LOGIC         conjunction preserved (no AND -> OR)
    AGGREGATION   same aggregate function over the same column
    PROJECTION    the requested objects / columns are returned
    GROUP         same grouping
    ORDER/LIMIT   same ranking direction, column and size
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .schema import Schema
from .usig import USIG, Constraint, canonicalize, values_equal
from .verbalize import OP_WORDS, DATE_WORDS


@dataclass
class Finding:
    kind: str
    severity: str          # critical | warning
    message: str

    def to_dict(self):
        return {"kind": self.kind, "severity": self.severity, "message": self.message}


def _fmt_raw(c: Constraint, types: dict) -> str:
    words = DATE_WORDS if types.get(c.column) == "date" else OP_WORDS
    v = c.value
    if isinstance(v, tuple):
        v = "(" + ", ".join(map(str, v)) + ")"
    elif isinstance(v, float) and v.is_integer():
        v = int(v)
    return f"{c.column.split('.')[-1]} {words.get(c.op, c.op)} {v}"


def _same_dir(a: str, b: str) -> bool:
    up, down = {">", ">="}, {"<", "<="}
    return (a in up and b in up) or (a in down and b in down)


NEG_PAIRS = {("=", "!="), ("!=", "="), ("IN", "NOT IN"), ("NOT IN", "IN"), ("LIKE", "NOT LIKE"),
             ("NOT LIKE", "LIKE")}


def prove(intent: USIG, action: USIG, schema: Schema) -> list:
    types = schema.col_types()
    I = canonicalize(intent, types)
    A = canonicalize(action, types)
    F: list[Finding] = []

    # ---------------------------------------------------------------- entity
    if I.primary_entity and I.primary_entity not in A.entities:
        F.append(Finding("WRONG_ENTITY", "critical",
                         f"request is about '{I.primary_entity}' but the action queries {sorted(A.entities)}"))
    extra_tables = A.entities - I.entities
    if extra_tables:
        F.append(Finding("EXTRA_JOIN", "warning", f"action joins tables not implied by the request: {sorted(extra_tables)}"))

    # ----------------------------------------------------------- constraints
    ic = list(I.constraints)
    ac = list(A.constraints)
    shown = {id(c): o for c, o in zip(I.constraints, intent.constraints)}
    shown.update({id(c): o for c, o in zip(A.constraints, action.constraints)})

    def _fmt(c, types):                       # show the user's / the SQL's own wording
        return _fmt_raw(shown.get(id(c), c), types)

    # 1. exact matches
    for c in list(ic):
        m = next((a for a in ac if a.column == c.column and a.op == c.op and values_equal(a.value, c.value)), None)
        if m is not None:
            ic.remove(c)
            ac.remove(m)
    # 2. same column, something changed
    for c in list(ic):
        same = [a for a in ac if a.column == c.column]
        if not same:
            continue
        # prefer a partner with the same value, then same direction
        m = next((a for a in same if values_equal(a.value, c.value)), None) or \
            next((a for a in same if _same_dir(a.op, c.op)), None) or same[0]
        ic.remove(c)
        ac.remove(m)
        if (c.op, m.op) in NEG_PAIRS and values_equal(c.value, m.value):
            F.append(Finding("NEGATION_FLIP", "critical",
                             f"user asked '{_fmt(c, types)}' but action has '{_fmt(m, types)}'"))
        elif values_equal(c.value, m.value) or (_same_dir(c.op, m.op) and _off_by_one(c, m, types)):
            F.append(Finding("OPERATOR_MISMATCH", "critical",
                             f"boundary/operator changed: expected '{_fmt(c, types)}', got '{_fmt(m, types)}'"))
        elif c.op == m.op:
            F.append(Finding("VALUE_DRIFT", "critical",
                             f"value drift: expected '{_fmt(c, types)}', got '{_fmt(m, types)}'"))
        else:
            F.append(Finding("CONSTRAINT_CHANGED", "critical",
                             f"constraint changed: expected '{_fmt(c, types)}', got '{_fmt(m, types)}'"))
    # 3. constraint moved to another column / missing
    for c in list(ic):
        m = next((a for a in ac if a.op == c.op and values_equal(a.value, c.value)), None)
        if m is not None:
            ac.remove(m)
            F.append(Finding("COLUMN_MISMATCH", "critical",
                             f"condition applied to the wrong attribute: expected '{_fmt(c, types)}', "
                             f"got '{_fmt(m, types)}'"))
        else:
            F.append(Finding("CONSTRAINT_MISSING", "critical", f"missing condition: '{_fmt(c, types)}'"))
    for a in ac:
        F.append(Finding("CONSTRAINT_EXTRA", "critical", f"condition the user did not ask for: '{_fmt(a, types)}'"))
    if A.has_or and not I.has_or:
        F.append(Finding("LOGIC_MISMATCH", "critical", "action combines conditions with OR; the request requires all"))

    # ----------------------------------------------------------- aggregation
    i_aggs = [(a, c) for a, c in I.projections if a]
    a_aggs = [(a, c) for a, c in A.projections if a]

    def agg_eq(x, y):
        if x[0] != y[0]:
            return False
        if x[0] == "count":
            return True                      # COUNT(*) == COUNT(col) for non-null keys
        return x[1] == y[1]

    for x in i_aggs:
        if not any(agg_eq(x, y) for y in a_aggs):
            got = ", ".join(f"{a.upper()}({c.split('.')[-1]})" for a, c in a_aggs) or "no aggregation"
            F.append(Finding("AGGREGATION_MISMATCH", "critical",
                             f"expected {x[0].upper()}({x[1].split('.')[-1]}), got {got}"))
    for y in a_aggs:
        if not any(agg_eq(x, y) for x in i_aggs):
            if i_aggs:
                continue                     # already reported
            F.append(Finding("AGGREGATION_MISMATCH", "critical",
                             f"action aggregates {y[0].upper()}({y[1].split('.')[-1]}) but the request asks for a list"))

    # ------------------------------------------------------------ projection
    if not i_aggs:
        prim = I.primary_entity
        ident = set()
        if prim and prim in schema.tables:
            t = schema.tables[prim]
            ident = {f"{prim}.{t.name_col}", f"{prim}.{t.pk}", "*"}
        a_cols = {c for a, c in A.projections if not a}
        if ident and not (a_cols & ident):
            F.append(Finding("PROJECTION_MISMATCH", "critical",
                             f"returns {sorted(a_cols)} instead of the requested {prim} "
                             f"({schema.tables[prim].name_col})"))
        for _, c in I.projections[1:]:
            if c not in a_cols and "*" not in a_cols:
                F.append(Finding("PROJECTION_MISSING", "critical", f"requested column '{c}' is not returned"))
    i_group = sorted(I.group_by)
    a_group = sorted(A.group_by)
    if i_group != a_group:
        F.append(Finding("GROUP_MISMATCH", "critical", f"grouping {a_group or 'none'} differs from requested {i_group or 'none'}"))

    # -------------------------------------------------------- order and limit
    io = [(c, d) for _, c, d in I.order_by]
    ao = [(c, d) for _, c, d in A.order_by]
    if io and not ao:
        F.append(Finding("ORDER_MISMATCH", "critical", "ranking/sorting requested but the action does not order results"))
    elif io and ao:
        if io[0][0] != ao[0][0]:
            F.append(Finding("ORDER_MISMATCH", "critical",
                             f"ranked by '{ao[0][0].split('.')[-1]}' instead of '{io[0][0].split('.')[-1]}'"))
        elif io[0][1] != ao[0][1]:
            F.append(Finding("ORDER_DIRECTION", "critical",
                             f"sort direction {ao[0][1]} but the request implies {io[0][1]}"))
    elif ao and not io:
        sev = "critical" if A.limit is not None else "warning"
        F.append(Finding("ORDER_EXTRA", sev, "action orders results although no ranking was requested"))
    if (I.limit or None) != (A.limit or None):
        F.append(Finding("LIMIT_MISMATCH", "critical", f"result size limit {A.limit} but the request implies {I.limit}"))
    return F


def _off_by_one(c: Constraint, m: Constraint, types: dict) -> bool:
    """`> 60` vs `>= 60` after canonicalisation shows up as `> 60` vs `> 59` (int)."""
    from .usig import to_number, parse_date
    t = types.get(c.column)
    if t == "int":
        a, b = to_number(c.value), to_number(m.value)
        return a is not None and b is not None and abs(a - b) == 1
    if t == "date":
        a, b = parse_date(c.value), parse_date(m.value)
        return a is not None and b is not None and abs((a - b).days) == 1
    return False
