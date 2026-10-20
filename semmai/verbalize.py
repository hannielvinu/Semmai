"""USIG -> plain English.  Used for explanations in the demo and as the
text side of the neural similarity check (question vs. what the SQL does)."""
from __future__ import annotations

from .usig import USIG

OP_WORDS = {">": "more than", ">=": "at least", "<": "less than", "<=": "at most", "=": "is", "!=": "is not",
            "IN": "is one of", "NOT IN": "is not one of", "LIKE": "contains", "NOT LIKE": "does not contain",
            "IS NULL": "is missing", "IS NOT NULL": "is present"}
DATE_WORDS = {">": "after", ">=": "on or after", "<": "before", "<=": "on or before", "=": "on", "!=": "not on"}
AGG_WORDS = {"count": "number of", "avg": "average", "sum": "total", "max": "maximum", "min": "minimum"}


def _col(c: str) -> str:
    if c == "*":
        return "records"
    return c.split(".")[-1].replace("_", " ")


def _val(v) -> str:
    if isinstance(v, tuple):
        return ", ".join(_val(x) for x in v)
    if isinstance(v, float) and v.is_integer():
        return str(int(v))
    return str(v)


def verbalize(g: USIG, col_types: dict | None = None) -> str:
    col_types = col_types or {}
    ent = g.primary_entity or "records"
    if g.group_by:
        parts = [f"{AGG_WORDS.get(a, a)} {_col(c) if c != '*' else ent}" for a, c in g.projections if a]
        head = " and ".join(parts) + f" of {ent} for each " + ", ".join(_col(c) for c in g.group_by)
    elif any(a for a, _ in g.projections):
        parts = [f"{AGG_WORDS.get(a, a)} {_col(c) if c != '*' else ent}" for a, c in g.projections if a]
        head = " and ".join(parts) + ("" if parts[0].endswith(ent) else f" of {ent}")
    else:
        head = "list " + " and ".join(_col(c) for _, c in g.projections) + f" of {ent}"
    conds = []
    for c in g.constraints:
        typ = col_types.get(c.column, "text")
        if typ == "date":
            conds.append(f"{_col(c.column)} {DATE_WORDS.get(c.op, c.op)} {_val(c.value)}")
        else:
            conds.append(f"{_col(c.column)} {OP_WORDS.get(c.op, c.op)} {_val(c.value)}")
    text = head
    if conds:
        text += " where " + (" or " if g.has_or else " and ").join(conds)
    if g.order_by:
        a, c, d = g.order_by[0]
        if g.limit:
            text += f", the {g.limit} with the {'highest' if d == 'DESC' else 'lowest'} {_col(c)}"
        else:
            text += f", sorted by {_col(c)} {'descending' if d == 'DESC' else 'ascending'}"
    elif g.limit:
        text += f", only {g.limit}"
    return text
