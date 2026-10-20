"""Semantic mutation operators = the kinds of errors LLMs make in SQL.

``mutate(gold, ...)`` returns *semantically different* SQL (negatives), and
``rewrite(gold, ...)`` returns *semantically equivalent* SQL written
differently (positives).  Both are used to build the verification benchmark:
a good verifier must reject every mutation and accept every rewrite.

The mutation catalogue follows the error taxonomy in the project slides:
wrong entity / table, wrong operator (>= vs >), missing or extra constraints,
incorrect parameters, dates or values - plus aggregation, ranking and logic
errors that are common in text-to-SQL output.
"""
from __future__ import annotations

import random
from dataclasses import replace
from datetime import date, timedelta
from typing import Optional

from .schema import Schema
from .sql_parser import render_sql
from .usig import USIG, Constraint, parse_date

MUTATIONS = ["OPERATOR_BOUNDARY", "OPERATOR_REVERSE", "VALUE_DRIFT", "DATE_DRIFT", "NEGATION_FLIP",
             "CONSTRAINT_DROP", "CONSTRAINT_ADD", "COLUMN_SWAP", "ENTITY_SWAP", "AGGREGATION_SWAP",
             "ORDER_FLIP", "LIMIT_CHANGE", "LOGIC_AND_TO_OR", "PROJECTION_SWAP", "GROUP_SWAP"]
REWRITES = ["ALIASES", "LOWERCASE", "REORDER_CONDITIONS", "BETWEEN", "BOUNDARY_EQUIVALENT", "IN_SINGLE",
            "YEAR_FUNCTION", "COUNT_KEY", "LITERAL_LEFT"]

BOUNDARY = {">": ">=", ">=": ">", "<": "<=", "<=": "<"}
REVERSE = {">": "<", "<": ">", ">=": "<=", "<=": ">="}


def _nice(v: float, typ: str):
    if typ == "int":
        return int(round(v))
    mag = 10 ** max(0, len(str(int(abs(v)))) - 2)
    return float(round(v / mag) * mag) if v >= 100 else round(v, 1)


class Mutator:
    def __init__(self, schema: Schema, rng: random.Random):
        self.s = schema
        self.rng = rng
        self.types = schema.col_types()

    def _num_cols(self, table):
        t = self.s.tables[table]
        return [c for c in t.columns.values() if c.type in ("int", "real") and c.name != t.pk
                and not c.name.endswith("_id")]

    def _cat_cols(self, table):
        return [c for c in self.s.tables[table].columns.values() if c.values]

    # --------------------------------------------------------------- negatives
    def apply(self, g: USIG, kind: str) -> Optional[USIG]:
        m = g.copy()
        cs = list(m.constraints)
        rng = self.rng
        num_cs = [i for i, c in enumerate(cs) if c.op in BOUNDARY]
        if kind == "OPERATOR_BOUNDARY" and num_cs:
            i = rng.choice(num_cs)
            cs[i] = replace(cs[i], op=BOUNDARY[cs[i].op])
        elif kind == "OPERATOR_REVERSE" and num_cs:
            i = rng.choice(num_cs)
            cs[i] = replace(cs[i], op=REVERSE[cs[i].op])
        elif kind == "VALUE_DRIFT":
            opts = [i for i, c in enumerate(cs) if self.types.get(c.column) in ("int", "real")
                    and isinstance(c.value, (int, float))] + \
                   [i for i, c in enumerate(cs) if self.types.get(c.column) == "text" and c.op in ("=", "!=")
                    and self.s.column(c.column) and len(self.s.column(c.column).values) > 1]
            if not opts:
                return None
            i = rng.choice(opts)
            c = cs[i]
            typ = self.types.get(c.column)
            if typ in ("int", "real"):
                f = rng.choice([0.5, 0.8, 1.25, 1.5, 2.0])
                nv = _nice(c.value * f, typ)
                if nv == c.value:
                    nv = c.value + (1 if typ == "int" else 0.5) * rng.choice([5, 10])
                cs[i] = replace(c, value=nv)
            else:
                others = [v for v in self.s.column(c.column).values if str(v).lower() != str(c.value).lower()]
                cs[i] = replace(c, value=rng.choice(others))
        elif kind == "DATE_DRIFT":
            opts = [i for i, c in enumerate(cs) if self.types.get(c.column) == "date" and parse_date(c.value)]
            if not opts:
                return None
            shift = rng.choice([365, -365, 30, -30])
            for i in opts:            # shift a whole date range consistently
                d = parse_date(cs[i].value)
                try:
                    nd = d.replace(year=d.year + shift // 365) if abs(shift) == 365 else d + timedelta(days=shift)
                except ValueError:
                    nd = d + timedelta(days=shift)
                cs[i] = replace(cs[i], value=nd.isoformat())
        elif kind == "NEGATION_FLIP":
            opts = [i for i, c in enumerate(cs) if c.op in ("=", "!=", "IN", "NOT IN")]
            if not opts:
                return None
            i = rng.choice(opts)
            cs[i] = replace(cs[i], op={"=": "!=", "!=": "=", "IN": "NOT IN", "NOT IN": "IN"}[cs[i].op])
        elif kind == "CONSTRAINT_DROP":
            if not cs:
                return None
            # drop a whole range (both bounds) when it is a range
            i = rng.randrange(len(cs))
            col = cs[i].column
            rng_pair = [j for j, c in enumerate(cs) if c.column == col]
            cs = [c for j, c in enumerate(cs) if j not in (rng_pair if len(rng_pair) == 2 else [i])]
        elif kind == "CONSTRAINT_ADD":
            t = m.primary_entity
            used = {c.column for c in cs}
            cats = [c for c in self._cat_cols(t) if c.full not in used]
            nums = [c for c in self._num_cols(t) if c.full not in used]
            if cats and (not nums or rng.random() < 0.5):
                c = rng.choice(cats)
                cs.append(Constraint(c.full, "=", rng.choice(c.values)))
            elif nums:
                c = rng.choice(nums)
                v = _nice(float(c.min) + (float(c.max) - float(c.min)) * rng.uniform(0.3, 0.7), c.type)
                cs.append(Constraint(c.full, rng.choice([">", "<"]), v))
            else:
                return None
        elif kind == "COLUMN_SWAP":
            opts = []
            for i, c in enumerate(cs):
                t, name = c.column.split(".")
                typ = self.types.get(c.column)
                if typ in ("int", "real"):
                    alts = [x for x in self._num_cols(t) if x.full != c.column]
                elif typ == "date":
                    alts = [x for x in self.s.tables[t].columns.values() if x.type == "date" and x.full != c.column]
                else:
                    alts = [x for x in self._cat_cols(t) if x.full != c.column and c.value in x.values]
                    if not alts:
                        alts = [x for x in self._cat_cols(t) if x.full != c.column]
                        if alts and c.op in ("=", "!="):
                            a = rng.choice(alts)
                            opts.append((i, a, rng.choice(a.values)))
                            continue
                for a in alts:
                    opts.append((i, a, c.value))
            if not opts:
                return None
            i, a, v = rng.choice(opts)
            cs[i] = replace(cs[i], column=a.full, value=v)
        elif kind == "ENTITY_SWAP":
            t = m.primary_entity
            if len(m.entities) > 1:
                return None
            cols = {c.column.split(".")[1] for c in cs} | {c.split(".")[1] for a, c in m.projections if c != "*"} \
                | {c.split(".")[1] for c in m.group_by} | {c.split(".")[1] for _, c, _ in m.order_by}
            alts = [x for x in self.s.tables if x != t and all(cn in self.s.tables[x].columns for cn in cols)]
            if not alts:
                return None
            nt = rng.choice(alts)
            sw = lambda ref: ref if ref == "*" else f"{nt}.{ref.split('.')[1]}"
            cs = [replace(c, column=sw(c.column)) for c in cs]
            m.projections = [(a, sw(c)) for a, c in m.projections]
            m.group_by = [sw(c) for c in m.group_by]
            m.order_by = [(a, sw(c), d) for a, c, d in m.order_by]
            m.entities = {nt}
            m.primary_entity = nt
        elif kind == "AGGREGATION_SWAP":
            aggs = [i for i, (a, c) in enumerate(m.projections) if a]
            if not aggs:
                return None
            i = aggs[0]
            a, c = m.projections[i]
            if a == "count":
                nums = self._num_cols(m.primary_entity)
                if not nums:
                    return None
                m.projections[i] = ("sum", rng.choice(nums).full)
            else:
                m.projections[i] = (rng.choice([x for x in ("avg", "sum", "max", "min") if x != a]), c)
        elif kind == "ORDER_FLIP":
            if not m.order_by:
                return None
            a, c, d = m.order_by[0]
            m.order_by = [(a, c, "ASC" if d == "DESC" else "DESC")]
        elif kind == "LIMIT_CHANGE":
            if m.limit is None:
                return None
            m.limit = rng.choice([x for x in (1, 3, 5, 10, 20) if x != m.limit])
        elif kind == "LOGIC_AND_TO_OR":
            if len({c.column for c in cs}) < 2 or any(c.op in ("IN", "NOT IN") for c in cs):
                return None
            m.has_or = True
        elif kind == "PROJECTION_SWAP":
            if any(a for a, _ in m.projections) or m.group_by:
                return None
            t = m.primary_entity
            name = self.s.tables[t].name_col
            alts = [c for c in self.s.tables[t].columns.values() if c.name not in (name, self.s.tables[t].pk)]
            if not alts:
                return None
            m.projections = [(None, rng.choice(alts).full)] + m.projections[1:]
        elif kind == "GROUP_SWAP":
            if not m.group_by:
                return None
            t = m.primary_entity
            alts = [c for c in self._cat_cols(t) if c.full not in m.group_by]
            if not alts:
                return None
            ng = rng.choice(alts).full
            m.projections = [(a, ng if c == m.group_by[0] else c) for a, c in m.projections]
            m.group_by = [ng]
        else:
            return None
        m.constraints = cs
        return m

    def mutate(self, g: USIG, kind: str) -> Optional[str]:
        m = self.apply(g, kind)
        if m is None:
            return None
        try:
            return render_sql(m, self.s, aliases=self.rng.random() < 0.3)
        except Exception:
            return None

    # --------------------------------------------------------------- positives
    def rewrite(self, g: USIG, kind: str) -> Optional[str]:
        m = g.copy()
        cs = list(m.constraints)
        rng = self.rng
        if kind == "ALIASES":
            return render_sql(m, self.s, aliases=True)
        if kind == "LOWERCASE":
            return render_sql(m, self.s, lower_kw=True)
        if kind == "REORDER_CONDITIONS":
            if len(cs) < 2:
                return None
            return render_sql(m, self.s, reverse_conds=True)
        if kind == "BETWEEN":
            cols = [c.column for c in cs]
            if not any(cols.count(x) == 2 for x in cols):
                return None
            return render_sql(m, self.s, between=True)
        if kind == "BOUNDARY_EQUIVALENT":
            opts = [i for i, c in enumerate(cs) if c.op in BOUNDARY and self.types.get(c.column) in ("int", "date")]
            if not opts:
                return None
            i = rng.choice(opts)
            c = cs[i]
            if self.types[c.column] == "int":
                nv = {">": c.value + 1, ">=": c.value - 1, "<": c.value - 1, "<=": c.value + 1}[c.op]
            else:
                d = parse_date(c.value)
                delta = {">": 1, ">=": -1, "<": -1, "<=": 1}[c.op]
                nv = (d + timedelta(days=delta)).isoformat()
            cs[i] = replace(c, op=BOUNDARY[c.op], value=nv)
            m.constraints = cs
            return render_sql(m, self.s)
        if kind == "IN_SINGLE":
            opts = [i for i, c in enumerate(cs) if c.op in ("=", "!=") and isinstance(c.value, str)]
            if not opts:
                return None
            i = rng.choice(opts)
            cs[i] = replace(cs[i], op="IN" if cs[i].op == "=" else "NOT IN", value=(cs[i].value,))
            m.constraints = cs
            return render_sql(m, self.s)
        if kind == "YEAR_FUNCTION":
            for i, c in enumerate(cs):
                if self.types.get(c.column) == "date" and c.op == ">=" and str(c.value).endswith("-01-01"):
                    j = next((j for j, d in enumerate(cs) if d.column == c.column and d.op == "<="
                              and str(d.value) == str(c.value)[:4] + "-12-31"), None)
                    if j is not None:
                        sql = render_sql(m, self.s)
                        col = c.column.split(".")[1] if len(m.entities) == 1 else c.column
                        lo = f"{col} >= '{c.value}' AND {col} <= '{cs[j].value}'"
                        if lo in sql:
                            return sql.replace(lo, f"strftime('%Y', {col}) = '{str(c.value)[:4]}'")
            return None
        if kind == "COUNT_KEY":
            if ("count", "*") not in m.projections:
                return None
            pk = self.s.tables[m.primary_entity].pk
            m.projections = [("count", f"{m.primary_entity}.{pk}") if p == ("count", "*") else p for p in m.projections]
            return render_sql(m, self.s)
        if kind == "LITERAL_LEFT":
            sql = render_sql(m, self.s)
            for c in cs:
                if c.op in (">", "<", ">=", "<=") and isinstance(c.value, (int, float)):
                    col = c.column.split(".")[1] if len(m.entities) == 1 else c.column
                    v = int(c.value) if float(c.value).is_integer() else c.value
                    frm = f"{col} {c.op} {v}"
                    if frm in sql:
                        flip = {">": "<", "<": ">", ">=": "<=", "<=": ">="}[c.op]
                        return sql.replace(frm, f"{v} {flip} {col}", 1)
            return None
        if kind == "SELECT_STAR":
            if any(a for a, _ in m.projections) or m.group_by or len(m.entities) > 1 or len(m.projections) > 1:
                return None
            m.projections = [(None, "*")]
            return render_sql(m, self.s)
        return None
