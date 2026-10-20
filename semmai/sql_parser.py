"""Action-semantic extraction for SQL:  SQL text  ->  USIG.

A small recursive-descent parser for the SELECT subset produced by
text-to-SQL models (projections, aggregates, joins, WHERE with AND/OR/NOT,
BETWEEN, IN, LIKE, IS NULL, strftime year filters, GROUP BY, HAVING,
ORDER BY, LIMIT).  Anything outside the subset (sub-queries, UNION, ...)
raises ``UnsupportedSQL`` so the verifier can refuse to certify it instead
of guessing.

Also contains ``render_sql`` (USIG -> SQL) used by the benchmark generator.
"""
from __future__ import annotations

import re
from typing import Any, Optional

from .schema import Schema
from .usig import AGGS, Constraint, USIG, NEGATE


class UnsupportedSQL(Exception):
    pass


TOKEN = re.compile(r"""
    (?P<ws>\s+)
  | (?P<str>'(?:[^']|'')*')
  | (?P<qid>"[^"]*"|`[^`]*`|\[[^\]]*\])
  | (?P<num>\d+\.\d+|\d+|\.\d+)
  | (?P<op><>|!=|>=|<=|==|=|<|>|\|\|)
  | (?P<punc>[(),.*;+\-/%])
  | (?P<id>[A-Za-z_][A-Za-z0-9_]*)
""", re.X)

KEYWORDS = {"select", "distinct", "from", "where", "and", "or", "not", "in", "like", "between", "is",
            "null", "group", "by", "order", "asc", "desc", "limit", "offset", "as", "join", "inner",
            "left", "right", "outer", "on", "having", "union", "except", "intersect", "cross", "natural"}


def tokenize(sql: str) -> list:
    toks, pos = [], 0
    sql = sql.strip().rstrip(";")
    while pos < len(sql):
        m = TOKEN.match(sql, pos)
        if not m:
            raise UnsupportedSQL(f"unexpected character {sql[pos]!r}")
        pos = m.end()
        kind = m.lastgroup
        text = m.group()
        if kind == "ws":
            continue
        if kind == "str":
            toks.append(("str", text[1:-1].replace("''", "'")))
        elif kind == "qid":
            toks.append(("id", text[1:-1]))
        elif kind == "num":
            toks.append(("num", float(text) if "." in text else int(text)))
        elif kind == "id" and text.lower() in KEYWORDS:
            toks.append(("kw", text.lower()))
        elif kind == "id":
            toks.append(("id", text))
        else:
            toks.append((kind, "!=" if text == "<>" else ("=" if text == "==" else text)))
    return toks


class _Col:          # operand that is a column reference (optionally aggregated)
    def __init__(self, ref, agg=None, year=False, fmt=None):
        self.ref, self.agg, self.year, self.fmt = ref, agg, year, fmt


class Parser:
    def __init__(self, sql: str, schema: Schema):
        self.toks = tokenize(sql)
        self.i = 0
        self.schema = schema
        self.alias: dict[str, str] = {}
        self.tables: list[str] = []
        self.sel_alias: dict[str, tuple] = {}

    # ----------------------------------------------------------- token utils
    def peek(self, k=0):
        j = self.i + k
        return self.toks[j] if j < len(self.toks) else ("eof", None)

    def kw(self, *words) -> bool:
        t = self.peek()
        return t[0] == "kw" and t[1] in words

    def take(self, kind=None, val=None):
        t = self.peek()
        if (kind and t[0] != kind) or (val is not None and t[1] != val):
            raise UnsupportedSQL(f"expected {val or kind}, got {t[1]!r}")
        self.i += 1
        return t

    def accept(self, kind, val=None) -> bool:
        t = self.peek()
        if t[0] == kind and (val is None or t[1] == val):
            self.i += 1
            return True
        return False

    # -------------------------------------------------------------- grammar
    def parse(self) -> USIG:
        g = USIG()
        self.take("kw", "select")
        g.distinct = self.accept("kw", "distinct")
        raw_items = self._select_items()
        self.take("kw", "from")
        self._from_clause(g)
        g.entities = set(self.tables)
        g.primary_entity = self.tables[0]
        g.projections = [self._resolve_item(it) for it in raw_items]
        if self.accept("kw", "where"):
            g.constraints, g.has_or = self._where(g)
        if self.kw("group"):
            self.take("kw", "group"); self.take("kw", "by")
            while True:
                op = self._operand()
                g.group_by.append(op.ref)
                if not self.accept("punc", ","):
                    break
        if self.accept("kw", "having"):
            hav, _ = self._where(g)
            g.constraints.extend(hav)
        if self.kw("order"):
            self.take("kw", "order"); self.take("kw", "by")
            while True:
                op = self._operand(order=True, g=g)
                d = "ASC"
                if self.accept("kw", "desc"):
                    d = "DESC"
                else:
                    self.accept("kw", "asc")
                g.order_by.append((op.agg, op.ref, d))
                if not self.accept("punc", ","):
                    break
        if self.accept("kw", "limit"):
            g.limit = int(self.take("num")[1])
            if self.accept("kw", "offset") or self.accept("punc", ","):
                self.take("num")
                g.notes.append("offset")
        if self.peek()[0] != "eof":
            raise UnsupportedSQL(f"unsupported trailing SQL near {self.peek()[1]!r}")
        return g

    def _select_items(self) -> list:
        items = []
        while True:
            start = self.i
            depth = 0
            while True:
                t = self.peek()
                if t[0] == "eof":
                    raise UnsupportedSQL("missing FROM")
                if t == ("punc", "("):
                    depth += 1
                elif t == ("punc", ")"):
                    depth -= 1
                elif depth == 0 and (t == ("punc", ",") or t == ("kw", "from")):
                    break
                if t == ("kw", "select"):
                    raise UnsupportedSQL("sub-queries are not supported")
                self.i += 1
            items.append(self.toks[start:self.i])
            if not self.accept("punc", ","):
                break
        return items

    def _from_clause(self, g: USIG):
        self._table_ref()
        while True:
            if self.accept("punc", ","):
                self._table_ref()
                continue
            if self.kw("join", "inner", "left", "right", "cross", "natural"):
                while self.kw("inner", "left", "right", "outer", "cross", "natural"):
                    self.i += 1
                self.take("kw", "join")
                self._table_ref()
                if self.accept("kw", "on"):
                    conds, _ = self._where(g)
                    for c in conds:
                        g.constraints.append(c)
                continue
            break

    def _table_ref(self):
        if self.peek() == ("punc", "("):
            raise UnsupportedSQL("derived tables are not supported")
        name = self.take("id")[1].lower()
        if name not in self.schema.tables:
            raise UnsupportedSQL(f"unknown table {name}")
        self.tables.append(name)
        self.alias[name] = name
        if self.accept("kw", "as"):
            self.alias[self.take("id")[1].lower()] = name
        elif self.peek()[0] == "id":
            self.alias[self.take("id")[1].lower()] = name

    def _resolve_col(self, parts: list) -> str:
        if parts == ["*"]:
            return "*"
        if len(parts) == 2:
            t = self.alias.get(parts[0].lower())
            if t is None:
                raise UnsupportedSQL(f"unknown alias {parts[0]}")
            if parts[1] == "*":
                return "*"
            return f"{t}.{parts[1].lower()}"
        name = parts[0].lower()
        if name in self.sel_alias:
            return self.sel_alias[name][1]
        full = self.schema.find_column(name, self.tables)
        if full is None:
            raise UnsupportedSQL(f"unknown column {name}")
        return full

    def _resolve_item(self, toks: list) -> tuple:
        sub = Parser.__new__(Parser)
        sub.toks, sub.i, sub.schema = toks, 0, self.schema
        sub.alias, sub.tables, sub.sel_alias = self.alias, self.tables, {}
        op = sub._operand()
        if sub.accept("kw", "as") or sub.peek()[0] == "id":
            name = sub.take("id")[1].lower()
            self.sel_alias[name] = (op.agg, op.ref)
        return (op.agg, op.ref)

    def _operand(self, order=False, g: Optional[USIG] = None) -> _Col:
        t = self.peek()
        if t[0] == "num" and order and g is not None:          # ORDER BY 2
            self.i += 1
            a, c = g.projections[t[1] - 1]
            return _Col(c, a)
        if t == ("punc", "*"):
            self.i += 1
            return _Col("*")
        if t[0] == "id" and self.peek(1) == ("punc", "("):
            fn = t[1].lower()
            self.i += 2
            if fn in AGGS:
                self.accept("kw", "distinct")
                inner = self._operand()
                self.take("punc", ")")
                return _Col(inner.ref, fn)
            if fn == "strftime":
                fmt = self.take("str")[1]
                self.take("punc", ",")
                inner = self._operand()
                self.take("punc", ")")
                if fmt not in ("%Y", "%Y-%m"):
                    raise UnsupportedSQL(f"unsupported strftime format {fmt}")
                return _Col(inner.ref, None, year=True, fmt=fmt)
            if fn in ("lower", "upper", "trim", "date", "abs", "round"):
                inner = self._operand()
                while self.accept("punc", ","):
                    self.take("num")
                self.take("punc", ")")
                return inner
            if fn == "cast":
                inner = self._operand()
                self.take("kw", "as"); self.take("id")
                self.take("punc", ")")
                return inner
            raise UnsupportedSQL(f"unsupported function {fn}")
        if t[0] == "id" and self.peek(1) != ("punc", ".") and t[1].lower() in getattr(self, "sel_alias", {}):
            self.i += 1
            a, c = self.sel_alias[t[1].lower()]
            return _Col(c, a)
        if t[0] == "id":
            self.i += 1
            parts = [t[1]]
            if self.accept("punc", "."):
                nxt = self.take()
                parts.append(nxt[1] if nxt[0] != "punc" else "*")
            return _Col(self._resolve_col(parts))
        raise UnsupportedSQL(f"unexpected token {t[1]!r}")

    def _literal(self) -> Any:
        t = self.peek()
        neg = False
        if t == ("punc", "-"):
            neg = True
            self.i += 1
            t = self.peek()
        if t[0] in ("num", "str"):
            self.i += 1
            return -t[1] if neg else t[1]
        if t == ("kw", "null"):
            self.i += 1
            return None
        raise UnsupportedSQL(f"expected literal, got {t[1]!r}")

    # WHERE  ->  list of constraints (AND semantics); OR over the same column
    # becomes IN; any other OR is kept but flagged with has_or.
    def _where(self, g: USIG):
        groups = [self._and_expr(g)]
        while self.accept("kw", "or"):
            groups.append(self._and_expr(g))
        if len(groups) == 1:
            return groups[0], False
        flat = [c for grp in groups for c in grp]
        if all(len(grp) == 1 for grp in groups) and len({c.column for c in flat}) == 1 \
                and all(c.op in ("=", "IN") for c in flat):
            vals = []
            for c in flat:
                vals.extend(c.value if isinstance(c.value, tuple) else (c.value,))
            return [Constraint(flat[0].column, "IN", tuple(vals))], False
        return flat, True

    def _and_expr(self, g):
        out = self._not_expr(g)
        while self.accept("kw", "and"):
            out = out + self._not_expr(g)
        return out

    def _not_expr(self, g):
        if self.accept("kw", "not"):
            inner = self._not_expr(g)
            if len(inner) != 1:
                raise UnsupportedSQL("NOT over compound condition")
            c = inner[0]
            return [Constraint(c.column, NEGATE[c.op], c.value)]
        return self._predicate(g)

    def _predicate(self, g):
        if self.peek() == ("punc", "(") :
            # parenthesised boolean expression or arithmetic? assume boolean
            self.i += 1
            if self.kw("select"):
                raise UnsupportedSQL("sub-queries are not supported")
            conds, has_or = self._where(g)
            self.take("punc", ")")
            if has_or:
                g.has_or = True
            return conds
        left_is_lit = self.peek()[0] in ("num", "str")
        if left_is_lit:
            lit = self._literal()
            op = self.take("op")[1]
            col = self._operand()
            flip = {">": "<", "<": ">", ">=": "<=", "<=": ">=", "=": "=", "!=": "!="}
            return self._mk(col, flip[op], lit)
        col = self._operand()
        colname = col.ref if not col.agg else f"{col.agg}({col.ref})"
        if self.accept("kw", "is"):
            neg = self.accept("kw", "not")
            self.take("kw", "null")
            return [Constraint(colname, "IS NOT NULL" if neg else "IS NULL", None)]
        neg = self.accept("kw", "not")
        if self.accept("kw", "between"):
            lo = self._literal(); self.take("kw", "and"); hi = self._literal()
            if neg:
                raise UnsupportedSQL("NOT BETWEEN")
            return self._mk(col, ">=", lo) + self._mk(col, "<=", hi)
        if self.accept("kw", "in"):
            self.take("punc", "(")
            if self.kw("select"):
                raise UnsupportedSQL("sub-queries are not supported")
            vals = [self._literal()]
            while self.accept("punc", ","):
                vals.append(self._literal())
            self.take("punc", ")")
            return [Constraint(colname, "NOT IN" if neg else "IN", tuple(vals))]
        if self.accept("kw", "like"):
            return [Constraint(colname, "NOT LIKE" if neg else "LIKE", self._literal())]
        op = self.take("op")[1]
        if self.peek()[0] == "id" or (self.peek()[0] == "id" and self.peek(1) == ("punc", "(")):
            other = self._operand()
            g.joins.append((col.ref, other.ref))
            return []
        return self._mk(col, op, self._literal())

    def _mk(self, col: _Col, op: str, val: Any) -> list:
        name = col.ref if not col.agg else f"{col.agg}({col.ref})"
        if col.year:                         # strftime('%Y', d) op 'YYYY' -> date range
            s = str(val)
            if col.fmt == "%Y":
                lo, hi = f"{s}-01-01", f"{s}-12-31"
            else:
                y, m = s.split("-")[:2]
                import calendar
                lo, hi = f"{y}-{m}-01", f"{y}-{m}-{calendar.monthrange(int(y), int(m))[1]:02d}"
            table = {"=": [(">=", lo), ("<=", hi)], ">": [(">", hi)], ">=": [(">=", lo)],
                     "<": [("<", lo)], "<=": [("<=", hi)]}.get(op)
            if table is None:
                raise UnsupportedSQL("unsupported year comparison")
            return [Constraint(name, o, v) for o, v in table]
        return [Constraint(name, op, val)]


def sql_to_usig(sql: str, schema: Schema) -> USIG:
    g = Parser(sql, schema).parse()
    # join conditions inside ON were parsed as constraints only if literal; col=col went to joins
    return g


# ----------------------------------------------------------------- renderer
def _lit(v: Any) -> str:
    if isinstance(v, tuple):
        return "(" + ", ".join(_lit(x) for x in v) + ")"
    if isinstance(v, str):
        return "'" + v.replace("'", "''") + "'"
    if isinstance(v, float) and v == int(v):
        return str(int(v))
    return str(v)


def render_sql(g: USIG, schema: Schema, aliases: bool = False, between: bool = False,
               lower_kw: bool = False, reverse_conds: bool = False) -> str:
    tables = [g.primary_entity] + sorted(t for t in g.entities if t != g.primary_entity)
    al = {t: (f"T{i + 1}" if aliases else t) for i, t in enumerate(tables)}
    multi = len(tables) > 1

    def col(ref: str) -> str:
        if ref == "*":
            return "*"
        t, c = ref.split(".", 1)
        return f"{al[t]}.{c}" if (multi or aliases) else c

    def proj(a, c):
        return f"{a.upper()}({col(c)})" if a else col(c)

    sel = ", ".join(proj(a, c) for a, c in g.projections) or "*"
    sql = "SELECT " + ("DISTINCT " if g.distinct else "") + sel
    sql += f" FROM {tables[0]}" + (f" AS {al[tables[0]]}" if aliases else "")
    placed = {tables[0]}
    for t in tables[1:]:
        j = next((jj for jj in g.joins if any(x.startswith(t + ".") for x in jj)
                  and any(x.split(".")[0] in placed for x in jj)), None)
        if j is None:
            raise ValueError(f"no join for {t}")
        sql += f" JOIN {t}" + (f" AS {al[t]}" if aliases else "") + f" ON {col(j[0])} = {col(j[1])}"
        placed.add(t)
    conds = list(g.constraints)
    if reverse_conds:
        conds = conds[::-1]
    parts, used = [], set()
    for i, c in enumerate(conds):
        if i in used:
            continue
        name = c.column
        m = re.match(r"(\w+)\((.+)\)$", name)
        cref = f"{m.group(1).upper()}({col(m.group(2))})" if m else col(name)
        if between and c.op == ">=":
            k = next((k for k in range(i + 1, len(conds)) if conds[k].column == name and conds[k].op == "<="), None)
            if k is not None:
                used.add(k)
                parts.append(f"{cref} BETWEEN {_lit(c.value)} AND {_lit(conds[k].value)}")
                continue
        if c.op in ("IS NULL", "IS NOT NULL"):
            parts.append(f"{cref} {c.op}")
        else:
            parts.append(f"{cref} {c.op} {_lit(c.value)}")
    having = [p for p, c in zip(parts, [c for i, c in enumerate(conds) if i not in used]) if "(" in c.column]
    where = [p for p in parts if p not in having]
    if where:
        sql += " WHERE " + (" OR " if g.has_or else " AND ").join(where)
    if g.group_by:
        sql += " GROUP BY " + ", ".join(col(c) for c in g.group_by)
    if having:
        sql += " HAVING " + " AND ".join(having)
    if g.order_by:
        sql += " ORDER BY " + ", ".join(f"{proj(a, c)} {d}" for a, c, d in g.order_by)
    if g.limit is not None:
        sql += f" LIMIT {g.limit}"
    if lower_kw:
        for k in ["SELECT", "DISTINCT", "FROM", "WHERE", "AND", "GROUP BY", "ORDER BY", "LIMIT", "JOIN", " ON ",
                  " AS ", "BETWEEN", "HAVING", "DESC", "ASC", "COUNT", "AVG", "SUM", "MIN", "MAX", " IN ", "NOT"]:
            sql = sql.replace(k, k.lower())
    return sql
