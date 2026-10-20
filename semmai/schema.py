"""Database schema + value index used for schema linking.

Works with any SQLite database (including official BIRD databases). Column
synonyms come from an optional metadata file; when no metadata exists they
are derived automatically from the column names (``consultation_fee`` ->
"consultation fee", "fee").
"""
from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

STOP = {"id", "no", "of", "the", "a", "an", "date", "num", "number"}


@dataclass
class Column:
    table: str
    name: str
    type: str                     # int | real | text | date
    synonyms: list = field(default_factory=list)
    values: list = field(default_factory=list)   # distinct values (categorical text columns)
    min: object = None
    max: object = None

    @property
    def full(self) -> str:
        return f"{self.table}.{self.name}"


@dataclass
class Table:
    name: str
    columns: dict
    nouns: list
    name_col: Optional[str]
    pk: Optional[str]
    fks: list                      # [(col, ref_table, ref_col)]


def _type_of(decl: str, name: str) -> str:
    d = (decl or "").upper()
    if "DATE" in d or "TIME" in d or name.endswith("_date") or name in ("date", "dob"):
        return "date"
    if "INT" in d:
        return "int"
    if any(k in d for k in ("REAL", "FLOA", "DOUB", "NUM", "DEC")):
        return "real"
    return "text"


def _auto_synonyms(col: str) -> list:
    words = [w for w in re.split(r"[_\s]+", col.lower()) if w]
    syn = [" ".join(words)]
    content = [w for w in words if w not in STOP]
    if content and " ".join(content) != syn[0]:
        syn.append(" ".join(content))
    if len(content) > 1:
        syn.append(content[-1])
    return syn


def _auto_nouns(table: str) -> list:
    base = table.lower().replace("_", " ")
    sing = base[:-1] if base.endswith("s") else base
    return list(dict.fromkeys([base, sing, sing + "s"]))


class Schema:
    def __init__(self, db_id: str, db_path: str | Path, meta: Optional[dict] = None,
                 max_categorical: int = 120):
        self.db_id = db_id
        self.db_path = str(db_path)
        self.tables: dict[str, Table] = {}
        con = sqlite3.connect(self.db_path)
        try:
            names = [r[0] for r in con.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")]
            for t in names:
                tl = t.lower()
                tmeta = (meta or {}).get(tl, {})
                cols, pk = {}, None
                for _, cname, ctype, _, _, ispk in con.execute(f'PRAGMA table_info("{t}")'):
                    cl = cname.lower()
                    typ = _type_of(ctype, cl)
                    syn = tmeta.get("cols", {}).get(cl) or _auto_synonyms(cl)
                    col = Column(tl, cl, typ, list(syn))
                    if typ in ("int", "real", "date"):
                        col.min, col.max = con.execute(f'SELECT MIN("{cname}"), MAX("{cname}") FROM "{t}"').fetchone()
                    if typ == "text":
                        n = con.execute(f'SELECT COUNT(DISTINCT "{cname}") FROM "{t}"').fetchone()[0]
                        if 0 < n <= max_categorical:
                            col.values = [r[0] for r in con.execute(
                                f'SELECT DISTINCT "{cname}" FROM "{t}" WHERE "{cname}" IS NOT NULL')]
                    cols[cl] = col
                    if ispk:
                        pk = cl
                fks = [(r[3].lower(), r[2].lower(), (r[4] or "").lower())
                       for r in con.execute(f'PRAGMA foreign_key_list("{t}")')]
                name_col = tmeta.get("name_col")
                if not name_col:
                    name_col = next((c for c in cols if c == "name" or c.endswith("_name") or c == "title"), pk)
                self.tables[tl] = Table(tl, cols, tmeta.get("noun") or _auto_nouns(tl), name_col, pk, fks)
        finally:
            con.close()

    # ---------------------------------------------------------------- api
    @classmethod
    def from_bird_style(cls, root: str | Path, db_id: str) -> "Schema":
        root = Path(root)
        meta_path = root / "schema_meta.json"
        meta = json.loads(meta_path.read_text()).get(db_id) if meta_path.exists() else None
        return cls(db_id, root / "databases" / db_id / f"{db_id}.sqlite", meta)

    def column(self, full: str) -> Optional[Column]:
        if "." not in full:
            return None
        t, c = full.split(".", 1)
        return self.tables.get(t, Table(t, {}, [], None, None, [])).columns.get(c)

    def col_types(self) -> dict:
        return {c.full: c.type for t in self.tables.values() for c in t.columns.values()}

    def find_column(self, name: str, tables: list) -> Optional[str]:
        name = name.lower()
        for t in tables:
            if t in self.tables and name in self.tables[t].columns:
                return f"{t}.{name}"
        for t, tab in self.tables.items():
            if name in tab.columns:
                return f"{t}.{name}"
        return None

    def join_path(self, a: str, b: str) -> Optional[tuple]:
        """Return a foreign-key equality joining tables a and b (one hop)."""
        for col, rt, rc in self.tables[a].fks:
            if rt == b:
                return (f"{a}.{col}", f"{b}.{rc or self.tables[b].pk}")
        for col, rt, rc in self.tables[b].fks:
            if rt == a:
                return (f"{b}.{col}", f"{a}.{rc or self.tables[a].pk}")
        return None

    def describe(self) -> str:
        """CREATE-style schema text for LLM prompts."""
        out = []
        for t in self.tables.values():
            cols = ", ".join(f"{c.name} {c.type.upper()}" for c in t.columns.values())
            out.append(f"CREATE TABLE {t.name} ({cols});")
        return "\n".join(out)
