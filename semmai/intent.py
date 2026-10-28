"""Intent extraction:  natural-language request  ->  USIG.

A schema-grounded semantic parser (no training data required):

  1. normalise text, numbers ("₹1,00,000", "10 lakh", "five"), contractions
  2. link categorical VALUES against the database value index
     (exact, multi-word, and stem-prefix fuzzy: "cardiologist" -> Cardiology)
  3. link ENTITY nouns and COLUMN synonyms (stemmed matching)
  4. detect DATE expressions (ISO / "15 March 2023" / "March 2024" / "2024")
     with temporal operators: after / since / before / on or before / in
  5. detect NUMERIC constraints with comparator phrases
     ("more than", "at least", "60 and above", "doesn't exceed", "between")
     and attach each number to the most plausible column using synonyms,
     units, comparator hints and the column's value range in the database
  6. detect NEGATION scope ("not", "except", "other than", "outside")
  7. detect aggregation, superlatives / top-k, grouping and ordering

The output is the *intended* meaning, against which the generated SQL is
checked by the proof engine.
"""
from __future__ import annotations

import calendar
import os
import re
from dataclasses import dataclass
from typing import Optional

from .schema import Schema
from .usig import Constraint, USIG

# ------------------------------------------------------------------ lexicons
WORDNUM = {"two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
           "ten": 10, "eleven": 11, "twelve": 12, "fifteen": 15, "twenty": 20, "thirty": 30, "forty": 40,
           "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90, "hundred": 100}
MONTHS = {m.lower(): i for i, m in enumerate(calendar.month_name) if m}
MONTHS.update({m.lower(): i for i, m in enumerate(calendar.month_abbr) if m})
MONTHS["sept"] = 9

CMP_PREFIX = [  # (phrase, op)  - longest match wins
    ("greater than or equal to", ">="), ("more than or equal to", ">="), ("less than or equal to", "<="),
    ("not less than", ">="), ("no less than", ">="), ("not fewer than", ">="), ("no fewer than", ">="),
    ("not below", ">="), ("not under", ">="), ("a minimum of", ">="), ("minimum of", ">="), ("at least", ">="),
    ("atleast", ">="),
    ("not more than", "<="), ("no more than", "<="), ("not greater than", "<="), ("not above", "<="),
    ("not over", "<="), ("does not exceed", "<="), ("do not exceed", "<="), ("not exceeding", "<="),
    ("not exceed", "<="), ("no older than", "<="), ("no higher than", "<="), ("a maximum of", "<="),
    ("maximum of", "<="), ("at most", "<="), ("up to", "<="), ("upto", "<="), ("within", "<="),
    ("greater than", ">"), ("more than", ">"), ("higher than", ">"), ("larger than", ">"), ("bigger than", ">"),
    ("older than", ">"), ("longer than", ">"), ("costlier than", ">"), ("more expensive than", ">"),
    ("in excess of", ">"), ("upwards of", ">"), ("exceeding", ">"), ("exceeds", ">"), ("exceed", ">"),
    ("above", ">"), ("over", ">"), ("beyond", ">"), ("after", ">"),
    ("less than", "<"), ("fewer than", "<"), ("lower than", "<"), ("smaller than", "<"), ("younger than", "<"),
    ("shorter than", "<"), ("cheaper than", "<"), ("less expensive than", "<"), ("below", "<"), ("under", "<"),
    ("equal to", "="), ("equals", "="), ("exactly", "="),
]
CMP_SUFFIX = [("or more", ">="), ("or above", ">="), ("or higher", ">="), ("or older", ">="), ("or over", ">="),
              ("or greater", ">="), ("and above", ">="), ("and over", ">="), ("and up", ">="), ("and older", ">="),
              ("or later", ">="), ("plus", ">="),
              ("or less", "<="), ("or below", "<="), ("or fewer", "<="), ("or younger", "<="), ("or under", "<="),
              ("and below", "<="), ("or lower", "<="), ("and under", "<="), ("or earlier", "<=")]
CMP_HINT = {"older than": "age", "younger than": "age", "aged": "age", "no older than": "age",
            "cheaper than": "money", "costlier than": "money", "more expensive than": "money",
            "less expensive than": "money", "longer than": "duration", "shorter than": "duration"}
DATE_CMP = [("on or after", ">="), ("on or before", "<="), ("no later than", "<="), ("not later than", "<="),
            ("prior to", "<"), ("earlier than", "<"), ("later than", ">"), ("after", ">"), ("since", ">="),
            ("starting", ">="), ("from", ">="), ("before", "<"), ("until", "<="), ("till", "<="), ("up to", "<="),
            ("by", "<="), ("during", "in"), ("in", "in"), ("on", "in"), ("of", "in")]
SUFFIX_WORDS = {"or", "and", "more", "above", "higher", "older", "over", "greater", "up", "plus", "less", "below",
                "fewer", "younger", "under", "lower", "later", "earlier"}
NEG_WORDS = {"not", "except", "excluding", "exclude", "outside", "besides", "non", "without", "never"}
NEG_PHRASES = [("other", "than"), ("apart", "from"), ("aside", "from")]
BOUNDARY = {"and", "or", "but", "with", "who", "whose", "which", "that", "where", ",", "having", "while"}
SUPER_DESC = {"highest", "maximum", "max", "largest", "biggest", "greatest", "most", "top", "costliest",
              "oldest", "longest", "best", "latest", "newest", "priciest", "richest"}
SUPER_ASC = {"lowest", "minimum", "min", "smallest", "least", "cheapest", "youngest", "shortest", "fewest",
             "earliest", "bottom", "worst", "poorest"}
SUPER_HINT = {"cheapest": "money", "costliest": "money", "priciest": "money", "oldest": "age",
              "youngest": "age", "longest": "duration", "shortest": "duration", "latest": "date",
              "newest": "date", "earliest": "date", "richest": "money", "poorest": "money"}
AGG_WORDS = {"average": "avg", "mean": "avg", "avg": "avg", "total": "sum", "sum": "sum", "overall": "sum",
             "combined": "sum", "maximum": "max", "max": "max", "highest": "max", "largest": "max",
             "minimum": "min", "min": "min", "lowest": "min", "smallest": "min"}
MONEY_WORDS = ("price", "amount", "fee", "balance", "bill", "cost", "fare", "salary", "revenue", "income",
               "charge", "payment", "value")
LIST_STARTS = {"which", "who", "list", "show", "give", "find", "get", "display", "retrieve", "name", "fetch",
               "return", "select", "identify"}


def stem(w: str) -> str:
    w = w.lower()
    if len(w) > 4 and w.endswith("ies"):
        w = w[:-3] + "y"
    elif w.endswith("sses"):
        w = w[:-2]
    elif w.endswith("ss"):
        pass
    elif len(w) > 5 and w.endswith("ing"):
        w = w[:-3]
    elif len(w) > 4 and w.endswith("ed"):
        w = w[:-2]
    elif len(w) > 4 and w.endswith("es"):
        w = w[:-1]
    elif len(w) > 3 and w.endswith("s"):
        w = w[:-1]
    if len(w) > 3 and w[-1] == w[-2] and w[-1] not in "sl" and w[-1].isalpha():
        w = w[:-1]
    if len(w) > 4 and w.endswith("e"):
        w = w[:-1]
    return w


def is_money_col(name: str) -> bool:
    return any(k in name for k in MONEY_WORDS)


def is_age_col(name: str) -> bool:
    return name == "age" or name.endswith("_age") or name.startswith("age_")


def is_duration_col(name: str) -> bool:
    return any(k in name for k in ("duration", "minutes", "hours", "length", "time_taken"))


@dataclass
class Tok:
    text: str
    start: int
    end: int
    stem: str
    num: Optional[float] = None       # numeric value (after unit scaling)
    money: bool = False               # carries a currency marker
    used: bool = False                # consumed by some construct


# --------------------------------------------------------------- the parser
class IntentExtractor:
    def __init__(self, schema: Schema):
        self.schema = schema
        self._syn_index = []          # (stems tuple, column full name, synonym)
        for t in schema.tables.values():
            for c in t.columns.values():
                if c.name == t.pk or c.name.endswith("_id"):
                    continue
                for syn in c.synonyms:
                    st = tuple(stem(w) for w in re.findall(r"[a-z0-9]+", syn.lower()))
                    if st:
                        self._syn_index.append((st, c.full, tuple(re.findall(r"[a-z0-9]+", syn.lower()))))
        self._syn_index.sort(key=lambda x: -len(x[0]))
        self._noun_index = []
        for t in schema.tables.values():
            for n in t.nouns:
                self._noun_index.append((n.lower().split(), t.name))
        self._noun_index.sort(key=lambda x: -len(x[0]))

    # ------------------------------------------------------------ normalise
    @staticmethod
    def normalise(q: str) -> str:
        s = q.lower().replace("’", "'").replace("`", "'")
        s = re.sub(r"\bwhat's\b", "what is", s)
        s = re.sub(r"\b(does|do|did|is|are|was|were|has|have|should|would|could)n't\b", r"\1 not", s)
        s = re.sub(r"\bcan't\b", "can not", s)
        s = re.sub(r"\bwon't\b", "will not", s)
        s = re.sub(r"\b(\d+)\s*(?:yrs?|years?)\b", r"\1 years", s)
        s = re.sub(r"\b(" + "|".join(WORDNUM) + r")\b", lambda m: str(WORDNUM[m.group(1)]), s)
        s = re.sub(r"(₹|\$)\s*", r" \1", s)
        s = re.sub(r"\brs\.?\s*(?=\d)", " ₹", s)
        s = re.sub(r"\binr\s*(?=\d)", " ₹", s)
        return re.sub(r"\s+", " ", s).strip()

    def tokenize(self, s: str) -> list:
        toks = []
        for m in re.finditer(r"\d{4}-\d{2}-\d{2}|₹|\$|\d[\d,]*(?:\.\d+)?|[a-z0-9]+(?:[-+'][a-z0-9]+)*[+\-]?|,", s):
            text = m.group()
            if text.endswith(",") and text != ",":
                text = text[:-1]
            toks.append(Tok(text, m.start(), m.start() + len(text), stem(text)))
        # merge numbers with currency and scale words
        out = []
        i = 0
        while i < len(toks):
            t = toks[i]
            money = False
            if t.text in ("₹", "$") and i + 1 < len(toks) and re.fullmatch(r"\d[\d,]*(?:\.\d+)?", toks[i + 1].text):
                money = True
                i += 1
                t = toks[i]
            if re.fullmatch(r"\d[\d,]*(?:\.\d+)?", t.text):
                v = float(t.text.replace(",", ""))
                j = i + 1
                if j < len(toks) and toks[j].text in ("lakh", "lakhs", "lac", "lacs"):
                    v *= 1e5; money = True; j += 1
                elif j < len(toks) and toks[j].text in ("crore", "crores"):
                    v *= 1e7; money = True; j += 1
                elif j < len(toks) and toks[j].text in ("k", "thousand"):
                    v *= 1e3; j += 1
                elif j < len(toks) and toks[j].text in ("million",):
                    v *= 1e6; j += 1
                if j < len(toks) and toks[j].text in ("rupees", "rupee", "inr", "rs", "dollars", "₹"):
                    money = True; j += 1
                nt = Tok(t.text, t.start, toks[j - 1].end, t.text, num=v, money=money)
                out.append(nt)
                i = j
                continue
            out.append(t)
            i += 1
        return out

    # --------------------------------------------------------------- helpers
    @staticmethod
    def _match_seq(toks, i, words, use_stem=False) -> bool:
        if i < 0 or i + len(words) > len(toks):
            return False
        for k, w in enumerate(words):
            tw = toks[i + k].stem if use_stem else toks[i + k].text
            if tw != w:
                return False
        return True

    def _syn_mentions(self, toks):
        """All column-synonym mentions: list of (start_idx, end_idx, column)."""
        found = []
        noun_words = {w for words, _ in self._noun_index for w in words}
        for st, col, words in self._syn_index:
            n = len(st)
            for i in range(len(toks) - n + 1):
                if all(toks[i + k].stem == st[k] for k in range(n)):
                    exact = all(toks[i + k].text == words[k] for k in range(n))
                    if not exact and any(toks[i + k].text in noun_words for k in range(n)):
                        continue          # "orders" (entity noun) is not the column word "ordered"
                    found.append((i, i + n, col, exact))
        # prefer exact mentions: drop stem-only mentions that overlap an exact one
        exact_spans = [(a, b) for a, b, _, e in found if e]
        out = []
        for a, b, c, e in found:
            if not e and any(not (b <= x or a >= y) for x, y in exact_spans):
                continue
            out.append((a, b, c))
        return out

    # --------------------------------------------------------------- extract
    def extract(self, question: str) -> USIG:
        s = self.normalise(question)
        toks = self.tokenize(s)
        g = USIG()
        notes = []

        # ---------- 1. categorical value linking (on token sequence)
        values = self._link_values(s, toks)

        # ---------- 2. nouns and column synonyms
        nouns = self._link_nouns(toks)
        syns = self._syn_mentions(toks)

        # a value word that is really a column word of a mentioned entity
        # ("count the accounts for each branch" - 'Branch' is also a channel value)
        noun_tables = {t for _, t in nouns}
        self._noun_tables = noun_tables
        keep = []
        for v in values:
            word = " ".join(t.text for t in toks[v["i"]:v["j"]])
            vt = {c.split(".")[0] for c in v["cols"]}
            col_word = any(c.split(".")[0] in noun_tables and (a, b) == (v["i"], v["j"]) for a, b, c in syns) \
                or any(word in [x.lower() for x in self.schema.column(f"{t}.{cn}").synonyms]
                       for t in noun_tables for cn in self.schema.tables[t].columns)
            if col_word and not (vt & noun_tables):
                for k in range(v["i"], v["j"]):
                    toks[k].used = False
                continue
            keep.append(v)
        values = keep

        # entity tables
        tables_order = []
        for pos, t in sorted(nouns):
            if t not in tables_order:
                tables_order.append(t)
        value_tables = [v["col"].split(".")[0] for v in values]
        primary = tables_order[0] if tables_order else None

        # ---------- 3. dates
        date_mentions = self._find_dates(toks)

        # ---------- 4. limits / superlatives / aggregation / grouping
        shape = self._shape(toks, syns, nouns)

        # candidate tables for column resolution
        cand = list(tables_order)
        for t in value_tables:
            if t not in cand:
                cand.append(t)
        if primary is None:
            # no entity noun: use first linked value / column
            first = sorted([(v["i"], v["col"].split(".")[0]) for v in values] +
                           [(a, c.split(".")[0]) for a, _, c in syns if self._col_type(c) != "text"])
            if first:
                primary = first[0][1]
                if primary not in cand:
                    cand.insert(0, primary)
            elif self.schema.tables:
                primary = next(iter(self.schema.tables))
                cand.append(primary)
        cand_all = cand + [t for t in self.schema.tables if t not in cand]

        # resolve value columns that are ambiguous (same value in several columns)
        for v in values:
            v["col"] = self._choose_value_column(v, toks, syns, cand)
        # merge "Chennai or Madurai" into IN lists and attach negation
        constraints = self._value_constraints(values, toks)

        # resolve span / hint references of the shape against candidate tables
        for key in ("agg_col", "order_col", "group_col"):
            if isinstance(shape.get(key), tuple):
                shape[key] = self._resolve_ref(shape[key], syns, cand, cand_all)
        if shape.get("extra_proj"):
            shape["extra_proj"] = [self._resolve_ref(r, syns, cand, cand_all) if isinstance(r, tuple) else r
                                   for r in shape["extra_proj"]]
            shape["extra_proj"] = [r for r in shape["extra_proj"] if r]
        for key in ("agg_col", "order_col", "group_col"):
            if key in shape and shape[key] is None:
                shape.pop(key)
                if key == "order_col":
                    shape.pop("limit", None)
        if shape.get("agg") and shape["agg"] != "count" and not shape.get("agg_col"):
            shape.pop("agg")

        # ---------- 5. date constraints
        for dm in date_mentions:
            col = self._choose_date_column(dm, toks, syns, cand, cand_all)
            if col is None:
                continue
            constraints.extend(self._date_constraints(dm, col))

        # ---------- 6. numeric constraints
        constraints.extend(self._numeric_constraints(toks, syns, cand, cand_all, shape))

        # ---------- 7. "no seats left", "out of stock"
        constraints.extend(self._zero_constraints(toks, syns, cand))

        # ---------- assemble
        g.constraints = self._dedupe(constraints)
        tabs = []
        for t in [primary] + cand:
            if t and t not in tabs:
                tabs.append(t)
        used_tables = {primary} | {c.column.split(".")[0] for c in g.constraints}
        agg_col = shape.get("agg_col")
        if agg_col and agg_col != "*":
            used_tables.add(agg_col.split(".")[0])
        if shape.get("order_col"):
            used_tables.add(shape["order_col"].split(".")[0])
        if shape.get("group_col"):
            used_tables.add(shape["group_col"].split(".")[0])
        used_tables |= set(tables_order)
        g.entities = {t for t in used_tables if t}
        g.primary_entity = primary
        self._joins(g)

        # projections
        name_col = f"{primary}.{self.schema.tables[primary].name_col}" if primary else "*"
        if shape.get("group_col"):
            agg = shape.get("agg") or "count"
            g.group_by = [shape["group_col"]]
            g.projections = [(None, shape["group_col"]), (agg, shape.get("agg_col") or "*")]
        elif shape.get("agg"):
            g.projections = [(shape["agg"], shape.get("agg_col") or "*")]
        else:
            g.projections = [(None, name_col)] + [(None, c) for c in shape.get("extra_proj", [])]
        if shape.get("order_col"):
            g.order_by = [(None, shape["order_col"], shape["order_dir"])]
            g.limit = shape.get("limit")
        elif shape.get("limit") is not None:
            g.limit = shape["limit"]
        g.notes = notes
        if toks and toks[0].text in ("delete", "remove", "erase"):
            g.action = "delete"
            g.projections = []
            g.order_by, g.limit, g.group_by = [], None, []
        return g

    # ----------------------------------------------------------- value linking
    def _link_values(self, s, toks):
        cands = []
        for t in self.schema.tables.values():
            for c in t.columns.values():
                for val in c.values:
                    if not isinstance(val, str) or not val.strip():
                        continue
                    vw = self.tokenize(self.normalise(val))
                    vt = [x.text for x in vw]
                    if not vt:
                        continue
                    n = len(vt)
                    for i in range(len(toks) - n + 1):
                        if all(toks[i + k].text == vt[k] for k in range(n)):
                            cands.append({"i": i, "j": i + n, "col": c.full, "value": val, "cols": [c.full]})
                    # fuzzy single-token stem prefix ("cardiologist" -> "cardiology")
                    if n == 1 and len(vt[0]) >= 7 and vt[0].isalpha():
                        for i, tk in enumerate(toks):
                            w = tk.text
                            if w != vt[0] and len(w) >= 7 and w.isalpha():
                                p = len(os.path.commonprefix([w, vt[0]]))
                                if p >= 7 and p >= 0.75 * min(len(w), len(vt[0])):
                                    cands.append({"i": i, "j": i + 1, "col": c.full, "value": val, "cols": [c.full],
                                                  "fuzzy": True})
        # group by span; keep longest spans, drop overlaps
        cands.sort(key=lambda x: (-(x["j"] - x["i"]), x.get("fuzzy", False), x["i"]))
        chosen = []
        for cnd in cands:
            clash = [c for c in chosen if not (cnd["j"] <= c["i"] or cnd["i"] >= c["j"])]
            if not clash:
                chosen.append(dict(cnd))
            else:
                for c in clash:
                    if c["i"] == cnd["i"] and c["j"] == cnd["j"] and cnd["col"] not in c["cols"] \
                            and c.get("fuzzy", False) == cnd.get("fuzzy", False):
                        c["cols"].append(cnd["col"])
        # a value word that is really a column word ("Chennai branch") is not a value
        out = []
        for c in sorted(chosen, key=lambda x: x["i"]):
            word = " ".join(t.text for t in toks[c["i"]:c["j"]])
            prev = out[-1] if out else None
            if prev and prev["j"] == c["i"]:
                prev_cols = prev["cols"]
                if any(word in (self.schema.column(pc).synonyms if self.schema.column(pc) else []) for pc in prev_cols):
                    continue
            out.append(c)
        for c in out:
            for k in range(c["i"], c["j"]):
                toks[k].used = True
        return out

    def _link_nouns(self, toks):
        found = []
        taken = set()
        for words, t in self._noun_index:
            n = len(words)
            for i in range(len(toks) - n + 1):
                if any(k in taken for k in range(i, i + n)) or toks[i].used:
                    continue
                if all(toks[i + k].text == words[k] for k in range(n)):
                    # "order" inside "in descending order of"
                    prev = toks[i - 1].text if i > 0 else ""
                    nxt = toks[i + n].text if i + n < len(toks) else ""
                    if words == ["order"] and (prev in ("ascending", "descending", "in", "sorted", "sort", "the")
                                               and nxt in ("of", "by")):
                        continue
                    found.append((i, t))
                    taken.update(range(i, i + n))
        return found

    def _col_type(self, full):
        c = self.schema.column(full)
        return c.type if c else "text"

    def _choose_value_column(self, v, toks, syns, cand):
        cols = v["cols"]
        if len(cols) == 1:
            return cols[0]
        # cue words right before the value ("from Chennai" -> origin, "to Delhi" -> destination)
        best, best_score = None, -1e9
        for col in cols:
            score = 0.0
            for a, b, c in syns:
                if c == col and b <= v["i"] and v["i"] - b <= 3:
                    score = max(score, 10 - (v["i"] - b))
            t = col.split(".")[0]
            if t in cand:
                score += 3 - 0.5 * cand.index(t)
            if t in self._noun_tables:
                score += 8
            if score > best_score:
                best, best_score = col, score
        return best

    def _negated(self, toks, i) -> bool:
        j = i - 1
        steps = 0
        while j >= 0 and steps < 5:
            w = toks[j].text
            if w in NEG_WORDS and not toks[j].used:
                return True
            if j > 0 and (toks[j - 1].text, w) in NEG_PHRASES:
                return True
            if w in BOUNDARY and w not in ("who", "that", "which", "whose"):
                break
            j -= 1
            steps += 1
        return False

    def _value_constraints(self, values, toks):
        out = []
        k = 0
        while k < len(values):
            v = values[k]
            group = [v]
            while k + 1 < len(values):
                nxt = values[k + 1]
                between = [t.text for t in toks[group[-1]["j"]:nxt["i"]]]
                if nxt["col"] == v["col"] and between and all(w in ("or", "and", ",", "nor") for w in between):
                    group.append(nxt)
                    k += 1
                else:
                    break
            neg = self._negated(toks, v["i"])
            if len(group) == 1:
                out.append(Constraint(v["col"], "!=" if neg else "=", v["value"],
                                      " ".join(t.text for t in toks[v["i"]:v["j"]])))
            else:
                vals = tuple(x["value"] for x in group)
                out.append(Constraint(v["col"], "NOT IN" if neg else "IN", vals,
                                      " ".join(t.text for t in toks[group[0]["i"]:group[-1]["j"]])))
            k += 1
        return out

    # -------------------------------------------------------------- dates
    def _find_dates(self, toks):
        res = []
        n = len(toks)
        i = 0
        while i < n:
            t = toks[i]
            if t.used:
                i += 1
                continue
            m = re.fullmatch(r"(\d{4})-(\d{2})-(\d{2})", t.text)
            span = None
            if m:
                span = (i, i + 1, "day", int(m.group(1)), int(m.group(2)), int(m.group(3)))
            elif t.num is not None and i + 2 < n and toks[i + 1].text in MONTHS and toks[i + 2].num is not None \
                    and 1900 <= toks[i + 2].num <= 2100 and 1 <= t.num <= 31:
                span = (i, i + 3, "day", int(toks[i + 2].num), MONTHS[toks[i + 1].text], int(t.num))
            elif t.text in MONTHS and i + 1 < n and toks[i + 1].num is not None:
                if i + 2 < n and toks[i + 2].num is not None and 1900 <= toks[i + 2].num <= 2100 \
                        and 1 <= toks[i + 1].num <= 31:
                    span = (i, i + 3, "day", int(toks[i + 2].num), MONTHS[t.text], int(toks[i + 1].num))
                elif 1900 <= toks[i + 1].num <= 2100:
                    span = (i, i + 2, "month", int(toks[i + 1].num), MONTHS[t.text], 1)
            elif t.num is not None and float(t.num).is_integer() and 1900 <= t.num <= 2100 and not t.money \
                    and "," not in t.text:
                prev = toks[i - 1].text if i > 0 else ""
                prev2 = " ".join(x.text for x in toks[max(0, i - 3):i])
                if prev in {w for w, _ in DATE_CMP} or any(p in prev2 for p in ("prior to", "on or", "later than",
                                                                                "earlier than", "no later")) \
                        or prev in ("year", "onwards"):
                    span = (i, i + 1, "year", int(t.num), 1, 1)
            if span:
                a, b = span[0], span[1]
                op = "in"
                j = a - 1
                # look back for comparator phrase
                for phrase, o in DATE_CMP:
                    w = phrase.split()
                    if self._match_seq(toks, a - len(w), w):
                        op = o
                        break
                if b < n and toks[b].text in ("onwards", "onward") and op in ("in", ">="):
                    op = ">="
                if b + 1 < n and (toks[b].text, toks[b + 1].text) in (("or", "later"), ("or", "after")):
                    op = ">="
                for k in range(a, b):
                    toks[k].used = True
                res.append({"i": a, "j": b, "gran": span[2], "y": span[3], "m": span[4], "d": span[5], "op": op})
                i = b
                continue
            i += 1
        return res

    def _choose_date_column(self, dm, toks, syns, cand, cand_all):
        best, best_score = None, -1e9
        for t in cand_all:
            for c in self.schema.tables[t].columns.values():
                if c.type != "date":
                    continue
                score = 0.0
                for a, b, col in syns:
                    if col == c.full:
                        dist = dm["i"] - b if b <= dm["i"] else a - dm["j"]
                        if dist >= 0:
                            score = max(score, 8 - 0.5 * dist)
                if t in cand:
                    score += 3 - 0.5 * cand.index(t)
                else:
                    score -= 5
                if score > best_score:
                    best, best_score = c.full, score
        return best

    @staticmethod
    def _date_constraints(dm, col):
        y, m, d, gran, op = dm["y"], dm["m"], dm["d"], dm["gran"], dm["op"]
        if gran == "year":
            lo, hi = f"{y}-01-01", f"{y}-12-31"
        elif gran == "month":
            lo, hi = f"{y}-{m:02d}-01", f"{y}-{m:02d}-{calendar.monthrange(y, m)[1]:02d}"
        else:
            lo = hi = f"{y}-{m:02d}-{d:02d}"
        src = f"date:{op}:{lo}"
        if op == "in":
            if lo == hi:
                return [Constraint(col, "=", lo, src)]
            return [Constraint(col, ">=", lo, src), Constraint(col, "<=", hi, src)]
        if op == ">":
            return [Constraint(col, ">", hi, src)]
        if op == ">=":
            return [Constraint(col, ">=", lo, src)]
        if op == "<":
            return [Constraint(col, "<", lo, src)]
        return [Constraint(col, "<=", hi, src)]

    # ------------------------------------------------------------ numbers
    def _comparator(self, toks, i):
        """Comparator for number at token i -> (op, hint, consumed indexes)."""
        # suffix: "60 or more", "90 marks or above"
        for skip in range(0, 4):
            k = i + 1 + skip
            for phrase, op in CMP_SUFFIX:
                w = phrase.split()
                if self._match_seq(toks, k, w):
                    between = toks[i + 1:k]
                    if all(not x.num and x.text not in BOUNDARY - {"and", "or"} for x in between):
                        return op, None, list(range(k, k + len(w)))
            if k < len(toks) and (toks[k].text in BOUNDARY or toks[k].num is not None):
                break
        # prefix: "more than 60", "above the age of 60", "fee doesn't exceed rs 800"
        j = i - 1
        skipped = 0
        while j >= 0 and skipped <= 4:
            for phrase, op in CMP_PREFIX:
                w = phrase.split()
                if self._match_seq(toks, j - len(w) + 1, w):
                    hint = CMP_HINT.get(phrase)
                    idx = list(range(j - len(w) + 1, j + 1))
                    if phrase == "after":
                        continue
                    return op, hint, idx
            if toks[j].text in BOUNDARY or toks[j].num is not None:
                break
            j -= 1
            skipped += 1
        # "aged 60" -> equality with age hint
        if i > 0 and toks[i - 1].text == "aged":
            return "=", "age", []
        return "=", None, []

    def _numeric_constraints(self, toks, syns, cand, cand_all, shape):
        out = []
        n = len(toks)
        skip = set(shape.get("limit_tokens", []))
        i = 0
        while i < n:
            t = toks[i]
            if t.num is None or t.used or i in skip:
                i += 1
                continue
            # ranges: "between X and Y", "from X to Y"
            if i >= 1 and toks[i - 1].text in ("between", "from") and i + 2 < n and toks[i + 2].num is not None \
                    and toks[i + 1].text in ("and", "to", "-"):
                lo_t, hi_t = t, toks[i + 2]
                money = lo_t.money or hi_t.money
                col, scale = self._choose_num_column(toks, i, syns, cand, cand_all, None, money,
                                                     (lo_t.num + hi_t.num) / 2, end=i + 2)
                if col:
                    out.append(Constraint(col, ">=", self._cast(col, lo_t.num * scale), f"between {lo_t.text}"))
                    out.append(Constraint(col, "<=", self._cast(col, hi_t.num * scale), f"and {hi_t.text}"))
                for k in (i - 1, i, i + 1, i + 2):
                    toks[k].used = True
                i += 3
                continue
            op, hint, consumed = self._comparator(toks, i)
            col, scale = self._choose_num_column(toks, i, syns, cand, cand_all, hint, t.money, t.num)
            if col:
                for k in consumed:
                    toks[k].used = True
                t.used = True
                out.append(Constraint(col, op, self._cast(col, t.num * scale), f"{op} {t.text}"))
            i += 1
        return out

    def _cast(self, col, v):
        typ = self._col_type(col)
        if typ == "int" and float(v).is_integer():
            return int(v)
        return float(v)

    def _choose_num_column(self, toks, i, syns, cand, cand_all, hint, money, value, end=None):
        end = i if end is None else end
        # units: "2 hours" for a minutes column
        unit_hours = end + 1 < len(toks) and toks[end + 1].text in ("hours", "hour", "hrs", "hr")
        best, best_score, best_scale = None, -1e9, 1.0
        for t in cand_all:
            for c in self.schema.tables[t].columns.values():
                if c.type not in ("int", "real") or c.name == self.schema.tables[t].pk or c.name.endswith("_id"):
                    continue
                score = 0.0
                # synonym proximity (same clause)
                for a, b, col in syns:
                    if col != c.full:
                        continue
                    if b <= i:
                        between = toks[b:i]
                        if any(x.text in BOUNDARY - {"with", "having"} for x in between) or len(between) > 6:
                            continue
                        score = max(score, 10 - 0.6 * len(between))
                    elif a > end:
                        between = [x for x in toks[end + 1:a] if x.text not in SUFFIX_WORDS]
                        if any(x.text in BOUNDARY or x.num is not None for x in between) or len(between) > 3:
                            continue
                        score = max(score, 9.5 - 0.6 * len(between))
                if hint == "age" and is_age_col(c.name):
                    score += 7
                if hint == "money" and is_money_col(c.name):
                    score += 7
                if hint == "duration" and is_duration_col(c.name):
                    score += 7
                if money:
                    score += 4 if is_money_col(c.name) or c.type == "real" and is_money_col(c.name) else -4
                scale = 60.0 if (unit_hours and "minute" in c.name) else 1.0
                v = value * scale
                if c.min is not None and c.max is not None:
                    lo, hi = float(c.min), float(c.max)
                    span = max(hi - lo, 1e-9)
                    if lo - 0.05 * span <= v <= hi + 0.05 * span:
                        score += 3
                    else:
                        score -= 6
                if t in cand:
                    score += 2 - 0.4 * cand.index(t)
                else:
                    score -= 6
                if score > best_score:
                    best, best_score, best_scale = c.full, score, scale
        if best_score < -3:
            return None, 1.0
        return best, best_scale

    def _zero_constraints(self, toks, syns, cand):
        out = []
        for i, t in enumerate(toks):
            if t.text == "out" and i + 2 < len(toks) and toks[i + 1].text == "of" and toks[i + 2].stem == "stock":
                col = next((c for a, b, c in syns if a == i + 2), None)
                if col and col.split(".")[0] in cand:
                    out.append(Constraint(col, "=", 0, "out of stock"))
            if t.text in ("no", "zero") and not t.used:
                for a, b, col in syns:
                    if 1 <= a - i <= 2 and self._col_type(col) in ("int", "real") and col.split(".")[0] in cand:
                        out.append(Constraint(col, "=", 0, "no " + toks[a].text))
                        break
        return out

    # -------------------------------------------------------------- shape
    def _shape(self, toks, syns, nouns):
        """Aggregation, superlative (order+limit), grouping, explicit projections."""
        shape: dict = {"limit_tokens": []}
        words = [t.text for t in toks]
        n = len(toks)
        noun_pos = {p for p, _ in nouns}

        def next_col(i, kinds=("int", "real"), window=3):
            for a, b, c in sorted(syns, key=lambda x: (x[0], -(x[1] - x[0]))):
                if i < a <= i + window and self._col_type(c) in kinds:
                    return ("span", a, b, kinds), a, b
            return None, None, None

        def num_col_by_hint(hint):
            return ("hint", hint)

        first = words[0] if words else ""
        asks_entity = first in LIST_STARTS or (first in ("what",) and len(words) > 1 and words[1] == "are")

        # COUNT
        txt = " ".join(words)
        if re.search(r"\bhow many\b|\bnumber of\b|\bcount\b|\bno\. of\b|\btotal number\b", txt):
            shape["agg"], shape["agg_col"] = "count", "*"

        # AVG / SUM / MAX / MIN with a column
        for i, w in enumerate(words):
            if w in AGG_WORDS and not (w == "total" and i + 1 < n and words[i + 1] == "number"):
                col, a, b = next_col(i, window=4)
                if col is None and w in ("average", "mean", "avg", "total", "sum"):
                    # "average of the bills": look a little further
                    col, a, b = next_col(i, window=6)
                if col is None:
                    continue
                agg = AGG_WORDS[w]
                if agg in ("max", "min"):
                    # "which doctor has the highest fee" / "the 5 products with the highest price"
                    # -> ORDER BY + LIMIT, not MAX
                    prev = words[max(0, i - 2):i]
                    if first in ("which", "who") or any(p in prev for p in ("with", "has", "have", "by")) \
                            or (b < n and b in noun_pos) or (i > 0 and toks[i - 1].num is not None) \
                            or any(toks[j].num is not None and j + 1 in noun_pos for j in range(i)):
                        continue
                if shape.get("agg") == "count" and agg in ("max", "min"):
                    continue
                shape["agg"], shape["agg_col"] = agg, col
                break

        # superlatives -> ORDER BY + LIMIT (entity requests)
        if shape.get("agg") in (None,):
            for i, w in enumerate(words):
                if w in SUPER_DESC or w in SUPER_ASC:
                    if w in ("most", "least") and i > 0 and words[i - 1] == "at":
                        continue
                    desc = w in SUPER_DESC
                    col, a, b = next_col(i, window=3)
                    if col is None and w in SUPER_HINT:
                        hint = SUPER_HINT[w]
                        col = ("hint", hint)
                    if col is None:
                        # "top 10 students by cgpa"
                        for k in range(i + 1, n):
                            if words[k] == "by":
                                col, a, b = next_col(k, window=3)
                                break
                    if col is None and w in ("top", "bottom"):
                        continue
                    if col is None:
                        continue
                    if w == "least" and i + 1 < n and words[i + 1] == "expensive":
                        desc = False
                    shape["order_col"], shape["order_dir"] = col, "DESC" if desc else "ASC"
                    # k
                    k = None
                    for j in [i + 1, i - 1, i - 2] + [j for j in range(i) if j + 1 in noun_pos]:
                        if 0 <= j < n and toks[j].num is not None and toks[j].num <= 100 and j not in noun_pos \
                                and float(toks[j].num).is_integer():
                            k = int(toks[j].num)
                            shape["limit_tokens"].append(j)
                            break
                    shape["limit"] = k if k is not None else 1
                    break

        # explicit "top N" / "first N" without superlative column found
        for i, w in enumerate(words):
            if w in ("top", "first") and i + 1 < n and toks[i + 1].num is not None:
                shape["limit_tokens"].append(i + 1)
                if "order_col" in shape:
                    shape["limit"] = int(toks[i + 1].num)

        # ORDER BY phrases
        m = None
        for i, w in enumerate(words):
            if (w in ("sorted", "ordered", "sort", "order", "arranged", "rank", "ranked") and i + 1 < n
                    and words[i + 1] in ("by", "of", "on")) or (w == "order" and i > 0 and
                                                              words[i - 1] in ("ascending", "descending")):
                col, a, b = next_col(i, kinds=("int", "real", "date", "text"), window=4)
                if col:
                    rest = " ".join(words[i:])
                    desc = bool(re.search(r"descending|desc\b|highest to lowest|high to low|largest first|"
                                          r"highest first|decreasing", rest)) or \
                        (i > 0 and words[i - 1] == "descending")
                    shape["order_col"], shape["order_dir"] = col, "DESC" if desc else "ASC"
                    shape["limit"] = None
                    m = col
                    break
        if m and not shape.get("agg"):
            shape["extra_proj"] = [m]

        # GROUP BY: "per city", "in each ward", "for every department", "count ... by city"
        for i, w in enumerate(words):
            if w in ("each", "per", "every") or (w == "by" and shape.get("agg") and "order_col" not in shape):
                col, a, b = next_col(i, kinds=("text",), window=2 if w != "by" else 1)
                if col and w == "by" and b < n and toks[b].used:
                    col = None
                if col:
                    shape["group_col"] = col
                    if not shape.get("agg"):
                        shape["agg"], shape["agg_col"] = "count", "*"
                    break

        # explicit projections: "the name and age of ..."
        for i, w in enumerate(words):
            if w in ("name", "names") and i + 2 < n and words[i + 1] == "and":
                col, a, b = next_col(i + 1, kinds=("int", "real", "date", "text"), window=2)
                if col:
                    shape["extra_proj"] = [col]
        return shape

    def _resolve_ref(self, ref, syns, cand, cand_all):
        if ref[0] == "span":
            _, a, b, kinds = ref
            options = [c for x, y, c in syns if x == a and y == b and self._col_type(c) in kinds]
            if not options:
                return None
            options.sort(key=lambda c: cand.index(c.split(".")[0]) if c.split(".")[0] in cand else 99)
            return options[0]
        hint = ref[1]
        for t in cand_all:
            tab = self.schema.tables[t]
            for c in tab.columns.values():
                if c.name == tab.pk or c.name.endswith("_id"):
                    continue
                if (hint == "money" and c.type in ("int", "real") and is_money_col(c.name)) or \
                        (hint == "age" and c.type in ("int", "real") and is_age_col(c.name)) or \
                        (hint == "duration" and c.type in ("int", "real") and is_duration_col(c.name)) or \
                        (hint == "date" and c.type == "date"):
                    return c.full
        return None

    # ----------------------------------------------------------- assembly
    @staticmethod
    def _dedupe(cs):
        seen, out = set(), []
        for c in cs:
            k = (c.column, c.op, str(c.value))
            if k not in seen:
                seen.add(k)
                out.append(c)
        return out

    def _joins(self, g: USIG):
        tabs = [g.primary_entity] + sorted(t for t in g.entities if t != g.primary_entity)
        placed = {tabs[0]}
        for t in tabs[1:]:
            j = None
            for p in list(placed):
                j = self.schema.join_path(p, t)
                if j:
                    break
            if j is None:
                # bridge table
                for bt in self.schema.tables:
                    if bt in placed or bt == t:
                        continue
                    j1 = next((self.schema.join_path(p, bt) for p in placed if self.schema.join_path(p, bt)), None)
                    j2 = self.schema.join_path(bt, t)
                    if j1 and j2:
                        g.joins += [j1]
                        g.entities.add(bt)
                        placed.add(bt)
                        j = j2
                        break
            if j:
                g.joins.append(j)
                placed.add(t)
