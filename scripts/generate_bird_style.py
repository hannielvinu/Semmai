"""Generate the Semmai BIRD-style text-to-SQL benchmark.

Questions are produced from many paraphrase frames (several leads, comparator
phrasings, number / currency / date formats) over the five databases and
paired with gold SQL. The output follows BIRD's dev.json format so the same
evaluation code runs on the official BIRD data.

Output: data/bird_style/questions.json

Usage:
    python scripts/generate_bird_style.py
"""
from __future__ import annotations

import json
import random
import sqlite3
import sys
from pathlib import Path

if hasattr(__import__("sys").stdout, "reconfigure"):
    __import__("sys").stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from semmai.schema import Schema                       # noqa: E402
from semmai.sql_parser import render_sql               # noqa: E402
from semmai.usig import Constraint, USIG               # noqa: E402

DATA = ROOT / "data" / "bird_style"
MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September",
          "October", "November", "December"]

# ----------------------------------------------------------------- lexicons
CMP = {
    ">": ["greater than {v}", "more than {v}", "above {v}", "over {v}", "exceeding {v}", "higher than {v}"],
    ">=": ["at least {v}", "greater than or equal to {v}", "not less than {v}", "no less than {v}",
           "{v} or more", "{v} or above", "a minimum of {v}"],
    "<": ["less than {v}", "below {v}", "under {v}", "lower than {v}"],
    "<=": ["at most {v}", "less than or equal to {v}", "not more than {v}", "no more than {v}",
           "{v} or less", "{v} or below", "up to {v}", "a maximum of {v}"],
}
COUNT_CMP = {">": ["more than {v}"], ">=": ["at least {v}", "{v} or more"], "<": ["fewer than {v}", "less than {v}"],
             "<=": ["at most {v}", "{v} or fewer"]}
AGE_CMP = {
    ">": ["older than {v}", "aged over {v}", "above {v} years of age", "over {v} years old"],
    ">=": ["aged {v} or above", "at least {v} years old", "aged {v} or older"],
    "<": ["younger than {v}", "under {v} years old", "below {v} years of age"],
    "<=": ["aged {v} or below", "at most {v} years old", "aged {v} or younger"],
}
DATE_CMP = {">": ["after {d}"], ">=": ["since {d}", "on or after {d}", "from {d} onwards"],
            "<": ["before {d}", "prior to {d}"], "<=": ["on or before {d}", "no later than {d}"]}
LIST_LEADS = ["Show", "List", "Find", "Get", "Display", "Give me", "Retrieve", "Show me", "List all",
              "Find all", "Show the names of", "List the names of"]

# --------------------------------------------------------- entity configs
# num: col -> (phrases, value-format, kind)   kind in {age, plain, count}
# cat: col -> (positive frames, negative frames)
# date: col -> verb phrase used before date comparator
ENT = {
    "hospital": {
        "patients": dict(
            num={"age": (["age"], "int", "age"), "bill_amount": (["bill amount", "hospital bill"], "money", "plain")},
            cat={"city": (["from {v}", "living in {v}", "based in {v}"], ["not from {v}", "living outside {v}", "except those from {v}"]),
                 "ward": (["in the {v} ward", "admitted to the {v} ward"], ["not in the {v} ward", "outside the {v} ward"]),
                 "blood_group": (["with blood group {v}"], ["whose blood group is not {v}"])},
            date={"admission_date": "admitted"},
            group={"city": "city", "ward": "ward", "blood_group": "blood group"}),
        "doctors": dict(
            num={"experience_years": (["years of experience"], "int", "count"),
                 "consultation_fee": (["consultation fee"], "money", "plain")},
            cat={"specialty": (["specialising in {v}", "in the {v} department", "whose specialty is {v}"],
                               ["not specialising in {v}", "outside {v}", "other than {v} specialists"]),
                 "city": (["practising in {v}", "based in {v}"], ["not based in {v}"])},
            date={}, group={"specialty": "specialty", "city": "city"}),
    },
    "retail": {
        "customers": dict(
            num={"age": (["age"], "int", "age")},
            cat={"city": (["from {v}", "located in {v}", "living in {v}"], ["not from {v}", "outside {v}", "except those from {v}"]),
                 "loyalty_tier": (["in the {v} tier", "with {v} membership"], ["not in the {v} tier"])},
            date={"join_date": "who joined"},
            group={"city": "city", "loyalty_tier": "loyalty tier"}),
        "products": dict(
            num={"price": (["price"], "money", "plain"), "stock": (["units in stock"], "int", "count"),
                 "rating": (["rating"], "real", "plain")},
            cat={"category": (["in the {v} category", "under {v}"], ["not in the {v} category", "excluding {v}"])},
            date={}, group={"category": "category"}),
        "orders": dict(
            num={"amount": (["order amount", "amount"], "money", "plain")},
            cat={"payment_method": (["paid via {v}", "paid using {v}"], ["not paid via {v}"]),
                 "status": (["with status {v}", "that are {vl}"], ["whose status is not {v}"])},
            date={"order_date": "placed"},
            group={"payment_method": "payment method", "status": "status"}),
    },
    "university": {
        "students": dict(
            num={"cgpa": (["CGPA", "cgpa"], "real", "plain"), "age": (["age"], "int", "age")},
            cat={"department": (["from the {v} department", "in {v}", "studying {v}"], ["not in {v}", "outside the {v} department"]),
                 "city": (["from {v}", "whose hometown is {v}"], ["not from {v}"])},
            date={"enrollment_date": "enrolled"},
            group={"department": "department", "city": "city"}),
        "courses": dict(
            num={"credits": (["credits"], "int", "count")},
            cat={"department": (["offered by the {v} department", "in {v}"], ["not offered by {v}"])},
            date={}, group={"department": "department"}),
    },
    "finance": {
        "accounts": dict(
            num={"balance": (["balance"], "money", "plain"), "holder_age": (["age"], "int", "age")},
            cat={"branch": (["in the {v} branch", "at the {v} branch"], ["not in the {v} branch", "outside the {v} branch"]),
                 "account_type": (["of type {v}", "that are {v} accounts"], ["that are not {v} accounts"])},
            date={"opened_date": "opened"},
            group={"branch": "branch", "account_type": "account type"}),
        "transactions": dict(
            num={"amount": (["amount"], "money", "plain")},
            cat={"channel": (["made via {v}", "made through {v}"], ["not made via {v}"]),
                 "transaction_type": (["of type {v}"], ["that are not {v}"])},
            date={"transaction_date": "made"},
            group={"channel": "channel", "transaction_type": "transaction type"}),
    },
    "travel": {
        "flights": dict(
            num={"price": (["fare", "ticket price"], "money", "plain"),
                 "duration_minutes": (["duration in minutes"], "int", "plain"),
                 "seats_available": (["seats available"], "int", "count")},
            cat={"airline": (["operated by {v}", "run by {v}"], ["not operated by {v}"]),
                 "origin": (["from {v}", "departing from {v}"], ["not departing from {v}"]),
                 "destination": (["to {v}", "arriving in {v}"], ["not going to {v}"])},
            date={"departure_date": "departing"},
            group={"airline": "airline", "origin": "origin", "destination": "destination"}),
        "bookings": dict(
            num={"passenger_age": (["age"], "int", "age"), "amount": (["amount paid"], "money", "plain")},
            cat={"seat_class": (["in {v} class", "travelling in {v}"], ["not in {v} class"])},
            date={"booking_date": "booked"},
            group={"seat_class": "seat class"}),
    },
}
NOUN = {"patients": ("patients", "patient"), "doctors": ("doctors", "doctor"), "customers": ("customers", "customer"),
        "products": ("products", "product"), "orders": ("orders", "order"), "students": ("students", "student"),
        "courses": ("courses", "course"), "accounts": ("accounts", "account"),
        "transactions": ("transactions", "transaction"), "flights": ("flights", "flight"),
        "bookings": ("bookings", "booking")}


class Gen:
    def __init__(self, db_id: str, rng: random.Random):
        self.db = db_id
        self.rng = rng
        self.schema = Schema.from_bird_style(DATA, db_id)
        self.con = sqlite3.connect(self.schema.db_path)

    # ---------------------------------------------------------------- values
    def num_value(self, table, col, kind):
        lo, hi = self.con.execute(
            f"SELECT MIN({col}), MAX({col}) FROM {table}").fetchone()
        q = self.rng.uniform(0.15, 0.85)
        v = lo + (hi - lo) * q
        if kind == "money":
            mag = 10 ** max(1, len(str(int(v))) - 2)
            v = int(round(v / mag) * mag)
        elif kind == "int":
            v = int(round(v))
            if v > 20:
                v = int(round(v / 5) * 5)
        else:
            v = round(v, 1)
        return v

    def fmt_num(self, v, kind):
        r = self.rng.random()
        if kind == "money":
            if v >= 100000 and v % 100000 == 0 and r < 0.3:
                return f"{v // 100000} lakh rupees"
            return [f"₹{v:,}", f"Rs. {v:,}", f"{v} rupees", f"{v:,}", f"INR {v}"][self.rng.randrange(5)]
        return str(v)

    def cat_value(self, table, col):
        return self.rng.choice(self.schema.tables[table].columns[col].values)

    def date_value(self, table, col):
        lo, hi = self.con.execute(f"SELECT MIN({col}), MAX({col}) FROM {table}").fetchone()
        y = self.rng.randint(int(lo[:4]) + 1, int(hi[:4]))
        m = self.rng.randint(1, 12)
        d = self.rng.randint(1, 28)
        return y, m, d

    # -------------------------------------------------------------- clauses
    def num_clause(self, t, col):
        phrases, fmt, kind = ENT[self.db][t]["num"][col]
        op = self.rng.choice([">", ">=", "<", "<="])
        v = self.num_value(t, col, fmt)
        vs = self.fmt_num(v, fmt)
        p = self.rng.choice(phrases)
        if kind == "age" and self.rng.random() < 0.7:
            txt = self.rng.choice(AGE_CMP[op]).format(v=vs)
        elif kind == "count":
            txt = "with " + self.rng.choice(COUNT_CMP[op]).format(v=vs) + " " + p
        else:
            cmpp = self.rng.choice(CMP[op]).format(v=vs)
            frame = self.rng.choice(["with {p} {c}", "whose {p} is {c}", "having a {p} {c}", "with a {p} of {c}"])
            txt = frame.format(p=p, c=cmpp)
        return txt, [Constraint(f"{t}.{col}", op, v)]

    def cat_clause(self, t, col, neg=False):
        pos, negs = ENT[self.db][t]["cat"][col]
        v = self.cat_value(t, col)
        frame = self.rng.choice(negs if neg else pos)
        return frame.format(v=v, vl=v.lower()), [Constraint(f"{t}.{col}", "!=" if neg else "=", v)]

    def cat_in_clause(self, t, col):
        pos, _ = ENT[self.db][t]["cat"][col]
        vals = self.rng.sample(self.schema.tables[t].columns[col].values, 2)
        frame = self.rng.choice(pos)
        return frame.format(v=f"{vals[0]} or {vals[1]}", vl=f"{vals[0]} or {vals[1]}".lower()), \
            [Constraint(f"{t}.{col}", "IN", tuple(vals))]

    def date_clause(self, t, col):
        verb = ENT[self.db][t]["date"][col]
        y, m, d = self.date_value(t, col)
        mode = self.rng.random()
        if mode < 0.35:                               # whole year
            kind = self.rng.choice(["in", ">", ">=", "<"])
            if kind == "in":
                return f"{verb} in {y}", [Constraint(f"{t}.{col}", ">=", f"{y}-01-01"),
                                          Constraint(f"{t}.{col}", "<=", f"{y}-12-31")]
            txt = {">": f"after {y}", ">=": f"since {y}", "<": f"before {y}"}[kind]
            val = {">": f"{y}-12-31", ">=": f"{y}-01-01", "<": f"{y}-01-01"}[kind]
            return f"{verb} {txt}", [Constraint(f"{t}.{col}", kind, val)]
        if mode < 0.5:                                # month
            import calendar
            last = calendar.monthrange(y, m)[1]
            return f"{verb} in {MONTHS[m - 1]} {y}", [Constraint(f"{t}.{col}", ">=", f"{y}-{m:02d}-01"),
                                                     Constraint(f"{t}.{col}", "<=", f"{y}-{m:02d}-{last:02d}")]
        op = self.rng.choice([">", ">=", "<", "<="])
        iso = f"{y}-{m:02d}-{d:02d}"
        ds = self.rng.choice([f"{d} {MONTHS[m - 1]} {y}", f"{MONTHS[m - 1]} {d}, {y}", iso])
        return f"{verb} " + self.rng.choice(DATE_CMP[op]).format(d=ds), [Constraint(f"{t}.{col}", op, iso)]

    def between_clause(self, t, col):
        phrases, fmt, kind = ENT[self.db][t]["num"][col]
        vals = sorted({self.num_value(t, col, fmt) for _ in range(4)})
        if len(vals) < 2:
            return None
        a, b = vals[0], vals[-1]
        p = self.rng.choice(phrases)
        txt = self.rng.choice(["with {p} between {a} and {b}", "whose {p} is between {a} and {b}"]).format(
            p=p, a=self.fmt_num(a, fmt), b=self.fmt_num(b, fmt))
        return txt, [Constraint(f"{t}.{col}", ">=", a), Constraint(f"{t}.{col}", "<=", b)]

    def random_filter(self, t, kinds=("num", "cat", "date")):
        cfg = ENT[self.db][t]
        choices = [k for k in kinds if cfg.get(k)]
        k = self.rng.choice(choices)
        col = self.rng.choice(list(cfg[k]))
        if k == "num":
            return self.num_clause(t, col)
        if k == "cat":
            return self.cat_clause(t, col)
        return self.date_clause(t, col)

    # -------------------------------------------------------------- builders
    def base(self, t) -> USIG:
        g = USIG(entities={t}, primary_entity=t)
        return g

    def name_proj(self, t):
        return [(None, f"{t}.{self.schema.tables[t].name_col}")]

    def q_list(self, t, nfilters=1, neg=False, use_in=False, between=False):
        g = self.base(t)
        g.projections = self.name_proj(t)
        clauses = []
        cfg = ENT[self.db][t]
        if neg:
            col = self.rng.choice(list(cfg["cat"]))
            c, cs = self.cat_clause(t, col, neg=True)
            clauses.append(c); g.constraints += cs
        elif use_in:
            col = self.rng.choice(list(cfg["cat"]))
            c, cs = self.cat_in_clause(t, col)
            clauses.append(c); g.constraints += cs
        elif between:
            col = self.rng.choice([c for c, v in cfg["num"].items() if v[2] != "age"] or list(cfg["num"]))
            r = self.between_clause(t, col)
            if r is None:
                return None
            clauses.append(r[0]); g.constraints += r[1]
        used = {c.column for c in g.constraints}
        tries = 0
        while len(clauses) < nfilters and tries < 10:
            tries += 1
            c, cs = self.random_filter(t)
            if cs[0].column in used:
                continue
            used.add(cs[0].column)
            clauses.append(c); g.constraints += cs
        lead = self.rng.choice(LIST_LEADS)
        ent = NOUN[t][0]
        if lead in ("List all", "Find all"):
            pass
        q = f"{lead} {ent} " + " and ".join(clauses)
        if self.rng.random() < 0.25:
            q = f"Which {ent} are " + " and ".join(clauses) + "?"
            q = q.replace("are with", "have").replace("are having", "have").replace("are whose", "have")
            return self._fix_which(q), g, "list"
        return q + self.rng.choice([".", ""]), g, "list"

    def _fix_which(self, q):
        # "Which patients are older than 60?" is fine; repair awkward forms
        for a, b in [("are admitted", "were admitted"), ("are who joined", "joined"), ("are placed", "were placed"),
                     ("are made", "were made"), ("are opened", "were opened"), ("are enrolled", "were enrolled"),
                     ("are departing", "are departing"), ("are booked", "were booked"),
                     ("are that are", "are"), ("are specialising", "are specialising")]:
            q = q.replace(a, b, 1)
        return q

    def q_count(self, t):
        g = self.base(t)
        g.projections = [("count", "*")]
        c, cs = self.random_filter(t)
        g.constraints += cs
        ent = NOUN[t][0]
        q = self.rng.choice([f"How many {ent} {c} are there?", f"What is the number of {ent} {c}?",
                             f"Count the {ent} {c}.", f"How many {ent} {c}?", f"Give the total number of {ent} {c}."])
        return q, g, "count"

    def q_agg(self, t):
        cfg = ENT[self.db][t]
        col = self.rng.choice(list(cfg["num"]))
        phrases = cfg["num"][col][0]
        p = phrases[0] if cfg["num"][col][2] != "count" else phrases[0]
        agg, word = self.rng.choice([("avg", "average"), ("avg", "mean"), ("sum", "total"), ("max", "maximum"),
                                     ("max", "highest"), ("min", "minimum"), ("min", "lowest")])
        if agg == "sum" and cfg["num"][col][2] == "age":
            agg, word = "avg", "average"
        g = self.base(t)
        g.projections = [(agg, f"{t}.{col}")]
        filt = ""
        if self.rng.random() < 0.75:
            k = self.rng.choice([k for k in ("cat", "date") if cfg.get(k)])
            colf = self.rng.choice(list(cfg[k]))
            filt, cs = (self.cat_clause(t, colf) if k == "cat" else self.date_clause(t, colf))
            g.constraints += cs
            filt = " " + filt
        ent = NOUN[t][0]
        q = self.rng.choice([f"What is the {word} {p} of {ent}{filt}?", f"Find the {word} {p} among {ent}{filt}.",
                             f"Calculate the {word} {p} for {ent}{filt}."])
        return q, g, "aggregate"

    def q_topk(self, t):
        cfg = ENT[self.db][t]
        col = self.rng.choice(list(cfg["num"]))
        p = cfg["num"][col][0][0]
        desc = self.rng.random() < 0.6
        k = self.rng.choice([1, 3, 5, 10])
        g = self.base(t)
        g.projections = self.name_proj(t)
        filt = ""
        if self.rng.random() < 0.5 and cfg["cat"]:
            c, cs = self.cat_clause(t, self.rng.choice(list(cfg["cat"])))
            g.constraints += cs
            filt = " " + c
        g.order_by = [(None, f"{t}.{col}", "DESC" if desc else "ASC")]
        g.limit = k
        ent, sing = NOUN[t]
        hi = "highest" if desc else "lowest"
        if k == 1:
            q = self.rng.choice([f"Which {sing}{filt} has the {hi} {p}?", f"Find the {sing}{filt} with the {hi} {p}."])
        else:
            opts = [f"Show the {k} {ent}{filt} with the {hi} {p}.", f"Which {k} {ent}{filt} have the {hi} {p}?"]
            if desc:
                opts.append(f"List the top {k} {ent}{filt} by {p}.")
            else:
                opts.append(f"List the bottom {k} {ent}{filt} by {p}.")
            q = self.rng.choice(opts)
        return q, g, "top-k"

    def q_sorted(self, t):
        cfg = ENT[self.db][t]
        col = self.rng.choice(list(cfg["num"]))
        p = cfg["num"][col][0][0]
        g = self.base(t)
        g.projections = self.name_proj(t) + [(None, f"{t}.{col}")]
        c, cs = self.random_filter(t, kinds=("cat",))
        g.constraints += cs
        desc = self.rng.random() < 0.5
        g.order_by = [(None, f"{t}.{col}", "DESC" if desc else "ASC")]
        ent = NOUN[t][0]
        order_txt = self.rng.choice(["in descending order of {p}", "sorted by {p} from highest to lowest",
                                     "ordered by {p} descending"] if desc else
                                    ["in ascending order of {p}", "sorted by {p} from lowest to highest",
                                     "ordered by {p} ascending"]).format(p=p)
        q = f"{self.rng.choice(['Show', 'List'])} the name and {p} of {ent} {c} {order_txt}."
        return q, g, "order"

    def q_group(self, t):
        cfg = ENT[self.db][t]
        gcol = self.rng.choice(list(cfg["group"]))
        gp = cfg["group"][gcol]
        ent = NOUN[t][0]
        g = self.base(t)
        g.group_by = [f"{t}.{gcol}"]
        if self.rng.random() < 0.55:
            g.projections = [(None, f"{t}.{gcol}"), ("count", "*")]
            q = self.rng.choice([f"How many {ent} are there in each {gp}?", f"Show the number of {ent} per {gp}.",
                                 f"Count the {ent} for each {gp}."])
        else:
            col = self.rng.choice(list(cfg["num"]))
            p = cfg["num"][col][0][0]
            g.projections = [(None, f"{t}.{gcol}"), ("avg", f"{t}.{col}")]
            q = self.rng.choice([f"What is the average {p} of {ent} in each {gp}?",
                                 f"Show the average {p} of {ent} per {gp}."])
        return q, g, "group-by"


# ----------------------------------------------------------- join templates
def join_questions(gen: Gen) -> list:
    rng, s = gen.rng, gen.schema
    out = []

    def mk(tables, proj, conds, q, qtype="join", distinct=False):
        g = USIG(entities=set(tables), primary_entity=tables[0], projections=proj, constraints=conds,
                 distinct=distinct)
        for t in tables[1:]:
            g.joins.append(s.join_path(tables[0], t) or s.join_path(tables[1], t))
        out.append((q, g, qtype))

    db = gen.db
    for _ in range(6):
        if db == "hospital":
            city = gen.cat_value("patients", "city"); fee = gen.num_value("appointments", "fee", "money")
            age = gen.num_value("patients", "age", "int"); status = gen.cat_value("appointments", "status")
            mk(["appointments", "patients"], [(None, "appointments.appointment_id")],
               [Constraint("patients.city", "=", city), Constraint("appointments.fee", ">", fee)],
               f"List the appointment IDs for patients from {city} with a fee above {gen.fmt_num(fee, 'money')}.")
            mk(["appointments", "patients"], [("count", "*")],
               [Constraint("patients.age", ">", age), Constraint("appointments.status", "=", status)],
               f"How many appointments of patients older than {age} have status {status}?")
            spec = gen.cat_value("doctors", "specialty")
            mk(["appointments", "doctors"], [("avg", "appointments.fee")], [Constraint("doctors.specialty", "=", spec)],
               f"What is the average fee of appointments with doctors specialising in {spec}?")
        elif db == "retail":
            city = gen.cat_value("customers", "city"); amt = gen.num_value("orders", "amount", "money")
            mk(["customers", "orders"], [(None, "customers.name")],
               [Constraint("customers.city", "=", city), Constraint("orders.amount", ">", amt)],
               f"Find customers from {city} with purchases above {gen.fmt_num(amt, 'money')}.", distinct=True)
            tier = gen.cat_value("customers", "loyalty_tier"); y = rng.randint(2020, 2025)
            mk(["orders", "customers"], [("count", "*")],
               [Constraint("customers.loyalty_tier", "=", tier), Constraint("orders.order_date", ">=", f"{y}-01-01"),
                Constraint("orders.order_date", "<=", f"{y}-12-31")],
               f"How many orders were placed in {y} by customers in the {tier} tier?")
            pm = gen.cat_value("orders", "payment_method")
            mk(["orders", "customers"], [("sum", "orders.amount")],
               [Constraint("orders.payment_method", "=", pm), Constraint("customers.city", "=", city)],
               f"What is the total amount of orders paid via {pm} by customers from {city}?")
        elif db == "university":
            dept = gen.cat_value("students", "department"); mk_ = gen.num_value("enrollments", "marks", "int")
            mk(["students", "enrollments"], [(None, "students.name")],
               [Constraint("students.department", "=", dept), Constraint("enrollments.marks", ">", mk_)],
               f"Which students from the {dept} department scored more than {mk_} marks?", distinct=True)
            title = gen.cat_value("courses", "title")
            mk(["enrollments", "courses"], [("avg", "enrollments.marks")], [Constraint("courses.title", "=", title)],
               f"What is the average marks of enrollments in the course {title}?")
            sem = gen.cat_value("enrollments", "semester"); lo = gen.num_value("enrollments", "marks", "int")
            mk(["enrollments", "students"], [("count", "*")],
               [Constraint("enrollments.semester", "=", sem), Constraint("enrollments.marks", "<", lo),
                Constraint("students.department", "=", dept)],
               f"How many enrollments of {dept} students in semester {sem} have marks below {lo}?")
        elif db == "finance":
            br = gen.cat_value("accounts", "branch"); amt = gen.num_value("transactions", "amount", "money")
            mk(["transactions", "accounts"], [(None, "transactions.transaction_id")],
               [Constraint("transactions.amount", ">", amt), Constraint("accounts.branch", "=", br)],
               f"List transactions with an amount above {gen.fmt_num(amt, 'money')} from accounts in the {br} branch.")
            ch = gen.cat_value("transactions", "channel"); at = gen.cat_value("accounts", "account_type")
            mk(["transactions", "accounts"], [("count", "*")],
               [Constraint("transactions.transaction_type", "=", "Debit"), Constraint("transactions.channel", "=", ch),
                Constraint("accounts.account_type", "=", at)],
               f"How many Debit transactions were made via {ch} by {at} account holders?")
            y = rng.randint(2012, 2024)
            mk(["transactions", "accounts"], [("sum", "transactions.amount")],
               [Constraint("transactions.transaction_type", "=", "Credit"), Constraint("accounts.opened_date", ">", f"{y}-12-31")],
               f"What is the total amount of Credit transactions for accounts opened after {y}?")
        elif db == "travel":
            al = gen.cat_value("flights", "airline"); age = gen.num_value("bookings", "passenger_age", "int")
            mk(["bookings", "flights"], [(None, "bookings.passenger_name")],
               [Constraint("flights.airline", "=", al), Constraint("bookings.passenger_age", ">", age)],
               f"Show passengers booked on {al} flights who are older than {age}.")
            o, d = rng.sample(gen.schema.tables["flights"].columns["origin"].values, 2)
            cls = gen.cat_value("bookings", "seat_class")
            mk(["bookings", "flights"], [("count", "*")],
               [Constraint("bookings.seat_class", "=", cls), Constraint("flights.origin", "=", o),
                Constraint("flights.destination", "=", d)],
               f"How many {cls} class bookings were made on flights from {o} to {d}?")
            mk(["bookings", "flights"], [("avg", "bookings.amount")], [Constraint("flights.destination", "=", d)],
               f"What is the average amount paid by passengers on flights to {d}?")
    return out


DIFF = {"list": "simple", "count": "simple", "aggregate": "simple", "top-k": "moderate", "order": "moderate",
        "group-by": "moderate", "negation": "moderate", "in-list": "moderate", "range": "moderate",
        "multi-filter": "moderate", "join": "challenging"}


def main():
    items = []
    qid = 0
    for i, db in enumerate(ENT):
        gen = Gen(db, random.Random(7 + i))
        produced = []
        seen = set()
        tables = list(ENT[db])
        plan = (["list"] * 14 + ["multi"] * 10 + ["neg"] * 6 + ["in"] * 4 + ["between"] * 4 + ["count"] * 8 +
                ["agg"] * 8 + ["topk"] * 6 + ["sorted"] * 3 + ["group"] * 5)
        for kind in plan:
            for _attempt in range(20):
                t = gen.rng.choice(tables)
                if kind == "list":
                    r = gen.q_list(t, 1)
                elif kind == "multi":
                    r = gen.q_list(t, 2)
                    r = (r[0], r[1], "multi-filter") if r else r
                elif kind == "neg":
                    r = gen.q_list(t, 1, neg=True)
                    r = (r[0], r[1], "negation") if r else r
                elif kind == "in":
                    r = gen.q_list(t, 1, use_in=True)
                    r = (r[0], r[1], "in-list") if r else r
                elif kind == "between":
                    r = gen.q_list(t, 1, between=True)
                    r = (r[0], r[1], "range") if r else r
                elif kind == "count":
                    r = gen.q_count(t)
                elif kind == "agg":
                    r = gen.q_agg(t)
                elif kind == "topk":
                    r = gen.q_topk(t)
                elif kind == "sorted":
                    r = gen.q_sorted(t)
                else:
                    r = gen.q_group(t)
                if r and r[0] not in seen:
                    seen.add(r[0])
                    produced.append(r)
                    break
        jq = join_questions(gen)
        gen.rng.shuffle(jq)
        for r in jq[:8]:
            if r[0] not in seen:
                seen.add(r[0])
                produced.append(r)
        for q, g, qtype in produced:
            q = q[0].upper() + q[1:]
            sql = render_sql(g, gen.schema)
            gen.con.execute(sql).fetchall()          # gold must execute
            items.append({"question_id": qid, "db_id": db, "question": q, "evidence": "", "SQL": sql,
                          "difficulty": DIFF[qtype], "qtype": qtype})
            qid += 1
    out = DATA / "questions.json"
    out.write_text(json.dumps(items, indent=2, ensure_ascii=False), encoding="utf-8")
    from collections import Counter
    print(f"wrote {len(items)} questions -> {out}")
    print(Counter(x["qtype"] for x in items))
    print(Counter(x["db_id"] for x in items))


if __name__ == "__main__":
    main()
