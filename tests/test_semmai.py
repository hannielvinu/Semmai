"""Unit tests.   Run:  python -m unittest discover -s tests -v"""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from semmai.intent import IntentExtractor, stem                  # noqa: E402
from semmai.schema import Schema                                  # noqa: E402
from semmai.sql_parser import UnsupportedSQL, sql_to_usig         # noqa: E402
from semmai.tools import ToolVerifier, load_massive, parse_annot  # noqa: E402
from semmai.usig import Constraint, canonicalize_constraint       # noqa: E402
from semmai.verifier import SemmaiVerifier                        # noqa: E402

DATA = ROOT / "data" / "bird_style"


def V(db):
    return SemmaiVerifier(Schema.from_bird_style(DATA, db), use_similarity=False)


class TestCanonicalForm(unittest.TestCase):
    def test_integer_boundaries(self):
        a = canonicalize_constraint(Constraint("t.age", ">=", 61), "int")
        b = canonicalize_constraint(Constraint("t.age", ">", 60), "int")
        self.assertEqual((a.op, a.value), (b.op, b.value))

    def test_date_boundaries(self):
        a = canonicalize_constraint(Constraint("t.d", "<=", "2020-12-31"), "date")
        b = canonicalize_constraint(Constraint("t.d", "<", "2021-01-01"), "date")
        self.assertEqual((a.op, a.value), (b.op, b.value))

    def test_stem(self):
        self.assertEqual(stem("departing"), stem("departs"))
        self.assertEqual(stem("admitted"), stem("admit"))


class TestSqlCompiler(unittest.TestCase):
    def setUp(self):
        self.s = Schema.from_bird_style(DATA, "retail")

    def test_join_and_aliases(self):
        g = sql_to_usig("SELECT T1.name FROM customers AS T1 JOIN orders AS T2 ON T1.customer_id = T2.customer_id "
                        "WHERE T1.city = 'Chennai' AND T2.amount > 10000", self.s)
        self.assertEqual(g.entities, {"customers", "orders"})
        self.assertEqual(len(g.joins), 1)
        self.assertEqual({c.column for c in g.constraints}, {"customers.city", "orders.amount"})

    def test_year_function_becomes_range(self):
        g = sql_to_usig("SELECT COUNT(*) FROM orders WHERE strftime('%Y', order_date) = '2023'", self.s)
        self.assertEqual(sorted((c.op, c.value) for c in g.constraints), [("<=", "2023-12-31"), (">=", "2023-01-01")])

    def test_or_same_column_is_in(self):
        g = sql_to_usig("SELECT name FROM customers WHERE city = 'Pune' OR city = 'Salem'", self.s)
        self.assertEqual(g.constraints[0].op, "IN")
        self.assertFalse(g.has_or)

    def test_subquery_is_unsupported(self):
        with self.assertRaises(UnsupportedSQL):
            sql_to_usig("SELECT name FROM customers WHERE age > (SELECT AVG(age) FROM customers)", self.s)


class TestIntentExtraction(unittest.TestCase):
    def test_slide_example(self):
        ie = IntentExtractor(Schema.from_bird_style(DATA, "hospital"))
        g = ie.extract("Show patients older than 60")
        self.assertEqual([(c.column, c.op, c.value) for c in g.constraints], [("patients.age", ">", 60)])

    def test_money_lakh_and_negation(self):
        ie = IntentExtractor(Schema.from_bird_style(DATA, "hospital"))
        g = ie.extract("Patients not from Chennai whose bill went beyond 1 lakh rupees")
        got = {(c.column, c.op, c.value) for c in g.constraints}
        self.assertIn(("patients.city", "!=", "Chennai"), got)
        self.assertIn(("patients.bill_amount", ">", 100000.0), got)

    def test_topk(self):
        ie = IntentExtractor(Schema.from_bird_style(DATA, "retail"))
        g = ie.extract("Show me the 5 most expensive products")
        self.assertEqual(g.limit, 5)
        self.assertEqual(g.order_by[0][1:], ("products.price", "DESC"))


class TestVerifier(unittest.TestCase):
    def test_operator_drift_blocked(self):
        v = V("hospital")
        self.assertFalse(v.verify("Show patients older than 60", "SELECT name FROM patients WHERE age >= 60").allowed)
        self.assertTrue(v.verify("Show patients older than 60", "SELECT name FROM patients WHERE age > 60").allowed)
        self.assertTrue(v.verify("Show patients older than 60", "SELECT * FROM patients WHERE 60 < age").allowed)

    def test_value_drift_blocked(self):
        v = V("retail")
        q = "Find customers from Chennai with purchases above ₹10,000"
        bad = ("SELECT c.name FROM customers c JOIN orders o ON c.customer_id = o.customer_id "
               "WHERE c.city = 'Chennai' AND o.amount > 5000")
        r = v.verify(q, bad)
        self.assertFalse(r.allowed)
        self.assertIn("VALUE_DRIFT", {f.kind for f in r.findings})

    def test_missing_constraint_and_wrong_aggregate(self):
        v = V("university")
        self.assertFalse(v.verify("How many CSE students have cgpa above 9?",
                                  "SELECT COUNT(*) FROM students WHERE cgpa > 9").allowed)
        self.assertFalse(v.verify("Average CGPA of CSE students",
                                  "SELECT MAX(cgpa) FROM students WHERE department = 'CSE'").allowed)

    def test_unparseable_is_rejected(self):
        v = V("retail")
        self.assertFalse(v.verify("customers older than 30", "DROP TABLE customers").allowed)


class TestTools(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        recs = load_massive(ROOT / "data" / "massive_style" / "massive_style.jsonl")
        cls.v = ToolVerifier(intent_threshold=0.1).fit([r for r in recs if r["partition"] == "train"])

    def test_parse_annot(self):
        self.assertEqual(parse_annot("wake me at [time : 7 am] [date : tomorrow]"), {"time": "7 am", "date": "tomorrow"})

    def test_tamil_alarm(self):
        u = "நாளை காலை 7 மணி க்கு அலாரம் வை"
        ok = {"intent": "alarm_set", "slots": {"date": "நாளை", "time": "காலை 7 மணி"}}
        drift = {"intent": "alarm_set", "slots": {"date": "நாளை", "time": "காலை 8 மணி"}}
        self.assertTrue(self.v.verify(u, ok).allowed)
        self.assertFalse(self.v.verify(u, drift).allowed)

    def test_dropped_number(self):
        r = self.v.verify("set an alarm for 8 am", {"intent": "alarm_set", "slots": {}})
        self.assertFalse(r.allowed)


class TestPlainPythonDemo(unittest.TestCase):
    """The demo must run with only the standard library (exported intent model)."""

    def test_exported_model_matches(self):
        model = ROOT / "models" / "intent_model.json"
        self.assertTrue(model.exists(), "run scripts/evaluate_tools.py to export the model")
        v = ToolVerifier.load(model)
        self.assertEqual(v.predict_intent("turn off the lights in the kitchen"), "iot_hue_lightoff")
        self.assertFalse(v.verify("नाळ", {"intent": "nope", "slots": {}}).allowed)

    def test_server_api(self):
        sys.path.insert(0, str(ROOT / "app"))
        import server
        r = server.api_verify({"db": "hospital", "question": "Show patients older than 60",
                               "sql": "SELECT name FROM patients WHERE age > 60"})
        self.assertEqual(r["decision"], "ALLOW")
        self.assertTrue(r["result"]["rows"])
        r = server.api_verify({"db": "hospital", "question": "Show patients older than 60",
                               "sql": "SELECT name FROM patients WHERE age >= 60"})
        self.assertEqual(r["decision"], "REJECT")
        self.assertIsNone(r["result"])                      # blocked queries are never executed
        self.assertIn(False, [row["ok"] for row in r["ledger"]])


class TestTamil(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from semmai.tamil import TamilParser
        from semmai.chat import ChatPipeline
        cls.s = Schema.from_bird_style(DATA, "hr")
        cls.tp = TamilParser(cls.s)
        cls.pipe = ChatPipeline(cls.s)

    def test_after_vs_from(self):
        from semmai.tamil import render_english
        self.assertIn("after 2022", render_english(self.tp.parse("2022க்கு பிறகு சேர்ந்த ஊழியர்களை காட்டு")))
        self.assertIn("since 2022", render_english(self.tp.parse("2022 முதல் சேர்ந்த ஊழியர்களைக் காட்டு")))

    def test_except_and_suffixes(self):
        f = self.tp.parse("சென்னையைத் தவிர மற்ற நகரங்களில் உள்ள மேலாளர்கள்")
        got = {(c.op, c.values) for c in f.clauses}
        self.assertIn(("!=", ("Chennai",)), got)
        self.assertIn(("=", ("Manager",)), got)

    def test_lakh_and_top_k(self):
        f = self.tp.parse("1 லட்சத்திற்கு மேல் சம்பளம் பெறும் ஊழியர்கள் எத்தனை பேர்?")
        self.assertEqual((f.op, f.clauses[0].op, f.clauses[0].values), ("count", ">", (100000.0,)))
        f = self.tp.parse("அதிக சம்பளம் வாங்கும் 5 ஊழியர்களைக் காட்டு")
        self.assertEqual((f.order_col, f.order_dir, f.limit), ("salary", "DESC", 5))

    def test_pipeline_blocks_translation_mistake(self):
        r = self.pipe.run("2022க்கு பிறகு சேர்ந்த ஊழியர்களை காட்டு", mistake_mode="always", seed=1,
                          preferred_mistake="AFTER_BECOMES_FROM")
        att = next(s for s in r["steps"] if s["key"] == "attempts")["data"]
        self.assertFalse(att[0]["verify"]["ok"])           # 'after' -> 'from' was caught
        self.assertIn("> '2022-12-31'", r["final_sql"])     # the executed query is the right one
        self.assertTrue(all(c["ok"] for c in next(s for s in r["steps"] if s["key"] == "final")["data"]["checks"]))

    def test_delete_is_dry_run_and_except_kept(self):
        r = self.pipe.run("மேலாளர்களைத் தவிர மற்ற ஊழியர்களை நீக்கு", mistake_mode="always", seed=2,
                          preferred_mistake="EXCEPT_DROPPED")
        self.assertIn("role != 'Manager'", r["final_sql"])
        self.assertTrue(r["result"]["dry_run"])
        import sqlite3
        n = sqlite3.connect(self.s.db_path).execute("SELECT COUNT(*) FROM employees").fetchone()[0]
        self.assertEqual(n, 600)                              # nothing was deleted


if __name__ == "__main__":
    unittest.main()
