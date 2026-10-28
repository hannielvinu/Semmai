"""Semmai local demo - runs on plain Python 3.9+ with NO pip installs.

    python app/server.py            ->  opens http://localhost:8000

Everything (web server, SQLite, the verifier, the multilingual intent model)
uses only the Python standard library.
"""
from __future__ import annotations

import argparse
import json
import mimetypes
import sqlite3
import sys
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from semmai.proof import _fmt_raw                          # noqa: E402
from semmai.schema import Schema                           # noqa: E402
from semmai.tools import ToolVerifier                      # noqa: E402
from semmai.usig import canonicalize, values_equal         # noqa: E402
from semmai.verifier import SemmaiVerifier                 # noqa: E402

DATA = ROOT / "data" / "bird_style"
WEB = ROOT / "app" / "web"
RES = ROOT / "results"
DBS = ["hospital", "retail", "university", "finance", "travel"]

SQL_EXAMPLES = [
    {"label": "Older than 60, but the SQL says ≥ 60", "db": "hospital", "question": "Show patients older than 60",
     "sql": "SELECT name FROM patients WHERE age >= 60"},
    {"label": "₹10,000 became ₹5,000", "db": "retail",
     "question": "Find customers from Chennai with purchases above ₹10,000",
     "sql": "SELECT DISTINCT c.name FROM customers c JOIN orders o ON c.customer_id = o.customer_id "
            "WHERE c.city = 'Chennai' AND o.amount > 5000"},
    {"label": "Same meaning, different SQL", "db": "hospital", "question": "Show patients older than 60",
     "sql": "SELECT p.name FROM patients AS p WHERE p.age >= 61"},
    {"label": "\"Except\" was lost", "db": "university",
     "question": "List all students except those from the Mechanical department",
     "sql": "SELECT name FROM students WHERE department = 'Mechanical'"},
    {"label": "A condition was dropped", "db": "finance",
     "question": "Savings accounts in the Chennai branch with balance under 50,000",
     "sql": "SELECT holder_name FROM accounts WHERE branch = 'Chennai' AND balance < 50000"},
    {"label": "Average became maximum", "db": "travel", "question": "Average fare of Air India flights",
     "sql": "SELECT MAX(price) FROM flights WHERE airline = 'Air India'"},
    {"label": "Wrong year", "db": "retail", "question": "How many orders were paid with UPI after March 2024?",
     "sql": "SELECT COUNT(*) FROM orders WHERE payment_method = 'UPI' AND order_date > '2023-03-31'"},
    {"label": "Cheapest sorted as most expensive", "db": "retail", "question": "What are the three cheapest books?",
     "sql": "SELECT name FROM products WHERE category = 'Books' ORDER BY price DESC LIMIT 3"},
    {"label": "Correct join query", "db": "retail",
     "question": "Find customers from Chennai with purchases above ₹10,000",
     "sql": "SELECT DISTINCT customers.name FROM customers JOIN orders ON customers.customer_id = orders.customer_id "
            "WHERE customers.city = 'Chennai' AND orders.amount > 10000"},
]
TOOL_EXAMPLES = [
    {"label": "Tamil: alarm time changed", "utt": "நாளை காலை 7 மணி க்கு அலாரம் வை",
     "action": {"intent": "alarm_set", "slots": {"date": "நாளை", "time": "காலை 8 மணி"}}},
    {"label": "Hindi: correct weather call", "utt": "कल चेन्नई में मौसम कैसा रहेगा",
     "action": {"intent": "weather_query", "slots": {"date": "कल", "place_name": "चेन्नई"}}},
    {"label": "English: lights on instead of off", "utt": "turn off the lights in the kitchen",
     "action": {"intent": "iot_hue_lighton", "slots": {"house_place": "kitchen"}}},
    {"label": "French: destination dropped", "utt": "réserve un billet de train pour Paris demain",
     "action": {"intent": "transport_ticket", "slots": {"transport_type": "train"}}},
    {"label": "Spanish: correct email", "utt": "envía un correo a María",
     "action": {"intent": "email_sendemail", "slots": {"person": "María"}}},
    {"label": "German: delete became set", "utt": "lösche meinen Wecker um 7 Uhr",
     "action": {"intent": "alarm_set", "slots": {"time": "7 Uhr"}}},
]

_verifiers: dict = {}
_tool = None
_lock = threading.Lock()


def verifier(db: str) -> SemmaiVerifier:
    with _lock:
        if db not in _verifiers:
            _verifiers[db] = SemmaiVerifier(Schema.from_bird_style(DATA, db))
        return _verifiers[db]


def tool_verifier() -> ToolVerifier:
    global _tool
    with _lock:
        if _tool is None:
            model = ROOT / "models" / "intent_model.json"
            if model.exists():
                _tool = ToolVerifier.load(model)
            else:                                   # needs scikit-learn
                from semmai.tools import load_massive
                recs = load_massive(ROOT / "data" / "massive_style" / "massive_style.jsonl")
                _tool = ToolVerifier(intent_threshold=0.1).fit([r for r in recs if r["partition"] == "train"])
        return _tool


# ---------------------------------------------------------------- the ledger
def ledger(v: SemmaiVerifier, r) -> list:
    """Row-by-row comparison of the request (intent) and the SQL (action)."""
    if r.intent is None or r.action is None:
        return []
    types = v.schema.col_types()
    I, A = canonicalize(r.intent, types), canonicalize(r.action, types)
    raw_i = dict(zip(map(id, I.constraints), r.intent.constraints))
    raw_a = dict(zip(map(id, A.constraints), r.action.constraints))
    rows = []

    def short(col):
        return col.split(".")[-1].replace("_", " ") if col and col != "*" else "rows"

    def agg(p):
        a, c = p
        return f"{a.upper()}({short(c)})" if a else short(c)

    rows.append({"aspect": "About", "asked": r.intent.primary_entity or "-",
                 "does": ", ".join(sorted(r.action.entities)) or "-",
                 "ok": (r.intent.primary_entity in r.action.entities)})
    i_aggs = [p for p in r.intent.projections if p[0]]
    a_aggs = [p for p in r.action.projections if p[0]]
    if i_aggs or a_aggs:
        ok = len(i_aggs) == len(a_aggs) and all(
            x[0] == y[0] and (x[0] == "count" or x[1] == y[1]) for x, y in zip(i_aggs, a_aggs))
        rows.append({"aspect": "Calculates", "asked": ", ".join(map(agg, i_aggs)) or "a list",
                     "does": ", ".join(map(agg, a_aggs)) or "a list", "ok": ok})
    used = set()
    for c in I.constraints:
        same = [a for a in A.constraints if a.column == c.column and id(a) not in used]
        exact = next((a for a in same if a.op == c.op and values_equal(a.value, c.value)), None)
        m = exact or (same[0] if same else None)
        if m is None:
            m = next((a for a in A.constraints if id(a) not in used and a.op == c.op
                      and values_equal(a.value, c.value)), None)
        if m is not None:
            used.add(id(m))
        rows.append({"aspect": "Condition", "asked": _fmt_raw(raw_i[id(c)], types),
                     "does": _fmt_raw(raw_a[id(m)], types) if m is not None else "missing",
                     "ok": exact is not None})
    for a in A.constraints:
        if id(a) not in used:
            rows.append({"aspect": "Condition", "asked": "not asked", "does": _fmt_raw(raw_a[id(a)], types),
                         "ok": False})
    if A.has_or and not I.has_or:
        rows.append({"aspect": "Logic", "asked": "all conditions (AND)", "does": "any condition (OR)", "ok": False})
    if I.group_by or A.group_by:
        rows.append({"aspect": "Grouped by", "asked": ", ".join(map(short, I.group_by)) or "none",
                     "does": ", ".join(map(short, A.group_by)) or "none", "ok": sorted(I.group_by) == sorted(A.group_by)})
    if I.order_by or A.order_by:
        def od(g):
            if not g.order_by:
                return "no order"
            _, c, d = g.order_by[0]
            return f"{'highest' if d == 'DESC' else 'lowest'} {short(c)} first"
        rows.append({"aspect": "Sorted", "asked": od(I), "does": od(A),
                     "ok": [(c, d) for _, c, d in I.order_by] == [(c, d) for _, c, d in A.order_by]})
    if I.limit or A.limit:
        rows.append({"aspect": "How many", "asked": str(I.limit or "all"), "does": str(A.limit or "all"),
                     "ok": (I.limit or None) == (A.limit or None)})
    return rows


def run_readonly(db: str, sql: str, limit: int = 50):
    uri = "file:" + str(DATA / "databases" / db / f"{db}.sqlite").replace("\\", "/") + "?mode=ro"
    con = sqlite3.connect(uri, uri=True)
    try:
        cur = con.execute(sql)
        cols = [d[0] for d in cur.description or []]
        rows = cur.fetchmany(limit + 1)
        return {"columns": cols, "rows": [list(r) for r in rows[:limit]], "truncated": len(rows) > limit}
    finally:
        con.close()


def api_verify(body: dict) -> dict:
    db = body.get("db", "hospital")
    if db not in DBS:
        raise ValueError(f"unknown database '{db}'")
    q, sql = (body.get("question") or "").strip(), (body.get("sql") or "").strip()
    if not q or not sql:
        raise ValueError("Enter both a request and an SQL query.")
    v = verifier(db)
    r = v.verify(q, sql)
    out = r.to_dict()
    out["ledger"] = ledger(v, r)
    out["result"] = None
    if r.allowed:
        try:
            out["result"] = run_readonly(db, sql)
        except sqlite3.Error as e:
            out["result"] = {"error": str(e)}
    return out


def api_tool(body: dict) -> dict:
    utt = (body.get("utt") or "").strip()
    action = body.get("action")
    if isinstance(action, str):
        action = json.loads(action)
    if not utt or not isinstance(action, dict):
        raise ValueError("Enter an utterance and a tool call as JSON.")
    r = tool_verifier().verify(utt, action)
    top = sorted(r.intent_probs.items(), key=lambda kv: -kv[1])[:4]
    return {"decision": r.decision, "findings": [f.to_dict() for f in r.findings],
            "top_intents": [{"intent": k, "p": round(p, 3)} for k, p in top]}


def api_meta() -> dict:
    schemas = {db: verifier(db).schema.describe() for db in DBS}
    results = {}
    for name in ("sql_results.json", "tools_results.json", "pipeline_results.json"):
        p = RES / name
        if p.exists():
            results[name.split("_")[0]] = json.loads(p.read_text(encoding="utf-8"))
    order = ["sql_methods.png", "sql_error_types.png", "similarity_overlap.png", "pipeline.png", "multilingual_f1.png"]
    have = {f.name for f in (RES / "figures").glob("*.png")} if (RES / "figures").exists() else set()
    figs = [f for f in order if f in have] + sorted(have - set(order))
    return {"dbs": DBS, "schemas": schemas, "sql_examples": SQL_EXAMPLES, "tool_examples": TOOL_EXAMPLES,
            "results": results, "figures": figs}


class Handler(BaseHTTPRequestHandler):
    server_version = "Semmai/1.0"

    def log_message(self, fmt, *args):          # quiet console
        if "/api/" in (args[0] if args else ""):
            sys.stdout.write("  " + (fmt % args) + "\n")

    def _send(self, code, body: bytes, ctype: str):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, code, obj):
        self._send(code, json.dumps(obj, ensure_ascii=False, default=str).encode("utf-8"),
                   "application/json; charset=utf-8")

    def do_GET(self):
        path = self.path.split("?")[0]
        if path == "/api/meta":
            return self._json(200, api_meta())
        if path in ("/", "/index.html"):
            f = WEB / "index.html"
        elif path.startswith("/figures/"):
            f = RES / "figures" / Path(path).name
        else:
            f = WEB / Path(path).name
        if not f.is_file():
            return self._send(404, b"not found", "text/plain")
        ctype = mimetypes.guess_type(f.name)[0] or "application/octet-stream"
        if ctype.startswith("text/"):
            ctype += "; charset=utf-8"
        self._send(200, f.read_bytes(), ctype)

    def do_POST(self):
        try:
            n = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(n).decode("utf-8") or "{}")
            if self.path == "/api/verify":
                return self._json(200, api_verify(body))
            if self.path == "/api/tool":
                return self._json(200, api_tool(body))
            return self._json(404, {"error": "unknown endpoint"})
        except json.JSONDecodeError as e:
            return self._json(400, {"error": f"The tool call is not valid JSON: {e.msg} (line {e.lineno})."})
        except ValueError as e:
            return self._json(400, {"error": str(e)})
        except Exception as e:  # noqa: BLE001
            return self._json(500, {"error": f"{type(e).__name__}: {e}"})


def main():
    ap = argparse.ArgumentParser(description="Semmai local demo")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--no-browser", action="store_true")
    a = ap.parse_args()
    print("Loading Semmai ...")
    for db in DBS:
        verifier(db)
    tool_verifier()
    srv = None
    for port in range(a.port, a.port + 20):
        try:
            srv = ThreadingHTTPServer(("127.0.0.1", port), Handler)
            break
        except OSError:
            continue
    if srv is None:
        sys.exit(f"No free port between {a.port} and {a.port + 19}.")
    url = f"http://localhost:{srv.server_address[1]}"
    print(f"\n  Semmai is running at {url}\n  Press Ctrl+C to stop.\n")
    if not a.no_browser:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
