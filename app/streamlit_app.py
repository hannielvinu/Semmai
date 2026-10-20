"""Semmai interactive demo.

    pip install streamlit
    streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from semmai.schema import Schema                     # noqa: E402
from semmai.tools import ToolVerifier, load_massive  # noqa: E402
from semmai.verifier import SemmaiVerifier           # noqa: E402

DATA = ROOT / "data" / "bird_style"
RES = ROOT / "results"
DBS = ["hospital", "retail", "university", "finance", "travel"]

EXAMPLES = {
    "Slide example: older than 60 (≥ vs >)": ("hospital", "Show patients older than 60",
                                               "SELECT name FROM patients WHERE age >= 60"),
    "Slide example: ₹10,000 vs ₹5,000": ("retail", "Find customers from Chennai with purchases above ₹10,000",
                                         "SELECT c.name FROM customers c JOIN orders o ON c.customer_id = o.customer_id "
                                         "WHERE c.city = 'Chennai' AND o.amount > 5000"),
    "Correct query (equivalent rewrite)": ("hospital", "Show patients older than 60",
                                           "SELECT p.name FROM patients AS p WHERE p.age >= 61"),
    "Negation lost": ("university", "List all students except those from the Mechanical department",
                      "SELECT name FROM students WHERE department = 'Mechanical'"),
    "Missing constraint": ("finance", "Savings accounts in the Chennai branch with balance under 50,000",
                           "SELECT holder_name FROM accounts WHERE branch = 'Chennai' AND balance < 50000"),
    "Wrong aggregation": ("travel", "Average fare of Air India flights",
                          "SELECT MAX(price) FROM flights WHERE airline = 'Air India'"),
    "Date drift": ("retail", "How many orders were paid with UPI after March 2024?",
                   "SELECT COUNT(*) FROM orders WHERE payment_method = 'UPI' AND order_date > '2023-03-31'"),
    "Wrong ranking direction": ("retail", "What are the three cheapest books?",
                                "SELECT name FROM products WHERE category = 'Books' ORDER BY price DESC LIMIT 3"),
}
TOOL_EXAMPLES = {
    "Tamil - alarm, time drift": ("நாளை காலை 7 மணி க்கு அலாரம் வை",
                                  {"intent": "alarm_set", "slots": {"date": "நாளை", "time": "காலை 8 மணி"}}),
    "Hindi - weather, correct": ("कल चेन्नई में मौसम कैसा रहेगा",
                                 {"intent": "weather_query", "slots": {"date": "कल", "place_name": "चेन्नई"}}),
    "English - lights, wrong action": ("turn off the lights in the kitchen",
                                       {"intent": "iot_hue_lighton", "slots": {"house_place": "kitchen"}}),
    "French - ticket, dropped slot": ("réserve un billet de train pour Paris demain",
                                      {"intent": "transport_ticket", "slots": {"transport_type": "train"}}),
    "Spanish - email, correct": ("envía un correo a María", {"intent": "email_sendemail", "slots": {"person": "María"}}),
    "German - alarm, wrong intent": ("lösche meinen Wecker um 7 Uhr",
                                     {"intent": "alarm_set", "slots": {"time": "7 Uhr"}}),
}

st.set_page_config(page_title="Semmai", layout="wide")


@st.cache_resource
def get_verifier(db):
    return SemmaiVerifier(Schema.from_bird_style(DATA, db))


@st.cache_resource
def get_tool_verifier():
    recs = load_massive(ROOT / "data" / "massive_style" / "massive_style.jsonl")
    thr = 0.1
    p = RES / "tools_results.json"
    if p.exists():
        thr = json.loads(p.read_text())["thresholds"]["intent"]
    return ToolVerifier(intent_threshold=thr).fit([r for r in recs if r["partition"] == "train"])


def verdict_banner(decision, n_crit):
    if decision == "ALLOW":
        st.success("✅ MATCH → execution allowed")
    else:
        st.error(f"⛔ SEMANTIC MISMATCH → BLOCK ({n_crit} critical finding{'s' if n_crit != 1 else ''})")


st.title("Semmai: Semantic Verification for LLM Agents")
st.caption("Before an AI agent acts, Semmai asks: does this action actually mean what the user asked for?")

tab_sql, tab_tool, tab_res = st.tabs(["SQL verification", "Multilingual tool actions", "Results"])

with tab_sql:
    ex = st.selectbox("Load an example", ["(write your own)"] + list(EXAMPLES))
    d_db, d_q, d_sql = EXAMPLES.get(ex, ("hospital", "Show patients older than 60",
                                         "SELECT name FROM patients WHERE age > 60"))
    c1, c2 = st.columns([1, 3])
    db = c1.selectbox("Database", DBS, index=DBS.index(d_db))
    with c1.expander("Schema"):
        st.code(get_verifier(db).schema.describe(), language="sql")
    q = c2.text_input("User request", d_q)
    sql = c2.text_area("LLM-generated SQL", d_sql, height=90)
    if st.button("Verify", type="primary"):
        r = get_verifier(db).verify(q, sql)
        verdict_banner(r.decision, len(r.critical()))
        st.markdown(f"**What the SQL actually does:** {r.action_text or '—'}")
        if r.findings:
            st.dataframe(pd.DataFrame([f.to_dict() for f in r.findings]), use_container_width=True, hide_index=True)
        g1, g2 = st.columns(2)
        g1.markdown("**Intent graph (from the request)**")
        g1.json(r.intent.to_dict() if r.intent else {})
        g2.markdown("**Action graph (from the SQL)**")
        g2.json(r.action.to_dict() if r.action else {})
        sim_txt = f"{r.similarity:.2f}" if r.similarity is not None else "n/a"
        st.caption(f"Similarity (question vs SQL meaning): {sim_txt} · verified in {r.latency_ms:.1f} ms")
        if r.allowed:
            con = sqlite3.connect(get_verifier(db).schema.db_path)
            try:
                df = pd.read_sql_query(sql, con)
                st.markdown(f"**Executed — {len(df)} row(s)**")
                st.dataframe(df.head(50), use_container_width=True)
            finally:
                con.close()

with tab_tool:
    tex = st.selectbox("Load an example", list(TOOL_EXAMPLES), key="tool_ex")
    utt0, act0 = TOOL_EXAMPLES[tex]
    utt = st.text_input("User utterance (any of: English, Tamil, Hindi, French, Spanish, German)", utt0)
    act = st.text_area("Agent's tool call (JSON)", json.dumps(act0, ensure_ascii=False, indent=1), height=110)
    if st.button("Verify tool call", type="primary"):
        try:
            action = json.loads(act)
        except json.JSONDecodeError as e:
            st.error(f"Invalid JSON: {e}")
        else:
            tv = get_tool_verifier()
            r = tv.verify(utt, action)
            verdict_banner(r.decision, len(r.findings))
            for f in r.findings:
                st.markdown(f"- **{f.kind}** — {f.message}")
            top = sorted(r.intent_probs.items(), key=lambda kv: -kv[1])[:5]
            st.markdown("**Intent model (top 5)**")
            st.dataframe(pd.DataFrame(top, columns=["intent", "probability"]), hide_index=True)

with tab_res:
    for name in ("sql_results.md", "tools_results.md", "pipeline_results.md"):
        p = RES / name
        if p.exists():
            st.markdown(p.read_text())
    for fig in sorted((RES / "figures").glob("*.png")) if (RES / "figures").exists() else []:
        st.image(str(fig))
