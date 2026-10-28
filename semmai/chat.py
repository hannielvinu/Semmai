"""Semmai chat pipeline (Tamil-first).

    Tamil question
      │
      ├─► [AI path]       translator (Tamil → English)  ─►  LLM (English → SQL)
      │
      ├─► [Semmai path]   Tamil semantic parser  ─►  user intent graph
      │
      ▼
    Semmai proof:  user intent graph  ≟  SQL intent graph
      │ match                         │ mismatch
      ▼                               ▼
    execute (read-only / dry-run)    block → regenerate with feedback (≤ N attempts)
      │                                → still wrong: compile SQL from the verified intent graph
      ▼
    final check: answer from the AI's SQL == answer from the user's own meaning graph
      ▼
    answer in Tamil

Backends are pluggable (environment variables, all optional):
    SEMMAI_TRANSLATOR = builtin | indictrans2          (default builtin)
    SEMMAI_IT2_MODEL  = ai4bharat/indictrans2-indic-en-dist-200M  (or ...-indic-en-1B)
    SEMMAI_LLM        = builtin | ollama | openai      (default builtin)
    SEMMAI_LLM_MODEL  = qwen2.5-coder:7b  /  gpt-4o-mini / llama-3.1-8b-instant ...
    SEMMAI_LLM_BASE_URL, SEMMAI_LLM_API_KEY            (OpenAI-compatible APIs: OpenAI, Groq, Gemini ...)
"""
from __future__ import annotations

import json
import os
import random
import re
import sqlite3
import time
import urllib.request
from pathlib import Path
from typing import Optional

from .explain import TA_AGG, TA_ENTITY, TA_FINDING, graph_view, ledger, ta_col
from .intent import IntentExtractor
from .proof import prove
from .schema import Schema
from .similarity import Similarity
from .sql_parser import UnsupportedSQL, render_sql, sql_to_usig
from .tamil import Frame, TamilParser, render_english
from .verbalize import verbalize

TAMIL_RE = re.compile(r"[஀-௿]")

# ------------------------------------------------------------ mistakes
MISTAKES = {
    "AFTER_BECOMES_FROM": "மொழிபெயர்ப்பாளர் 'பிறகு' (after) என்பதை 'முதல்' (from) என்று புரிந்துகொண்டது",
    "BEFORE_BECOMES_UNTIL": "மொழிபெயர்ப்பாளர் 'முன்' (before) என்பதை 'வரை' (until) என்று புரிந்துகொண்டது",
    "EXCEPT_DROPPED": "மொழிபெயர்ப்பில் 'தவிர' (except) என்ற சொல் விடுபட்டது",
    "NEGATION_LOST": "'அல்லாத / இல்லாத' என்ற மறுப்பு இழக்கப்பட்டது",
    "MORE_THAN_BECOMES_AT_LEAST": "'மேல்' (more than) என்பது 'குறைந்தது' (at least) ஆனது",
    "NUMBER_CHANGED": "மொழிபெயர்ப்பில் எண் மாறியது",
    "FILTER_DROPPED": "ஒரு நிபந்தனை மொழிபெயர்ப்பில் விடுபட்டது",
    "HIGHEST_BECOMES_LOWEST": "'அதிக' (highest) என்பது 'குறைந்த' (lowest) ஆனது",
    "TOP_K_CHANGED": "'முதல் N' எண்ணிக்கை மாறியது",
    "AVERAGE_BECOMES_MAXIMUM": "'சராசரி' (average) என்பது 'அதிகபட்சம்' (maximum) ஆனது",
    "YEAR_SHIFTED": "ஆண்டு தவறாக மொழிபெயர்க்கப்பட்டது",
}


def apply_mistake(f: Frame, rng: random.Random, preferred: Optional[str] = None):
    """Perturb a meaning frame the way a translator / LLM typically does. Returns (frame, kind) or (frame, None)."""
    g = f.copy()
    options = []
    for i, c in enumerate(g.clauses):
        if c.kind == "date" and c.op == ">":
            options.append(("AFTER_BECOMES_FROM", i))
        if c.kind == "date" and c.op == "<":
            options.append(("BEFORE_BECOMES_UNTIL", i))
        if c.kind == "date" and c.op in ("in", ">", "<", ">=", "<=") and not isinstance(c.values[0], tuple):
            options.append(("YEAR_SHIFTED", i))
        if c.kind == "val" and c.op in ("!=", "NOT IN"):
            options.append(("EXCEPT_DROPPED", i))
            options.append(("NEGATION_LOST", i))
        if c.kind == "num" and c.op == ">":
            options.append(("MORE_THAN_BECOMES_AT_LEAST", i))
        if c.kind == "num" and c.op in (">", "<", ">=", "<=", "="):
            options.append(("NUMBER_CHANGED", i))
        if c.kind != "val" or c.op not in ("!=", "NOT IN"):
            options.append(("FILTER_DROPPED", i))
    if g.order_col and not g.sorted_only:
        options.append(("HIGHEST_BECOMES_LOWEST", None))
        if (g.limit or 1) > 1:
            options.append(("TOP_K_CHANGED", None))
    if g.agg == "avg":
        options.append(("AVERAGE_BECOMES_MAXIMUM", None))
    if not options:
        return g, None
    story = [o for o in options if o[0] in ("AFTER_BECOMES_FROM", "EXCEPT_DROPPED", "MORE_THAN_BECOMES_AT_LEAST")]
    if preferred:
        pick = [o for o in options if o[0] == preferred] or options
    else:
        pick = story if story and rng.random() < 0.7 else options
    kind, i = rng.choice(pick)
    if kind in ("AFTER_BECOMES_FROM", "BEFORE_BECOMES_UNTIL"):
        g.clauses[i].op = ">=" if kind == "AFTER_BECOMES_FROM" else "<="
    elif kind == "YEAR_SHIFTED":
        y, m = g.clauses[i].values
        g.clauses[i].values = (y - 1, m)
    elif kind == "EXCEPT_DROPPED":
        del g.clauses[i]
    elif kind == "NEGATION_LOST":
        g.clauses[i].op = "=" if g.clauses[i].op == "!=" else "IN"
    elif kind == "MORE_THAN_BECOMES_AT_LEAST":
        g.clauses[i].op = ">="
    elif kind == "NUMBER_CHANGED":
        v = g.clauses[i].values[0]
        nv = v / 2 if v >= 10 else v + 2
        g.clauses[i].values = (int(nv) if float(nv).is_integer() else nv,)
    elif kind == "FILTER_DROPPED":
        del g.clauses[i]
    elif kind == "HIGHEST_BECOMES_LOWEST":
        g.order_dir = "ASC" if g.order_dir == "DESC" else "DESC"
    elif kind == "TOP_K_CHANGED":
        g.limit = max(1, (g.limit or 1) // 2)
    elif kind == "AVERAGE_BECOMES_MAXIMUM":
        g.agg = "max"
    return g, kind


# ------------------------------------------------------------ backends
class BuiltinTranslator:
    name = "Built-in Tamil→English (rule-based)"

    def translate(self, frame: Frame) -> str:
        return render_english(frame)


class IndicTrans2Translator:
    """AI4Bharat IndicTrans2 (https://huggingface.co/ai4bharat/indictrans2-indic-en-1B).
    pip install torch "transformers<5" IndicTransToolkit ; accept the model licence on Hugging Face."""

    def __init__(self, model_name: Optional[str] = None):
        import torch  # noqa: F401
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        from IndicTransToolkit.processor import IndicProcessor
        self.model_name = model_name or os.environ.get("SEMMAI_IT2_MODEL", "ai4bharat/indictrans2-indic-en-dist-200M")
        self.name = f"IndicTrans2 ({self.model_name.split('/')[-1]})"
        self.torch = __import__("torch")
        self.device = "cuda" if self.torch.cuda.is_available() else "cpu"
        self.tok = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name, trust_remote_code=True).to(self.device)
        self.ip = IndicProcessor(inference=True)

    def translate_text(self, text: str) -> str:
        batch = self.ip.preprocess_batch([text], src_lang="tam_Taml", tgt_lang="eng_Latn")
        inputs = self.tok(batch, truncation=True, padding="longest", return_tensors="pt",
                          return_attention_mask=True).to(self.device)
        with self.torch.no_grad():
            out = self.model.generate(**inputs, use_cache=True, min_length=0, max_length=256, num_beams=5,
                                      num_return_sequences=1)
        dec = self.tok.batch_decode(out, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        return self.ip.postprocess_batch(dec, lang="eng_Latn")[0]


class BuiltinSQLGenerator:
    name = "Built-in NL→SQL (rule-based)"

    def generate(self, english: str, schema: Schema, extractor: IntentExtractor, feedback: str = "") -> str:
        return render_sql(extractor.extract(english), schema)


SQL_PROMPT = """You are an expert SQLite developer. Write ONE SQLite query for the question.
Use only these tables and columns:
{schema}
{feedback}
Question: {question}
Return only the SQL, nothing else."""


class LLMSQLGenerator:
    """Ollama (local, e.g. qwen2.5-coder:7b) or any OpenAI-compatible chat API."""

    def __init__(self, kind: str):
        self.kind = kind
        self.model = os.environ.get("SEMMAI_LLM_MODEL", "qwen2.5-coder:7b" if kind == "ollama" else "gpt-4o-mini")
        self.base = os.environ.get("SEMMAI_LLM_BASE_URL",
                                   "http://localhost:11434" if kind == "ollama" else "https://api.openai.com/v1")
        self.key = os.environ.get("SEMMAI_LLM_API_KEY", "")
        self.name = f"{'Ollama' if kind == 'ollama' else 'API'}: {self.model}"

    def generate(self, english: str, schema: Schema, extractor=None, feedback: str = "") -> str:
        from .generator import extract_sql
        fb = f"\nA previous attempt was rejected because: {feedback}\nFix exactly that.\n" if feedback else ""
        prompt = SQL_PROMPT.format(schema=schema.describe(), question=english, feedback=fb)
        if self.kind == "ollama":
            url, body = f"{self.base.rstrip('/')}/api/generate", {"model": self.model, "prompt": prompt,
                                                                  "stream": False, "options": {"temperature": 0}}
            headers = {"Content-Type": "application/json"}
        else:
            url = f"{self.base.rstrip('/')}/chat/completions"
            body = {"model": self.model, "temperature": 0, "messages": [{"role": "user", "content": prompt}]}
            headers = {"Content-Type": "application/json", "Authorization": f"Bearer {self.key}"}
        req = urllib.request.Request(url, data=json.dumps(body).encode(), headers=headers)
        with urllib.request.urlopen(req, timeout=90) as r:
            data = json.loads(r.read())
        text = data["response"] if self.kind == "ollama" else data["choices"][0]["message"]["content"]
        return extract_sql(text)


def make_translator():
    if os.environ.get("SEMMAI_TRANSLATOR", "builtin").lower() == "indictrans2":
        try:
            return IndicTrans2Translator()
        except Exception as e:  # noqa: BLE001
            print(f"[semmai] IndicTrans2 not available ({e}); using the built-in translator")
    return BuiltinTranslator()


def make_generator():
    kind = os.environ.get("SEMMAI_LLM", "builtin").lower()
    if kind in ("ollama", "openai"):
        return LLMSQLGenerator(kind)
    return BuiltinSQLGenerator()


# ------------------------------------------------------------ the pipeline
class ChatPipeline:
    def __init__(self, schema: Schema, translator=None, generator=None, similarity: Optional[Similarity] = None):
        self.schema = schema
        self.types = schema.col_types()
        self.tamil = TamilParser(schema)
        self.extractor = IntentExtractor(schema)
        self.translator = translator or BuiltinTranslator()
        self.generator = generator or BuiltinSQLGenerator()
        self.sim = similarity or Similarity()

    # ---- helpers
    def _run_sql(self, sql: str, limit: int = 100):
        uri = "file:" + str(Path(self.schema.db_path).resolve()).replace("\\", "/") + "?mode=ro"
        con = sqlite3.connect(uri, uri=True)
        try:
            if sql.strip().lower().startswith("delete"):
                where = re.sub(r"(?is)^\s*delete\s+from\s+", "", sql.strip().rstrip(";"))
                count_sql = "SELECT COUNT(*) FROM " + where
                sample_sql = "SELECT * FROM " + where + " LIMIT 10"
                n = con.execute(count_sql).fetchone()[0]
                cur = con.execute(sample_sql)
                cols = [d[0] for d in cur.description]
                return {"dry_run": True, "affected": n, "columns": cols, "rows": [list(r) for r in cur.fetchall()],
                        "truncated": n > 10}
            cur = con.execute(sql)
            cols = [d[0] for d in cur.description or []]
            rows = cur.fetchall()
            return {"columns": cols, "rows": [list(r) for r in rows[:limit]], "truncated": len(rows) > limit,
                    "total": len(rows)}
        finally:
            con.close()

    def _full_result(self, sql: str):
        uri = "file:" + str(Path(self.schema.db_path).resolve()).replace("\\", "/") + "?mode=ro"
        con = sqlite3.connect(uri, uri=True)
        try:
            s = sql.strip()
            if s.lower().startswith("delete"):
                s = "SELECT * FROM " + re.sub(r"(?is)^\s*delete\s+from\s+", "", s)
            return con.execute(s).fetchall()
        finally:
            con.close()

    @staticmethod
    def _same(a, b, ordered):
        if ordered:
            return a == b
        k = lambda r: tuple(str(x) for x in r)
        return sorted(a, key=k) == sorted(b, key=k)

    # ---- main entry
    def run(self, question: str, mistake_mode: str = "off", max_attempts: int = 3, seed: Optional[int] = None,
            preferred_mistake: Optional[str] = None) -> dict:
        rng = random.Random(seed)
        steps = []
        t_all = time.perf_counter()

        def step(key, title_ta, title_en, status, data, t0):
            steps.append({"key": key, "title": title_ta, "title_en": title_en, "status": status, "data": data,
                          "ms": round((time.perf_counter() - t0) * 1000, 1)})

        q = question.strip()
        is_tamil = bool(TAMIL_RE.search(q))
        # 1. input
        t0 = time.perf_counter()
        step("input", "உள்ளீடு", "Input", "done", {"text": q, "language": "தமிழ் (Tamil)" if is_tamil else "English"}, t0)

        # 2. user intent (Semmai's own reading of the request)
        t0 = time.perf_counter()
        if is_tamil:
            frame = self.tamil.parse(q)
            canonical_en = render_english(frame)
        else:
            frame, canonical_en = None, q
        intent = self.extractor.extract(canonical_en)
        if frame is not None and frame.op == "delete":
            intent.action, intent.projections = "delete", []
        step("intent", "பயனர் நோக்கம் (Semmai)", "User intent graph", "done",
             {"graph": graph_view(intent), "canonical": canonical_en,
              "note": "Tamil semantic parser - independent of the AI translator" if is_tamil else ""}, t0)

        attempts = []
        final_sql, final_action, verdict_ok = None, None, False
        feedback = ""
        for n in range(1, max_attempts + 1):
            att = {"n": n}
            # 3. translation (AI)
            t0 = time.perf_counter()
            mistake = None
            inject = mistake_mode == "always" or (mistake_mode == "random" and rng.random() < 0.5)
            if n > 1:
                inject = inject and mistake_mode == "always" and n < max_attempts and rng.random() < 0.35
            if is_tamil:
                if isinstance(self.translator, IndicTrans2Translator):
                    english = self.translator.translate_text(q)
                else:
                    use = frame
                    if inject:
                        use, mistake = apply_mistake(frame, rng, preferred_mistake)
                    english = render_english(use)
            else:
                english = q              # English input: mistakes are simulated on the SQL below
            att["translation"] = {"english": english, "backend": self.translator.name if is_tamil else "-",
                                  "mistake": mistake, "mistake_ta": MISTAKES.get(mistake) if mistake else None,
                                  "ms": round((time.perf_counter() - t0) * 1000, 1)}
            # 4. SQL generation (AI)
            t0 = time.perf_counter()
            try:
                sql = self.generator.generate(english, self.schema, self.extractor, feedback)
                gen_err = None
            except Exception as e:  # noqa: BLE001
                sql, gen_err = "", f"{type(e).__name__}: {e}"
            if not is_tamil and inject and sql:
                from .mutations import Mutator
                try:
                    m = Mutator(self.schema, rng)
                    g0 = sql_to_usig(sql, self.schema)
                    for kind in ["OPERATOR_BOUNDARY", "CONSTRAINT_DROP", "NEGATION_FLIP", "VALUE_DRIFT"]:
                        s2 = m.mutate(g0, kind)
                        if s2:
                            sql, mistake = s2, kind
                            att["translation"]["mistake"] = kind
                            break
                except Exception:  # noqa: BLE001
                    pass
            att["sql"] = {"sql": sql, "backend": self.generator.name, "error": gen_err,
                          "ms": round((time.perf_counter() - t0) * 1000, 1)}
            # 5. SQL intent + 6. proof
            t0 = time.perf_counter()
            try:
                action = sql_to_usig(sql, self.schema)
                findings = prove(intent, action, self.schema)
            except UnsupportedSQL as e:
                action, findings = None, []
                att["unverifiable"] = str(e)
            crit = [f for f in findings if f.severity == "critical"]
            ok = action is not None and not crit
            text = verbalize(action, self.types) if action is not None else ""
            sim = self.sim.score(canonical_en, text) if text else None
            att["action_graph"] = graph_view(action) if action is not None else None
            att["verify"] = {"ok": ok, "ledger": ledger(intent, action, self.types) if action is not None else [],
                             "findings": [{"kind": f.kind, "message": f.message, "ta": TA_FINDING.get(f.kind, f.kind)}
                                          for f in crit] or
                                         ([{"kind": "UNVERIFIABLE", "message": att.get("unverifiable", ""),
                                            "ta": TA_FINDING["UNVERIFIABLE"]}] if action is None else []),
                             "similarity": round(sim, 3) if sim is not None else None,
                             "meaning_en": text, "ms": round((time.perf_counter() - t0) * 1000, 1)}
            attempts.append(att)
            if ok:
                final_sql, final_action, verdict_ok = sql, action, True
                break
            feedback = "; ".join(f.message for f in crit) or att.get("unverifiable", "")
        repaired = False
        if not verdict_ok:
            # Semmai repair: compile the SQL from the verified user intent graph
            t0 = time.perf_counter()
            final_sql = render_sql(intent, self.schema)
            final_action = sql_to_usig(final_sql, self.schema)
            repaired = True
            steps.append({"key": "repair", "title": "Semmai சரிசெய்தல்", "title_en": "Semmai repair",
                          "status": "done", "ms": round((time.perf_counter() - t0) * 1000, 1),
                          "data": {"sql": final_sql,
                                   "note": "AI could not produce a meaning-preserving query; Semmai compiled the SQL "
                                           "directly from your verified meaning graph."}})
        steps.append({"key": "attempts", "title": "AI → SQL → சரிபார்ப்பு", "title_en": "AI attempts",
                      "status": "done" if verdict_ok else "blocked", "data": attempts, "ms": 0})

        # 7. execution
        t0 = time.perf_counter()
        result = self._run_sql(final_sql)
        step("execute", "இயக்கம்" if intent.action != "delete" else "சோதனை ஓட்டம் (dry-run)",
             "Execution", "done", {"sql": final_sql, "rows": len(result.get("rows", [])),
                                   "dry_run": result.get("dry_run", False)}, t0)

        # 8. final check: output intent vs user intent
        t0 = time.perf_counter()
        checks = []
        out_findings = [f for f in prove(intent, final_action, self.schema) if f.severity == "critical"]
        checks.append({"ta": "SQL-இன் பொருள் உங்கள் கேள்வியின் பொருளுடன் பொருந்துகிறது", "en": "Output meaning = request meaning",
                       "ok": not out_findings})
        reference_sql = render_sql(intent, self.schema)
        same = self._same(self._full_result(final_sql), self._full_result(reference_sql), bool(intent.order_by))
        checks.append({"ta": "பதில், உங்கள் நோக்க வரைபடத்திலிருந்து தனியாகக் கணக்கிட்ட பதிலுடன் ஒன்றே",
                       "en": "Answer = answer computed independently from your intent graph", "ok": same,
                       "reference_sql": reference_sql})
        aggs = [a for a, _ in intent.projections if a]
        if intent.action == "delete":
            shape_ok = result.get("dry_run", False)
            shape_ta = "நீக்கும் கட்டளை சோதனை ஓட்டமாக மட்டுமே இயக்கப்பட்டது"
        elif aggs and not intent.group_by:
            shape_ok = len(result["rows"]) == 1 and len(result["rows"][0]) == 1
            shape_ta = "ஒரே எண் பதில் கேட்கப்பட்டது - ஒரே எண் கிடைத்தது"
        elif intent.limit:
            shape_ok = len(result["rows"]) <= intent.limit
            shape_ta = f"அதிகபட்சம் {intent.limit} முடிவுகள் கேட்கப்பட்டன - {len(result['rows'])} கிடைத்தன"
        else:
            shape_ok = True
            shape_ta = "பதில் வகை கேள்வியுடன் பொருந்துகிறது"
        checks.append({"ta": shape_ta, "en": "Answer shape matches the question", "ok": shape_ok})
        step("final", "இறுதிச் சரிபார்ப்பு", "Final check: output intent vs user intent",
             "done" if all(c["ok"] for c in checks) else "warn", {"checks": checks}, t0)

        # 9. answer
        answer = self._answer_ta(intent, result, attempts, repaired)
        return {"question": q, "steps": steps, "answer": answer, "result": result, "final_sql": final_sql,
                "verified": verdict_ok, "repaired": repaired, "blocked_attempts": sum(1 for a in attempts if not a["verify"]["ok"]),
                "total_ms": round((time.perf_counter() - t_all) * 1000, 1),
                "backends": {"translator": self.translator.name, "llm": self.generator.name}}

    # ---- Tamil answer
    def _answer_ta(self, intent, result, attempts, repaired) -> dict:
        ent = TA_ENTITY.get(intent.primary_entity, intent.primary_entity)
        lines = []
        blocked = [a for a in attempts if not a["verify"]["ok"]]
        if blocked:
            first = blocked[0]
            why = first["verify"]["findings"][0]["ta"] if first["verify"]["findings"] else ""
            mistake = first["translation"].get("mistake_ta")
            lines.append("⚠️ AI முதலில் உருவாக்கிய SQL உங்கள் கேள்வியின் பொருளை மாற்றியது"
                         + (f" ({mistake})" if mistake else "") + (f": {why}." if why else ".")
                         + " Semmai அதை இயக்காமல் தடுத்தது.")
            if repaired:
                lines.append("AI சரியான வினவலைத் தரவில்லை; எனவே Semmai உங்கள் சரிபார்க்கப்பட்ட நோக்க வரைபடத்திலிருந்து SQL-ஐ உருவாக்கியது.")
            else:
                lines.append(f"{len(attempts)}-வது முயற்சியில் AI சரியான வினவலை உருவாக்கியது; அது சரிபார்க்கப்பட்டு இயக்கப்பட்டது.")
        aggs = [(a, c) for a, c in intent.projections if a]
        rows = result.get("rows", [])
        if intent.action == "delete":
            ent1 = ent[:-3] if ent.endswith("கள்") else ent
            lines.append(f"இந்தக் கட்டளை {result['affected']} {ent1} பதிவுகளை நீக்கும். பாதுகாப்பிற்காக இது ஒரு சோதனை ஓட்டம் "
                         f"மட்டுமே - எந்தத் தரவும் உண்மையில் நீக்கப்படவில்லை. கீழே சில மாதிரிப் பதிவுகள்:")
        elif aggs and not intent.group_by:
            a, c = aggs[0]
            v = rows[0][0] if rows else None
            if a == "count":
                lines.append(f"மொத்தம் {v} {ent} உள்ளனர்." if intent.primary_entity in (
                    "employees", "patients", "doctors", "customers", "students") else f"மொத்தம் {v} {ent} உள்ளன.")
            else:
                money = any(k in (c or "") for k in ("salary", "price", "fee", "amount", "bill", "balance", "budget"))
                vs = "-" if v is None else (f"₹{v:,.0f}" if money else (f"{v:,.2f}" if isinstance(v, float) else str(v)))
                lines.append(f"{TA_AGG.get(a, a)} {ta_col(c)}: {vs}")
        elif intent.group_by:
            lines.append(f"{ta_col(intent.group_by[0])} வாரியான முடிவுகள்:")
        elif not rows:
            lines.append(f"உங்கள் நிபந்தனைகளுக்குப் பொருந்தும் {ent} எவரும் இல்லை.")
        elif intent.limit == 1:
            lines.append(f"பதில்: {rows[0][0]}")
        else:
            total = result.get("total", len(rows))
            more = f" (முதல் {len(rows)} காட்டப்படுகின்றன)" if result.get("truncated") else ""
            lines.append(f"{total} {ent} கிடைத்தனர்{more}:" if intent.primary_entity in (
                "employees", "patients", "doctors", "customers", "students") else f"{total} {ent} கிடைத்தன{more}:")
        return {"text": "\n".join(lines), "columns_ta": [ta_col(c) for c in result.get("columns", [])]}
