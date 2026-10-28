"""Human-readable views of intent graphs: the request-vs-action ledger and
Tamil / English verbalisations used by the chatbot."""
from __future__ import annotations

from .proof import _fmt_raw
from .usig import USIG, canonicalize, values_equal

TA_ENTITY = {"employees": "ஊழியர்கள்", "departments": "துறைகள்", "patients": "நோயாளிகள்", "doctors": "மருத்துவர்கள்",
             "appointments": "சந்திப்புகள்", "customers": "வாடிக்கையாளர்கள்", "products": "பொருட்கள்",
             "orders": "ஆர்டர்கள்", "students": "மாணவர்கள்", "courses": "பாடங்கள்", "enrollments": "சேர்க்கைகள்",
             "accounts": "கணக்குகள்", "transactions": "பரிவர்த்தனைகள்", "flights": "விமானங்கள்",
             "bookings": "முன்பதிவுகள்"}
TA_COL = {"name": "பெயர்", "salary": "சம்பளம்", "age": "வயது", "department": "துறை", "role": "பதவி", "city": "நகரம்",
          "gender": "பாலினம்", "join_date": "சேர்ந்த தேதி", "experience_years": "அனுபவம் (ஆண்டுகள்)",
          "consultation_fee": "ஆலோசனைக் கட்டணம்", "fee": "கட்டணம்", "ward": "வார்டு", "specialty": "சிறப்புத் துறை",
          "admission_date": "அனுமதி தேதி", "bill_amount": "கட்டணத் தொகை", "status": "நிலை", "price": "விலை",
          "category": "வகை", "rating": "மதிப்பீடு", "loyalty_tier": "உறுப்பினர் நிலை", "stock": "இருப்பு",
          "payment_method": "பணம் செலுத்தும் முறை", "amount": "தொகை", "order_date": "ஆர்டர் தேதி",
          "cgpa": "CGPA", "enrollment_date": "சேர்க்கை தேதி", "credits": "கிரெடிட்கள்", "title": "தலைப்பு",
          "marks": "மதிப்பெண்", "semester": "பருவம்", "blood_group": "இரத்த வகை", "order_id": "ஆர்டர் எண்",
          "appointment_id": "சந்திப்பு எண்", "dept_name": "துறை", "location": "இடம்", "budget": "பட்ஜெட்",
          "head": "தலைவர்", "emp_id": "ஊழியர் எண்", "experience": "அனுபவம்"}
TA_AGG = {"count": "எண்ணிக்கை", "avg": "சராசரி", "sum": "மொத்தம்", "max": "அதிகபட்சம்", "min": "குறைந்தபட்சம்"}
TA_OP = {">": ">", ">=": "≥", "<": "<", "<=": "≤", "=": "=", "!=": "≠", "IN": "∈", "NOT IN": "∉",
         "LIKE": "≈", "NOT LIKE": "≉"}
TA_VALUE = {}
try:
    from .tamil import VALUES, DB_VALUES
    for _ta, _en in list(VALUES.items()) + [kv for d in DB_VALUES.values() for kv in d.items()]:
        TA_VALUE.setdefault(_en.lower(), _ta)
except Exception:  # pragma: no cover
    pass


def ta_col(full: str) -> str:
    if not full or full == "*":
        return "பதிவுகள்"
    name = full.split(".")[-1]
    return TA_COL.get(name, name.replace("_", " "))


def ta_val(v) -> str:
    if isinstance(v, tuple):
        return ", ".join(ta_val(x) for x in v)
    if isinstance(v, float) and v.is_integer():
        v = int(v)
    t = TA_VALUE.get(str(v).lower())
    return f"{t} ({v})" if t else str(v)


def ta_constraint(c) -> str:
    return f"{ta_col(c.column)} {TA_OP.get(c.op, c.op)} {ta_val(c.value)}"


def graph_view(g: USIG) -> dict:
    """Tamil + English view of an intent graph for the pipeline panel."""
    if g is None:
        return {}
    shape = []
    if g.action == "delete":
        shape.append(("செயல்", "நீக்கு (DELETE)"))
    aggs = [(a, c) for a, c in g.projections if a]
    if aggs:
        shape.append(("கணக்கீடு", ", ".join(f"{TA_AGG.get(a, a)}({ta_col(c)})" for a, c in aggs)))
    elif g.action != "delete":
        shape.append(("காட்டு", ", ".join(ta_col(c) for _, c in g.projections) or "-"))
    if g.group_by:
        shape.append(("குழு", ", ".join(ta_col(c) for c in g.group_by)))
    if g.order_by:
        _, c, d = g.order_by[0]
        shape.append(("வரிசை", f"{ta_col(c)} {'↓ அதிகம் முதலில்' if d == 'DESC' else '↑ குறைவு முதலில்'}"))
    if g.limit:
        shape.append(("எத்தனை", str(g.limit)))
    return {"entity": TA_ENTITY.get(g.primary_entity, g.primary_entity), "entity_en": g.primary_entity,
            "constraints": [{"ta": ta_constraint(c), "en": c.describe()} for c in g.constraints],
            "logic": "OR" if g.has_or else "AND",
            "shape": [{"k": k, "v": v} for k, v in shape], "raw": g.to_dict()}


def ledger(intent: USIG, action: USIG, types: dict) -> list:
    """Row-by-row comparison of the request (intent) and the SQL (action)."""
    if intent is None or action is None:
        return []
    I, A = canonicalize(intent, types), canonicalize(action, types)
    raw_i = dict(zip(map(id, I.constraints), intent.constraints))
    raw_a = dict(zip(map(id, A.constraints), action.constraints))
    rows = []

    def agg(p):
        a, c = p
        return f"{TA_AGG.get(a, a.upper())}({ta_col(c)})" if a else ta_col(c)

    if intent.action != action.action:
        rows.append({"aspect": "செயல்", "asked": intent.action.upper(), "does": action.action.upper(), "ok": False})
    rows.append({"aspect": "பொருள்", "asked": TA_ENTITY.get(intent.primary_entity, intent.primary_entity or "-"),
                 "does": ", ".join(TA_ENTITY.get(e, e) for e in sorted(action.entities)) or "-",
                 "ok": intent.primary_entity in action.entities})
    i_aggs = [p for p in intent.projections if p[0]]
    a_aggs = [p for p in action.projections if p[0]]
    if i_aggs or a_aggs:
        ok = len(i_aggs) == len(a_aggs) and all(
            x[0] == y[0] and (x[0] == "count" or x[1] == y[1]) for x, y in zip(i_aggs, a_aggs))
        rows.append({"aspect": "கணக்கீடு", "asked": ", ".join(map(agg, i_aggs)) or "பட்டியல்",
                     "does": ", ".join(map(agg, a_aggs)) or "பட்டியல்", "ok": ok})
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
        rows.append({"aspect": "நிபந்தனை", "asked": ta_constraint(raw_i[id(c)]),
                     "does": ta_constraint(raw_a[id(m)]) if m is not None else "விடுபட்டது",
                     "ok": exact is not None, "asked_en": _fmt_raw(raw_i[id(c)], types)})
    for a in A.constraints:
        if id(a) not in used:
            rows.append({"aspect": "நிபந்தனை", "asked": "கேட்கவில்லை", "does": ta_constraint(raw_a[id(a)]), "ok": False})
    if A.has_or and not I.has_or:
        rows.append({"aspect": "தர்க்கம்", "asked": "அனைத்தும் (AND)", "does": "ஏதேனும் ஒன்று (OR)", "ok": False})
    if I.group_by or A.group_by:
        rows.append({"aspect": "குழு", "asked": ", ".join(map(ta_col, I.group_by)) or "இல்லை",
                     "does": ", ".join(map(ta_col, A.group_by)) or "இல்லை",
                     "ok": sorted(I.group_by) == sorted(A.group_by)})
    if I.order_by or A.order_by:
        def od(g):
            if not g.order_by:
                return "வரிசை இல்லை"
            _, c, d = g.order_by[0]
            return f"{ta_col(c)} {'அதிகம் முதலில்' if d == 'DESC' else 'குறைவு முதலில்'}"
        rows.append({"aspect": "வரிசை", "asked": od(I), "does": od(A),
                     "ok": [(c, d) for _, c, d in I.order_by] == [(c, d) for _, c, d in A.order_by]})
    if I.limit or A.limit:
        rows.append({"aspect": "எத்தனை", "asked": str(I.limit or "அனைத்தும்"), "does": str(A.limit or "அனைத்தும்"),
                     "ok": (I.limit or None) == (A.limit or None)})
    return rows


TA_FINDING = {
    "OPERATOR_MISMATCH": "ஒப்பீட்டுக் குறி மாறியுள்ளது (உதா: > என்பது ≥ ஆனது)",
    "VALUE_DRIFT": "மதிப்பு மாறியுள்ளது",
    "NEGATION_FLIP": "'தவிர / அல்லாத' என்ற மறுப்பு மாற்றப்பட்டுள்ளது",
    "CONSTRAINT_MISSING": "நீங்கள் கேட்ட ஒரு நிபந்தனை விடுபட்டுள்ளது",
    "CONSTRAINT_EXTRA": "நீங்கள் கேட்காத நிபந்தனை சேர்க்கப்பட்டுள்ளது",
    "COLUMN_MISMATCH": "நிபந்தனை தவறான பண்பில் பயன்படுத்தப்பட்டுள்ளது",
    "CONSTRAINT_CHANGED": "நிபந்தனை மாற்றப்பட்டுள்ளது",
    "LOGIC_MISMATCH": "'மற்றும்' என்பது 'அல்லது' ஆக மாறியுள்ளது",
    "AGGREGATION_MISMATCH": "கணக்கீட்டு வகை மாறியுள்ளது",
    "PROJECTION_MISMATCH": "தவறான தகவல் காட்டப்படுகிறது",
    "PROJECTION_MISSING": "கேட்ட தகவல் காட்டப்படவில்லை",
    "GROUP_MISMATCH": "குழுவாக்கம் மாறியுள்ளது",
    "ORDER_MISMATCH": "வரிசைப்படுத்தும் பண்பு மாறியுள்ளது",
    "ORDER_DIRECTION": "வரிசை திசை தலைகீழாக மாறியுள்ளது",
    "ORDER_EXTRA": "கேட்காத வரிசைப்படுத்தல் சேர்க்கப்பட்டுள்ளது",
    "LIMIT_MISMATCH": "முடிவுகளின் எண்ணிக்கை வரம்பு மாறியுள்ளது",
    "WRONG_ENTITY": "தவறான அட்டவணை (பொருள்) பயன்படுத்தப்பட்டுள்ளது",
    "ACTION_MISMATCH": "செயல் மாறியுள்ளது (உதா: காட்டு → நீக்கு)",
    "UNVERIFIABLE": "இந்த SQL-ஐ சரிபார்க்க முடியவில்லை",
    "LOW_SIMILARITY": "பொருள் ஒற்றுமை குறைவாக உள்ளது",
}
