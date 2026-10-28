"""Tamil semantic parser:  Tamil request  ->  meaning frame  ->  English  ->  USIG.

Tamil is agglutinative and verb-final, so the meaning words come with case
suffixes and the comparison words come *after* the number:

    2022க்குப் பிறகு சேர்ந்த ஊழியர்களைக் காட்டு
    2022 + DAT   after  joined  employees+ACC show
    ->  "Show employees who joined after 2022"   ->  join_date > 2022-12-31

Pipeline
    1. tokenise; split numbers from glued suffixes (2022க்கு -> 2022 + க்கு);
       drop sandhi consonants (ஊழியர்களைக் -> ஊழியர்களை)
    2. lexicon lookup with suffix-aware stem matching
       (சென்னையைச் -> சென்னை + யை ; சம்பளத்தின்படி -> சம்பளம் + த்தின்படி)
    3. concept grammar -> a structured Frame (entity, operation, filters,
       ranking, grouping) - each filter keeps its operator explicitly, e.g.
       பிறகு -> '>' , முதல் -> '>=' , தவிர -> NOT
    4. the Frame is rendered as an English sentence (the built-in translator)
       and parsed by Semmai's English intent extractor into the USIG.

The Frame is also what the "AI translator" mistake simulator perturbs
(after -> from, dropping "except", ...), so wrong translations can be
demonstrated and caught.
"""
from __future__ import annotations

import copy
import re
from dataclasses import dataclass, field
from typing import Optional

from .schema import Schema

# ----------------------------------------------------------------- lexicon
VERB_SHOW = ["காட்டு", "காட்டவும்", "காண்பி", "காண்பிக்கவும்", "பட்டியலிடு", "பட்டியலிடவும்", "கொடு", "தா",
             "தேடு", "கண்டுபிடி", "சொல்", "சொல்லு", "தெரிவி", "வேண்டும்", "பார்க்க"]
VERB_DELETE = ["நீக்கு", "நீக்கவும்", "அழி", "அழிக்கவும்", "நீக்கிவிடு", "டெலீட்"]
COUNT_WORDS = ["எத்தனை", "எண்ணிக்கை", "எண்ணிக்கையை"]
WHO = ["யார்", "யாவர்"]
WHAT = ["என்ன", "எவ்வளவு"]
PEOPLE = ["பேர்", "நபர்", "நபர்கள்", "பெறுபவர்", "உள்ளவர்", "வாங்குபவர்"]
AGG = {"சராசரி": "avg", "மொத்த": "sum", "மொத்தம்": "sum", "கூட்டுத்தொகை": "sum", "உச்ச": "max"}
# அதிகபட்சம் / குறைந்தபட்சம் are "at most / at least" before a number, otherwise MAX / MIN (see parse)
# superlatives (when NOT directly after a number)
SUP = {"அதிக": "DESC", "அதிகம்": "DESC", "அதிகமான": "DESC", "உயர்ந்த": "DESC", "கூடுதல்": "DESC",
       "குறைந்த": "ASC", "குறைவான": "ASC", "குறைவு": "ASC", "கீழான": "ASC"}
SUP_WITH_COL = {"மூத்த": ("age", "DESC"), "முதிய": ("age", "DESC"), "இளைய": ("age", "ASC"),
                "மலிவான": ("price", "ASC"), "மலிவு": ("price", "ASC"), "விலையுயர்ந்த": ("price", "DESC"),
                "விலைமதிப்புள்ள": ("price", "DESC")}
INTENSIFIER = ["மிக", "மிகவும்", "மிகச்", "மிகக்", "அதிகபட்ச"]
# comparators that come AFTER the number
CMP_AFTER = {
    ">": ["மேல்", "மேற்பட்ட", "மேலான", "மேலாக", "அதிகமான", "அதிகம்", "அதிகமாக", "அதிக", "கூடுதலான",
          "மிகுதியான", "மிஞ்சிய"],
    "<": ["குறைவான", "குறைவாக", "குறைவு", "குறைந்த", "கீழ்", "கீழே", "கீழான", "உட்பட்ட", "குறைவாகவே"],
}
CMP_PHRASE = {  # multi-word, after the number
    ">=": ["அல்லது அதற்கு மேல்", "அல்லது அதற்கு மேற்பட்ட", "அல்லது அதற்கு மேலான", "அல்லது அதற்கும் மேல்",
           "அல்லது அதற்கு அதிகமான", "அல்லது அதற்கும் அதிகமான", "அல்லது அதிகமான", "அல்லது மேல்",
           "அல்லது மேற்பட்ட", "அல்லது அதற்கு மேற்பட்டவர்கள்"],
    "<=": ["அல்லது அதற்கு குறைவான", "அல்லது அதற்கும் குறைவான", "அல்லது குறைவான", "அல்லது அதற்கு கீழ்",
           "அல்லது அதற்கு குறைவாக", "அல்லது குறைந்த", "அல்லது அதற்கு குறைந்த"],
}
CMP_BEFORE = {">=": ["குறைந்தது", "குறைந்தபட்சம்", "குறைந்தபட்ச"], "<=": ["அதிகபட்சம்", "அதிகபட்ச", "மிஞ்சாமல்"]}
DATE_CMP = {">": ["பிறகு", "பின்", "பின்னர்", "பின்பு", "அப்புறம்"], "<": ["முன்", "முன்பு", "முன்னர்", "முன்னால்"],
            ">=": ["முதல்", "இருந்து", "லிருந்து", "தொடங்கி"], "<=": ["வரை", "வரையில்", "வரைக்கும்"]}
BETWEEN_WORDS = ["இடையில்", "இடையே", "இடைப்பட்ட", "நடுவில்"]
NEG_AFTER = ["தவிர", "தவிர்த்து", "நீங்கலாக", "அல்லாத", "இல்லாத", "அல்லாதவர்கள்", "இல்லாதவர்கள்", "அல்லாதோர்",
             "சேராத", "தவிர்ந்த"]
OR_WORDS = ["அல்லது"]
AND_WORDS = ["மற்றும்"]
EACH = ["ஒவ்வொரு"]
WISE = ["வாரியாக", "வாரியான", "வாரி", "படி"]
ORDER_WORDS = ["வரிசைப்படுத்து", "வரிசைப்படுத்தி", "வரிசையில்", "வரிசைப்படி", "வரிசைப்படுத்தவும்"]
DESC_WORDS = ["இறங்கு", "இறங்குமுக", "பெரியதிலிருந்து"]
ASC_WORDS = ["ஏறு", "ஏறுமுக", "சிறியதிலிருந்து"]
NAME_WORDS = ["பெயர்"]
OTHER_WORDS = ["மற்ற", "பிற", "அனைத்து", "எல்லா", "எல்லாம்", "உள்ள", "உள்ளனர்", "உள்ளன", "இருக்கும்"]
YEAR_WORDS = ["ஆண்டு", "வருடம்", "ஆண்டில்", "வருடத்தில்"]
MONEY_WORDS = ["ரூபாய்", "ரூ", "ரூபா"]
SCALE = {"ஆயிரம்": 1e3, "ஆயிர": 1e3, "லட்சம்": 1e5, "லட்ச": 1e5, "இலட்சம்": 1e5, "கோடி": 1e7}
NUM_WORDS = {"ஒன்று": 1, "ஒரு": 1, "இரண்டு": 2, "மூன்று": 3, "நான்கு": 4, "ஐந்து": 5, "ஆறு": 6, "ஏழு": 7,
             "எட்டு": 8, "ஒன்பது": 9, "பத்து": 10, "இருபது": 20, "முப்பது": 30, "நாற்பது": 40, "ஐம்பது": 50,
             "அறுபது": 60, "எழுபது": 70, "எண்பது": 80, "தொண்ணூறு": 90, "நூறு": 100}
MONTHS = {"ஜனவரி": 1, "பிப்ரவரி": 2, "மார்ச்": 3, "மார்ச்சு": 3, "ஏப்ரல்": 4, "மே": 5, "ஜூன்": 6, "ஜூலை": 7,
          "ஆகஸ்ட்": 8, "ஆகஸ்டு": 8, "செப்டம்பர்": 9, "அக்டோபர்": 10, "நவம்பர்": 11, "டிசம்பர்": 12}
MONTH_EN = ["", "January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
            "November", "December"]

# Tamil word -> English value (the English parser links it to the right column)
VALUES = {
    # cities
    "சென்னை": "Chennai", "வேலூர்": "Vellore", "கோயம்புத்தூர்": "Coimbatore", "கோவை": "Coimbatore",
    "மதுரை": "Madurai", "பெங்களூரு": "Bengaluru", "பெங்களூர்": "Bengaluru", "ஹைதராபாத்": "Hyderabad",
    "ஐதராபாத்": "Hyderabad", "மும்பை": "Mumbai", "டெல்லி": "Delhi", "தில்லி": "Delhi", "கொல்கத்தா": "Kolkata",
    "புனே": "Pune", "சேலம்": "Salem", "திருச்சி": "Tiruchirappalli", "திருச்சிராப்பள்ளி": "Tiruchirappalli",
    "கொச்சி": "Kochi",
    # people
    "பெண்": "Female", "ஆண்": "Male",
    # hr departments / roles
    "பொறியியல்": "Engineering", "விற்பனை": "Sales", "மனிதவள": "HR", "மனிதவளம்": "HR", "நிதி": "Finance",
    "சந்தைப்படுத்தல்": "Marketing", "மார்க்கெட்டிங்": "Marketing", "ஆதரவு": "Support", "சப்போர்ட்": "Support",
    "பயிற்சியாளர்": "Intern", "இன்டர்ன்": "Intern", "பயிற்சி": "Intern", "பொறியாளர்": "Engineer",
    "இன்ஜினியர்": "Engineer", "ஆய்வாளர்": "Analyst", "நிர்வாகி": "Executive", "குழுத் தலைவர்": "Team Lead",
    "குழு தலைவர்": "Team Lead", "டீம் லீட்": "Team Lead", "மேலாளர்": "Manager", "மேனேஜர்": "Manager",
    # hospital
    "தீவிர சிகிச்சை பிரிவு": "ICU", "தீவிர சிகிச்சை": "ICU", "பொது வார்டு": "General", "மகப்பேறு": "Maternity",
    "குழந்தைகள் வார்டு": "Pediatric", "எலும்பு வார்டு": "Orthopedic", "இதய வார்டு": "Cardiac",
    "இதயநோய்": "Cardiology", "இதயவியல்": "Cardiology", "இதய": "Cardiology",
    "நரம்பியல்": "Neurology", "எலும்பியல்": "Orthopedics", "குழந்தை நல": "Pediatrics", "தோல்": "Dermatology",
    "புற்றுநோய்": "Oncology", "பொது மருத்துவம்": "General Medicine",
    "ரத்து செய்யப்பட்ட": "Cancelled", "ரத்தான": "Cancelled", "ரத்து": "Cancelled",
    "முடிந்த": "Completed", "முடிக்கப்பட்ட": "Completed", "திட்டமிடப்பட்ட": "Scheduled", "வராத": "No-show",
    # retail
    "மின்னணு": "Electronics", "எலக்ட்ரானிக்": "Electronics", "ஆடை": "Clothing", "உடை": "Clothing",
    "மளிகை": "Groceries", "புத்தக": "Books", "புத்தகம்": "Books", "மரச்சாமான்": "Furniture",
    "பொம்மை": "Toys", "வெண்கல": "Bronze", "வெள்ளி": "Silver", "தங்க": "Gold", "கோல்டு": "Gold",
    "பிளாட்டினம்": "Platinum", "அட்டை": "Card", "கார்டு": "Card", "பணம்": "Cash", "ரொக்கம்": "Cash",
    "ரொக்க": "Cash", "நெட்பேங்கிங்": "NetBanking", "டெலிவரி செய்யப்பட்ட": "Delivered",
    "வழங்கப்பட்ட": "Delivered", "திருப்பி அனுப்பப்பட்ட": "Returned", "திருப்பியனுப்பப்பட்ட": "Returned",
    "அனுப்பப்பட்ட": "Shipped",
}
DB_VALUES = {  # database-specific meanings
    "university": {"கணினி அறிவியல்": "CSE", "மின்னணு": "ECE", "மின்னணுவியல்": "ECE", "மின் பொறியியல்": "EEE",
                   "இயந்திரவியல்": "Mechanical", "இயந்திர": "Mechanical", "கட்டடவியல்": "Civil",
                   "சிவில்": "Civil", "தகவல் தொழில்நுட்பம்": "IT", "உயிரி தொழில்நுட்பம்": "Biotech"},
}
ENTITIES = {
    "hr": {"ஊழியர்": "employees", "பணியாளர்": "employees", "ஊழியர்கள்": "employees", "தொழிலாளர்": "employees",
           "ஸ்டாஃப்": "employees"},
    "hospital": {"நோயாளி": "patients", "நோயாளர்": "patients", "மருத்துவர்": "doctors", "டாக்டர்": "doctors",
                 "சந்திப்பு": "appointments", "அப்பாயின்ட்மென்ட்": "appointments"},
    "retail": {"வாடிக்கையாளர்": "customers", "பொருள்": "products", "பொருட்": "products", "தயாரிப்பு": "products",
               "ஆர்டர்": "orders", "ஆணை": "orders"},
    "university": {"மாணவர்": "students", "மாணவி": "students", "பாடம்": "courses", "பாடங்": "courses",
                   "கோர்ஸ்": "courses"},
}
PERSON_TABLE = {"hr": "employees", "hospital": "patients", "retail": "customers", "university": "students",
                "finance": "accounts", "travel": "bookings"}
COLUMNS = {  # Tamil word -> English column phrase understood by the English parser
    "சம்பளம்": "salary", "சம்பள": "salary", "ஊதியம்": "salary", "சம்பாதி": "salary", "சம்பாதிக்கும்": "salary",
    "வயது": "age", "வயதுடைய": "age", "துறை": "department", "பதவி": "role", "நகரம்": "city", "நகர": "city",
    "ஊர்": "city", "பாலினம்": "gender", "அனுபவம்": "experience", "அனுபவ": "experience",
    "ஆலோசனை கட்டணம்": "consultation fee", "கட்டணம்": "fee", "விலை": "price", "மதிப்பீடு": "rating",
    "ரேட்டிங்": "rating", "தொகை": "amount", "இருப்பு": "stock", "கையிருப்பு": "stock", "கிரெடிட்": "credits",
    "மதிப்பெண்": "marks", "வார்டு": "ward", "பிரிவு": "category", "வகை": "category", "பட்ஜெட்": "budget",
    "நிலை": "status",
}
DATE_VERBS = {"சேர்ந்த": None, "சேர்ந்தவர்கள்": None, "சேர்ந்தவர்": None, "இணைந்த": None, "அனுமதிக்கப்பட்ட": None,
              "செய்யப்பட்ட": None, "வைக்கப்பட்ட": None, "பதிவுசெய்த": None, "சேர்க்கப்பட்ட": None,
              "திறக்கப்பட்ட": None}
DATE_PHRASE = {"hr": "joined", "hospital": "admitted", "retail": "placed", "university": "enrolled",
               "finance": "opened", "travel": "booked"}

VOWEL_SIGNS = set(chr(c) for c in range(0x0BBE, 0x0BCD))
SUFFIX_START = VOWEL_SIGNS | set("கசதபயவமலனஙஆஇஉஐஎஒஓ")
SANDHI = ("க்", "ச்", "த்", "ப்")


def _norm_token(t: str) -> str:
    for s in SANDHI:
        if t.endswith(s) and len(t) > len(s) + 2:
            return t[: -len(s)]
    return t


def match_word(tok: str, key: str) -> Optional[str]:
    """Return the suffix if ``tok`` is ``key`` + a plausible Tamil suffix, else None."""
    if tok == key:
        return ""
    variants = [(key, "full")]
    if key.endswith("ம்") and len(key) > 3:
        variants.append((key[:-2], "m"))
    if key.endswith("ு") and len(key) > 3:
        variants.append((key[:-1], "u"))
    if key.endswith("்") and len(key) > 3:
        variants.append((key[:-1], "v"))
    for v, kind in variants:
        if not tok.startswith(v) or len(tok) == len(v):
            if tok == v:
                return ""
            continue
        rem = tok[len(v):]
        if kind == "m" and not (rem.startswith("த்") or rem.startswith("ம") or rem.startswith("ங்")):
            continue
        if kind in ("u", "v") and rem[0] not in VOWEL_SIGNS:
            continue
        if kind == "full" and len(v) <= 2 and rem not in ("யில்", "இல்", "மாதம்", "மாதத்தில்"):
            continue
        if rem[0] in SUFFIX_START:
            return rem
    return None


@dataclass
class Tok:
    text: str              # normalised token
    raw: str
    num: Optional[float] = None
    suffix: str = ""       # suffix glued to a number / matched word


@dataclass
class C:                   # a concept
    kind: str
    i: int                 # first token index
    j: int                 # one past last token
    val: object = None
    suffix: str = ""


@dataclass
class Clause:
    kind: str                       # val | num | date
    col: Optional[str] = None       # English column phrase
    op: str = "="                   # =, !=, IN, NOT IN, >, >=, <, <=, between, in
    values: tuple = ()              # val: English values ; num: (v,) or (lo, hi) ; date: (y, m) or ((y,m),(y,m))
    gran: str = ""                  # date granularity: year | month | day
    src: str = ""                   # Tamil words that produced it


@dataclass
class Frame:
    db: str
    entity: str
    op: str = "list"                # list | count | agg | delete
    agg: Optional[str] = None
    agg_col: Optional[str] = None
    clauses: list = field(default_factory=list)
    order_col: Optional[str] = None
    order_dir: Optional[str] = None
    limit: Optional[int] = None
    group_col: Optional[str] = None
    extra_cols: list = field(default_factory=list)
    sorted_only: bool = False       # ORDER BY without LIMIT
    notes: list = field(default_factory=list)

    def copy(self):
        return copy.deepcopy(self)


class TamilParser:
    def __init__(self, schema: Schema):
        self.schema = schema
        self.db = schema.db_id
        self.values = dict(VALUES)
        self.values.update(DB_VALUES.get(self.db, {}))
        self.entities = ENTITIES.get(self.db, {})
        lex = []
        add = lambda words, kind, val=None: lex.extend((w, kind, val) for w in words)
        add(VERB_SHOW, "SHOW"); add(VERB_DELETE, "DELETE"); add(COUNT_WORDS, "COUNT"); add(WHO, "WHO")
        add(WHAT, "WHAT"); add(PEOPLE, "PEOPLE"); add(INTENSIFIER, "INTENS"); add(NEG_AFTER, "NEG")
        add(OR_WORDS, "OR"); add(AND_WORDS, "AND"); add(EACH, "EACH"); add(WISE, "WISE"); add(ORDER_WORDS, "ORDER")
        add(DESC_WORDS, "DESCW"); add(ASC_WORDS, "ASCW"); add(NAME_WORDS, "NAME"); add(OTHER_WORDS, "FILLER")
        add(YEAR_WORDS, "YEARW"); add(MONEY_WORDS, "MONEY"); add(BETWEEN_WORDS, "BETWEEN")
        for w, v in AGG.items():
            lex.append((w, "AGG", v))
        for w, v in SUP.items():
            lex.append((w, "SUP", v))
        for w, v in SUP_WITH_COL.items():
            lex.append((w, "SUPCOL", v))
        for op, ws in CMP_AFTER.items():
            add(ws, "CMP", op)
        for op, ws in CMP_PHRASE.items():
            add(ws, "CMP", op)
        for op, ws in CMP_BEFORE.items():
            add(ws, "PCMP", op)
        for op, ws in DATE_CMP.items():
            add(ws, "DCMP", op)
        for w, v in SCALE.items():
            lex.append((w, "SCALE", v))
        for w, v in NUM_WORDS.items():
            lex.append((w, "NUMW", v))
        for w, v in MONTHS.items():
            lex.append((w, "MONTH", v))
        for w, v in self.values.items():
            lex.append((w, "VAL", v))
        for w, v in self.entities.items():
            lex.append((w, "ENT", v))
        for w, v in COLUMNS.items():
            lex.append((w, "COL", v))
        add(list(DATE_VERBS), "DATEVERB")
        # multi-word keys are matched token-by-token; order: more tokens, then longer text
        self.lex = sorted(((tuple(_norm_token(t) for t in w.split()), k, v) for w, k, v in lex),
                          key=lambda x: (-len(x[0]), -sum(len(t) for t in x[0])))
        # English words that are database values or column synonyms
        self.en_values = {}
        for t in schema.tables.values():
            for c in t.columns.values():
                for v in c.values:
                    if isinstance(v, str) and re.fullmatch(r"[A-Za-z0-9+\- ]+", v):
                        self.en_values.setdefault(v.lower(), v)
        self.en_cols = {}
        for t in schema.tables.values():
            for c in t.columns.values():
                for s in c.synonyms:
                    if re.fullmatch(r"[a-z]+", s):
                        self.en_cols.setdefault(s, s)

    # -------------------------------------------------------------- tokens
    def tokenize(self, text: str) -> list:
        text = text.replace("₹", " ₹ ").replace("?", " ").replace("!", " ").replace("।", " ")
        text = re.sub(r"(?<=\D)\.(?=\s|$)", " ", text)
        text = re.sub(r"[,;:\"'()\[\]]", lambda m: " " if m.group() != "," else ",", text)
        raw = text.split()
        toks = []
        for r in raw:
            r = r.strip(",")
            if not r:
                continue
            m = re.match(r"^₹?(\d[\d,]*(?:\.\d+)?)[-]?(.*)$", r)
            if m:
                num = float(m.group(1).replace(",", ""))
                suf = _norm_token(m.group(2)) if m.group(2) else ""
                toks.append(Tok(m.group(1), r, num, suf))
                continue
            if r == "₹":
                continue
            n = _norm_token(r)
            if toks and toks[-1].num is not None and not toks[-1].suffix and n in (
                    "க்கு", "க்கும்", "இல்", "ல்", "ஆம்", "லிருந்து", "இலிருந்து"):
                toks[-1].suffix = n
                continue
            toks.append(Tok(n, r))
        return toks

    def concepts(self, toks: list) -> list:
        out = []
        i = 0
        while i < len(toks):
            t = toks[i]
            if t.num is not None:
                out.append(C("NUM", i, i + 1, t.num, t.suffix))
                i += 1
                continue
            if re.fullmatch(r"[A-Za-z][A-Za-z0-9+\-]*", t.text):
                lw = t.text.lower()
                # multi-word English values ("Team Lead", "General Medicine")
                if i + 1 < len(toks) and f"{lw} {toks[i + 1].text.lower()}" in self.en_values:
                    out.append(C("VAL", i, i + 2, self.en_values[f"{lw} {toks[i + 1].text.lower()}"]))
                    i += 2
                    continue
                if lw in self.en_values:
                    out.append(C("VAL", i, i + 1, self.en_values[lw]))
                elif lw in self.en_cols:
                    out.append(C("COL", i, i + 1, lw))
                else:
                    out.append(C("EN", i, i + 1, t.text))
                i += 1
                continue
            hit = None
            for key, kind, val in self.lex:
                n = len(key)
                if i + n > len(toks):
                    continue
                sufs = []
                ok = True
                for k in range(n):
                    s = match_word(toks[i + k].text, key[k])
                    if s is None:
                        ok = False
                        break
                    sufs.append(s)
                if ok:
                    hit = C(kind, i, i + n, val, sufs[-1])
                    break
            if hit:
                out.append(hit)
                i = hit.j
            else:
                out.append(C("UNK", i, i + 1, t.text))
                i += 1
        return out

    # ------------------------------------------------------------ grammar
    def parse(self, text: str) -> Frame:
        toks = self.tokenize(text)
        cs = self.concepts(toks)
        kept = []
        for c in cs:                         # "1000 ரூபாய்க்கு": the suffix belongs to the number
            if c.kind in ("MONEY", "YEARW") and kept and kept[-1].kind == "NUM" and not kept[-1].suffix \
                    and c.suffix in ("க்கு", "க்கும்", "களுக்கு", "களுக்கும்", "உக்கு"):
                kept[-1].suffix = "க்கு"
            if c.kind == "SCALE" and kept and kept[-1].kind == "NUM":
                pass
            kept.append(c)
        cs = [c for c in kept if c.kind not in ("FILLER", "UNK", "MONEY", "EN")]
        # merge number word / scale words into numbers
        merged = []
        for c in cs:
            if c.kind == "NUMW":
                merged.append(C("NUM", c.i, c.j, float(c.val), c.suffix))
                continue
            if c.kind == "SCALE" and merged and merged[-1].kind == "NUM":
                merged[-1].val = merged[-1].val * c.val
                merged[-1].suffix = c.suffix
                merged[-1].j = c.j
                merged[-1].kind = "NUM"
                continue
            merged.append(c)
        cs = merged
        used = set()
        f = Frame(self.db, self._entity(cs))
        self._cur_entity = f.entity

        def k(idx):
            return cs[idx].kind if 0 <= idx < len(cs) else None

        # ---- operation words
        kinds = [c.kind for c in cs]
        if "DELETE" in kinds:
            f.op = "delete"
        # ---- months glued to years:  2024 ஜனவரி / ஜனவரி 2024
        dates = []      # (start_idx, end_idx, y, m, suffix)
        idx = 0
        while idx < len(cs):
            c = cs[idx]
            if c.kind == "NUM" and 1900 <= c.val <= 2100 and float(c.val).is_integer():
                if k(idx + 1) == "MONTH":
                    dates.append((idx, idx + 1, int(c.val), cs[idx + 1].val, cs[idx + 1].suffix)); idx += 2; continue
                if k(idx - 1) == "MONTH" and (idx - 1) not in {d[0] for d in dates}:
                    dates.append((idx - 1, idx, int(c.val), cs[idx - 1].val, c.suffix)); idx += 1; continue
                nxt = k(idx + 1)
                yearish = c.suffix in ("இல்", "ல்", "ஆம்", "லிருந்து", "இலிருந்து") or nxt in ("DCMP", "YEARW", "DATEVERB") \
                    or (c.suffix in ("க்கு", "க்கும்") and nxt == "DCMP") or k(idx - 1) == "DCMP" \
                    or (nxt == "NUM" and k(idx + 2) in ("DCMP", "BETWEEN")) or nxt == "BETWEEN"
                if yearish:
                    dates.append((idx, idx, int(c.val), None, c.suffix)); idx += 1; continue
            idx += 1
        date_idx = set()
        for a, b, *_ in dates:
            date_idx.update(range(a, b + 1))

        # ---- date clauses
        di = 0
        while di < len(dates):
            a, b, y, m, suf = dates[di]
            nxt = b + 1
            while k(nxt) == "YEARW":
                nxt += 1
            op = "in"
            if k(nxt) == "DCMP":
                op = cs[nxt].val
                used.add(nxt)
            elif suf in ("லிருந்து", "இலிருந்து"):
                op = ">="
            if di + 1 < len(dates) and dates[di + 1][0] == b + 1 and k(dates[di + 1][1] + 1) == "BETWEEN":
                a2, b2, y2, m2, _ = dates[di + 1]               # 2016க்கும் 2018க்கும் இடையில்
                used.update(range(a, b2 + 2))
                f.clauses.append(Clause("date", op="between", values=((y, m), (y2, m2)),
                                        gran="month" if m else "year", src=self._src(toks, cs[a].i, cs[b2 + 1].j)))
                di += 2
                continue
            if op == ">=" and di + 1 < len(dates):            # 2018 முதல் 2020 வரை
                a2, b2, y2, m2, suf2 = dates[di + 1]
                n2 = b2 + 1
                while k(n2) == "YEARW":
                    n2 += 1
                if k(n2) == "DCMP" and cs[n2].val == "<=":
                    used.add(n2)
                    f.clauses.append(Clause("date", op="between", values=((y, m), (y2, m2)),
                                            gran="month" if m else "year", src=self._src(toks, cs[a].i, cs[n2].j)))
                    used.update(range(a, n2 + 1))
                    di += 2
                    continue
            f.clauses.append(Clause("date", op=op, values=(y, m), gran="month" if m else "year",
                                    src=self._src(toks, cs[a].i, cs[min(nxt, len(cs) - 1)].j)))
            used.update(range(a, b + 1))
            di += 1

        # ---- prefix comparators that are really aggregates (குறைந்தபட்ச சம்பளம் = MIN salary)
        for idx, c in enumerate(cs):
            if c.kind == "PCMP" and k(idx + 1) != "NUM":
                c.kind = "AGG"
                c.val = "min" if c.val == ">=" else "max"

        # ---- numeric clauses
        for idx, c in enumerate(cs):
            if c.kind != "NUM" or idx in used or idx in date_idx:
                continue
            # range: N (col)? முதல் M (col)? வரை
            j = idx + 1
            col = None
            if k(j) == "COL":
                col = cs[j].val; j += 1
            if k(j) == "DCMP" and cs[j].val == ">=" and k(j + 1) == "NUM":
                j2 = j + 2
                col2 = None
                if k(j2) == "COL":
                    col2 = cs[j2].val; j2 += 1
                if k(j2) == "DCMP" and cs[j2].val == "<=":
                    col = col or col2 or self._col_before(cs, idx, used)
                    f.clauses.append(Clause("num", col=col, op="between", values=(c.val, cs[j + 1].val),
                                            src=self._src(toks, c.i, cs[j2].j)))
                    used.update(range(idx, j2 + 1))
                    if not (col or col2):
                        pass
                    continue
            if k(idx + 1) == "NUM" and k(idx + 2) == "BETWEEN":          # 40000க்கும் 60000க்கும் இடையில்
                col = self._col_before(cs, idx, used)
                if col is None and k(idx + 3) == "COL":
                    col = cs[idx + 3].val; used.add(idx + 3)
                f.clauses.append(Clause("num", col=col, op="between", values=(c.val, cs[idx + 1].val),
                                        src=self._src(toks, c.i, cs[idx + 2].j)))
                used.update({idx, idx + 1, idx + 2})
                continue
            if k(idx - 1) == "DCMP" and cs[idx - 1].val == ">=":           # "முதல் 10" = top 10
                continue
            # comparator after the number (optionally with a unit and a column between)
            j = idx + 1
            col = None
            if k(j) == "YEARW":
                used.add(j); j += 1
            if k(j) == "COL":
                col = cs[j].val; j += 1
            op = None
            sup_ok = k(j) == "SUP" and (c.suffix in ("க்கு", "க்கும்") or col is not None)
            if k(j) == "CMP" or sup_ok:
                op = cs[j].val if k(j) == "CMP" else (">" if cs[j].val == "DESC" else "<")
                used.add(j)
                end = j
            elif k(idx - 1) == "PCMP":
                op = cs[idx - 1].val
                used.add(idx - 1)
                end = j - 1
            elif c.suffix in ("க்கு", "க்கும்") and k(j) == "DCMP":
                # "50000க்கு பிறகு" is not a sensible comparison; treat as '>'
                op = ">" if cs[j].val in (">", ">=") else "<"
                used.add(j); end = j
            elif col is not None and not c.suffix:                       # "35 வயதுடைய" = aged 35
                op = "="
                end = j - 1
            else:
                continue
            if col is None:
                # column after the comparator: "50000க்கு மேல் சம்பளம்"
                if k(end + 1) == "COL":
                    col = cs[end + 1].val; used.add(end + 1)
                else:
                    col = self._col_before(cs, idx, used)
            f.clauses.append(Clause("num", col=col, op=op, values=(c.val,), src=self._src(toks, c.i, cs[end].j)))
            used.add(idx)
            if col and k(idx + 1) == "COL":
                used.add(idx + 1)

        # ---- values (with OR/AND lists and negation)
        vi = 0
        val_idx = [i for i, c in enumerate(cs) if c.kind == "VAL"]
        while vi < len(val_idx):
            group = [val_idx[vi]]
            while vi + 1 < len(val_idx) and val_idx[vi + 1] - group[-1] == 2 and \
                    k(group[-1] + 1) in ("OR", "AND") and \
                    self._same_col(cs[group[-1]].val, cs[val_idx[vi + 1]].val):
                group.append(val_idx[vi + 1]); vi += 1
            last = group[-1]
            neg = False
            for n in (last + 1, last + 2):          # "சென்னையைத் தவிர", "விற்பனைத் துறையில் இல்லாத"
                if k(n) == "NEG":
                    neg = True; used.add(n); break
                if k(n) not in ("COL", "ENT"):
                    break
            vals = tuple(cs[g].val for g in group)
            colname = self._value_col_phrase(vals[0])
            op = ("NOT IN" if neg else "IN") if len(vals) > 1 else ("!=" if neg else "=")
            f.clauses.append(Clause("val", col=colname, op=op, values=vals,
                                    src=self._src(toks, cs[group[0]].i, cs[min(last + (2 if neg else 1), len(cs) - 1)].j)))
            used.update(group)
            vi += 1

        # ---- aggregates
        has_count = "COUNT" in kinds
        for idx, c in enumerate(cs):
            if c.kind == "AGG" and idx not in used:
                colc = next((cs[j] for j in range(idx + 1, min(idx + 4, len(cs))) if cs[j].kind == "COL"), None)
                if colc is not None and not (c.val == "sum" and k(idx + 1) == "COUNT"):
                    f.agg, f.agg_col = c.val, colc.val
                    break
        # ---- superlatives / ranking
        if f.agg is None:
            for idx, c in enumerate(cs):
                if idx in used:
                    continue
                if c.kind == "SUPCOL":
                    f.order_col, f.order_dir = self._col_phrase_for(c.val[0]), c.val[1]
                    break
                if c.kind == "SUP" and (k(idx - 1) != "NUM" or (k(idx - 2) == "DCMP" and cs[idx - 2].val == ">=")):
                    colc = next((cs[j] for j in range(idx + 1, min(idx + 3, len(cs))) if cs[j].kind == "COL"), None)
                    if colc is None and k(idx - 1) == "COL":
                        colc = cs[idx - 1]
                    if colc is not None:
                        f.order_col, f.order_dir = colc.val, c.val
                        break
            if f.order_col:
                lim = None
                for idx, c in enumerate(cs):
                    if c.kind == "NUM" and idx not in used and idx not in date_idx and float(c.val).is_integer() \
                            and k(idx + 1) in ("ENT", "VAL", "PEOPLE", "COL") and c.val <= 1000:
                        lim = int(c.val); used.add(idx); break
                    if c.kind == "DCMP" and c.val == ">=" and k(idx + 1) == "NUM":      # "முதல் 5" = top 5
                        lim = int(cs[idx + 1].val); used.update({idx, idx + 1}); break
                f.limit = lim or 1
        # ---- explicit ordering: சம்பளத்தின்படி இறங்கு வரிசையில்
        for idx, c in enumerate(cs):
            if c.kind == "ORDER":
                colc = next((cs[j] for j in range(idx - 1, max(idx - 4, -1), -1)
                             if cs[j].kind == "COL" and cs[j].suffix.endswith("படி")), None) or \
                    next((cs[j] for j in range(len(cs)) if cs[j].kind == "COL" and cs[j].suffix.endswith("படி")), None)
                if colc is not None:
                    f.order_col = colc.val
                    f.order_dir = "DESC" if "DESCW" in kinds else "ASC"
                    f.limit = None
                    f.sorted_only = True
                    if colc.val not in f.extra_cols:
                        f.extra_cols.append(colc.val)
        # ---- grouping
        for idx, c in enumerate(cs):
            if c.kind == "EACH" and k(idx + 1) == "COL":
                f.group_col = cs[idx + 1].val; break
            if c.kind == "COL" and (c.suffix.endswith("வாரியாக") or k(idx + 1) == "WISE") and \
                    not c.suffix.endswith("படி"):
                f.group_col = c.val; break
        # ---- projections: பெயர் மற்றும் சம்பளம்
        for idx, c in enumerate(cs):
            if c.kind == "NAME" and k(idx + 1) == "AND" and k(idx + 2) == "COL":
                if cs[idx + 2].val not in f.extra_cols:
                    f.extra_cols.insert(0, cs[idx + 2].val)
        # ---- operation
        if f.op != "delete":
            if f.agg:
                f.op = "agg"
            elif has_count:
                f.op = "count"
        if f.group_col and f.op == "list":
            f.op = "count"
        return f

    # ------------------------------------------------------------ helpers
    @staticmethod
    def _src(toks, i, j):
        return " ".join(t.raw for t in toks[i:j])

    def _entity(self, cs) -> str:
        ent = next((c.val for c in cs if c.kind == "ENT"), None)
        if ent:
            return ent
        # from a value (a role/department/... belongs to a table)
        for c in cs:
            if c.kind == "VAL":
                t = self._value_table(c.val)
                if t:
                    return t
        return PERSON_TABLE.get(self.db) or next(iter(self.schema.tables))

    def _value_table(self, value):
        pref = PERSON_TABLE.get(self.db)
        hits = [t.name for t in self.schema.tables.values() for col in t.columns.values()
                if any(str(v).lower() == str(value).lower() for v in col.values)]
        if pref in hits:
            return pref
        return hits[0] if hits else None

    def _value_col_phrase(self, value, entity=None):
        tables = list(self.schema.tables.values())
        ent = entity or getattr(self, "_cur_entity", None)
        tables.sort(key=lambda t: (t.name != ent, t.name != PERSON_TABLE.get(self.db)))
        for t in tables:
            for col in t.columns.values():
                if any(str(v).lower() == str(value).lower() for v in col.values):
                    return col.name.replace("_", " ")
        return None

    def _same_col(self, a, b):
        return self._value_col_phrase(a) == self._value_col_phrase(b)

    def _col_phrase_for(self, name):
        return name

    @staticmethod
    def _col_before(cs, idx, used):
        for j in (idx - 1, idx - 2):
            if 0 <= j < len(cs) and cs[j].kind == "COL" and j not in used:
                used.add(j)
                return cs[j].val
        return None


# ------------------------------------------------------- Frame -> English
OP_EN = {">": "more than", ">=": "at least", "<": "less than", "<=": "at most", "=": "exactly"}
DATE_OP_EN = {">": "after", ">=": "since", "<": "before", "<=": "until", "in": "in", "=": "on"}
AGG_EN = {"avg": "average", "sum": "total", "max": "maximum", "min": "minimum"}
VAL_TEMPLATES = {
    "city": ("from {v}", "not from {v}"), "department": ("in the {v} department", "not in the {v} department"),
    "gender": ("who are {v}", "who are not {v}"),
}


def _fmt_num(v):
    return str(int(v)) if float(v).is_integer() else str(v)


def _date_text(y, m):
    return f"{MONTH_EN[m]} {y}" if m else str(y)


def render_english(f: Frame) -> str:
    ent = f.entity
    date_verb = DATE_PHRASE.get(f.db, "dated")
    parts = []
    for c in f.clauses:
        if c.kind == "val":
            vals = list(c.values)
            vtxt = " or ".join(vals)
            neg = c.op in ("!=", "NOT IN")
            tpl = VAL_TEMPLATES.get(c.col or "")
            if tpl:
                parts.append((tpl[1] if neg else tpl[0]).format(v=vtxt))
            else:
                parts.append(f"whose {c.col or 'value'} is {'not ' if neg else ''}{vtxt}")
        elif c.kind == "num":
            col = c.col or ""
            if c.op == "between":
                parts.append(f"with {col} between {_fmt_num(c.values[0])} and {_fmt_num(c.values[1])}".replace("  ", " "))
            else:
                parts.append(f"with {col} {OP_EN[c.op]} {_fmt_num(c.values[0])}".replace("  ", " "))
        elif c.kind == "date":
            if c.op == "between":
                (y1, m1), (y2, m2) = c.values
                parts.append(f"who {date_verb} from {_date_text(y1, m1)} until {_date_text(y2, m2)}")
            else:
                y, m = c.values
                parts.append(f"who {date_verb} {DATE_OP_EN[c.op]} {_date_text(y, m)}")
    cond = " and ".join(parts)
    cond = (" " + cond) if cond else ""
    sing = ent[:-1] if ent.endswith("s") else ent
    if f.op == "delete":
        return f"Delete {ent}{cond}."
    if f.group_col:
        if f.agg:
            return f"What is the {AGG_EN[f.agg]} {f.agg_col} of {ent}{cond} in each {f.group_col}?"
        return f"How many {ent}{cond} are there in each {f.group_col}?"
    if f.op == "agg":
        return f"What is the {AGG_EN[f.agg]} {f.agg_col} of {ent}{cond}?"
    if f.op == "count":
        return f"How many {ent}{cond} are there?"
    if f.order_col and not f.sorted_only:
        word = "highest" if f.order_dir == "DESC" else "lowest"
        if (f.limit or 1) == 1:
            return f"Which {sing}{cond} has the {word} {f.order_col}?"
        return f"Show the {f.limit} {ent}{cond} with the {word} {f.order_col}."
    if f.sorted_only:
        cols = " and ".join(["name"] + [c for c in f.extra_cols if c != "name"])
        return f"Show the {cols} of {ent}{cond} sorted by {f.order_col} {'descending' if f.order_dir == 'DESC' else 'ascending'}."
    if f.extra_cols:
        return f"Show the name and {' and '.join(f.extra_cols)} of {ent}{cond}."
    return f"Show {ent}{cond}."
