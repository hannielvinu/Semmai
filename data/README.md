# Datasets

Everything in this folder is generated deterministically by the scripts in `scripts/`
(fixed random seeds), so `./run_all.sh` recreates it byte-for-byte.

## 1. `bird_style/` - text-to-SQL (BIRD format)

| File | Contents |
|---|---|
| `databases/<db>/<db>.sqlite` | 5 SQLite databases: **hospital** (patients, doctors, appointments), **retail** (customers, products, orders), **university** (students, courses, enrollments), **finance** (accounts, transactions), **travel** (flights, bookings). 13 tables, 8,907 rows in total. |
| `schema_meta.json` | Column descriptions / synonyms (the equivalent of BIRD's `database_description` CSVs). |
| `questions.json` | **379** natural-language questions with gold SQL in BIRD's `dev.json` format (`question_id, db_id, question, evidence, SQL, difficulty`). 11 question types: filters, multi-filters, negation, IN-lists, ranges, counts, aggregates, top-k, ordering, GROUP BY, joins. |
| `handwritten.json` | **60** free-form questions written by hand (development check set). |
| `blind_test.json` | **80** free-form questions written and committed to git **before** the system was run on them; no tuning was done afterwards. |
| `verification_*.json` | The verification benchmarks built from the three question files (see below). |

### Verification benchmark (`scripts/build_verification_benchmark.py`)

For every question: the gold SQL and one **meaning-preserving rewrite** (label 1 = allow),
plus up to three **semantic mutations** (label 0 = must be rejected).

* Rewrites (9 kinds): table aliases, lower-case keywords, reordered conditions, `BETWEEN`,
  boundary-equivalent operators (`age > 60` ≡ `age >= 61`), single-value `IN`, `strftime('%Y')`,
  `COUNT(pk)`, literal on the left (`60 < age`). **Every rewrite is verified by execution** to
  return exactly the gold result.
* Mutations (15 kinds) follow the error taxonomy in the project slides: operator boundary
  (`>` → `>=`), operator reversal, value drift, date drift, negation flip, dropped constraint,
  extra constraint, wrong column, wrong entity/table, wrong aggregate, wrong sort direction,
  wrong LIMIT, AND→OR, wrong projection, wrong GROUP BY.
* `exec_equal` records whether a mutation happens to return the *same rows* as the gold
  query on the current data - the errors that execution-based checking cannot see.

| Set | Pairs | Correct | Mismatched | Split |
|---|---|---|---|---|
| `verification_pairs.json` | 1,886 | 758 | 1,128 | 30% dev (threshold tuning) / 70% test, split by question |
| `verification_handwritten.json` | 300 | 120 | 180 | test |
| `verification_blind.json` | 398 | 160 | 238 | test |

## 2. `massive_style/` - multilingual intents + slots (MASSIVE format)

`massive_style.jsonl`: **1,595** utterances in **6 locales** (en-US, ta-IN, hi-IN, fr-FR, es-ES,
de-DE) × **12 intents** (alarm_set, alarm_remove, weather_query, play_music, calendar_set,
email_sendemail, transport_ticket, takeaway_order, iot_hue_lightoff, iot_hue_lighton,
qa_currency, news_query) with 11 slot types, using the Amazon MASSIVE jsonl schema
(`id, locale, partition, scenario, intent, utt, annot_utt`).

* Train (892) and test (703) use **different sentence templates** (5 per intent per language:
  3 train, 2 test).
* 259 test utterances contain slot values **never seen in training**.

## Using the official datasets

The official downloads were not reachable from the environment the project was built in,
so the benchmarks above were constructed in the same formats. The code runs unchanged on
the real data:

```bash
# BIRD (https://bird-bench.github.io/) - unzip dev.zip to data/official/bird/dev
python scripts/build_verification_benchmark.py --bird-root data/official/bird/dev --out data/official/bird_pairs.json
python scripts/evaluate_sql.py --pairs data/official/bird_pairs.json --bird-root data/official/bird/dev --out-prefix bird

# MASSIVE 1.1 (https://github.com/alexa/massive) - extract to data/official/massive
python scripts/evaluate_tools.py --data data/official/massive/data --locales en-US,ta-IN,hi-IN,fr-FR,es-ES,de-DE
```

On official BIRD, gold queries with sub-queries / CASE / arithmetic are outside the SQL
subset Semmai currently compiles; they are reported as skipped, and at run time such
actions are rejected as `UNVERIFIABLE` (fail-closed).
