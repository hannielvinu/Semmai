# Tamil results

## 1. Understanding Tamil requests (exact intent-graph match with the gold SQL)

| Set | Questions | Exact meaning |
|---|---|---|
| Development (dev) | 48 | 100.0% |
| Development (dev2: mixed Tamil-English, number words, ₹) | 28 | 100.0% |
| **Blind test, first run** (frozen in git before the parser existed) | 50 | **98.0%** |
| Blind test after one post-hoc fix | 50 | 100.0% |

## 2. Verifying SQL against a Tamil request

Gold SQL and meaning-preserving rewrites must be allowed; semantic mutations must be rejected. Thresholds for the similarity baselines were tuned on the development sets.

**Blind set** - 249 SQL candidates (100 correct, 149 wrong)

| Method | Accuracy | F1 | Wrong SQL let through (FAR) | Correct SQL blocked (FRR) |
|---|---|---|---|---|
| Syntax + permission check | 40.2% | 0.0% | 100.0% | 0.0% |
| Similarity: Tamil question vs SQL meaning | 52.6% | 55.3% | 51.0% | 42.0% |
| Similarity: English translation vs SQL meaning | 61.8% | 64.2% | 43.0% | 31.0% |
| **Semmai** | 100.0% | 100.0% | 0.0% | 0.0% |

**All Tamil sets** - 627 SQL candidates (252 correct, 375 wrong)

| Method | Accuracy | F1 | Wrong SQL let through (FAR) | Correct SQL blocked (FRR) |
|---|---|---|---|---|
| Syntax + permission check | 40.2% | 0.0% | 100.0% | 0.0% |
| Similarity: Tamil question vs SQL meaning | 54.4% | 59.4% | 44.3% | 47.6% |
| Similarity: English translation vs SQL meaning | 63.5% | 65.4% | 42.4% | 27.8% |
| **Semmai** | 100.0% | 100.0% | 0.0% | 0.0% |

## 3. Translation mistakes (the project story)

Every Tamil question x every applicable realistic translator mistake. *Silently wrong* = the mistaken SQL returns a different answer than the user asked for.

| Mistake | Cases | Caught by Semmai | Would silently return a wrong answer |
|---|---|---|---|
| filter dropped | 96 | 100% | 99% |
| number changed | 33 | 100% | 100% |
| more than becomes at least | 16 | 100% | 56% |
| year shifted | 14 | 100% | 100% |
| except dropped | 12 | 100% | 100% |
| negation lost | 12 | 100% | 100% |
| average becomes maximum | 12 | 100% | 100% |
| highest becomes lowest | 12 | 100% | 100% |
| top k changed | 8 | 100% | 100% |
| after becomes from | 6 | 100% | 100% |
| before becomes until | 2 | 100% | 50% |
| **All** | 223 | **100.0%** | 96% |

## 4. End-to-end Tamil chatbot (3 seeds over all 126 questions)

| AI mistake rate | Pipeline | Correct answers | Wrong answers executed |
|---|---|---|---|
| 20% | Translate → SQL → Execute | 81.7% | 18.3% |
| 20% | + Semmai (verify, regenerate, repair) | **100.0%** | **0.0%** |
| 40% | Translate → SQL → Execute | 63.0% | 37.0% |
| 40% | + Semmai (verify, regenerate, repair) | **100.0%** | **0.0%** |