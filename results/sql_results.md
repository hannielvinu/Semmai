# SQL verification results

Similarity backend: `tfidf`. Thresholds tuned on the dev split only: similarity_sql = 0.446, similarity_text = 0.661, semmai_gate = 0.000 (balanced accuracy objective).

ROC-AUC of the similarity score for separating correct from mismatched actions: generated: question-vs-SQL 0.629, question-vs-meaning 0.621; handwritten: question-vs-SQL 0.631, question-vs-meaning 0.607; blind: question-vs-SQL 0.610, question-vs-meaning 0.601 (0.5 = chance).

## generated_test (1319 pairs: 530 correct, 789 mismatched; 62 mismatches return the same rows as the gold query)

| Method | Accuracy | Precision | Recall | F1 | FAR ↓ | FRR ↓ |
|---|---|---|---|---|---|---|
| Syntax + permission check | 40.2% | 0.0% | 0.0% | 0.0% | 100.0% | 0.0% |
| Execution sanity (non-empty) | 41.6% | 80.6% | 3.2% | 6.1% | 96.8% | 1.1% |
| Similarity: question vs SQL | 57.5% | 69.4% | 51.8% | 59.4% | 48.2% | 34.0% |
| Similarity: question vs SQL meaning | 61.9% | 66.3% | 74.1% | 70.0% | 25.9% | 56.2% |
| Semmai (proof engine only) | 100.0% | 100.0% | 100.0% | 100.0% | 0.0% | 0.0% |
| Semmai (proof + similarity) | 100.0% | 100.0% | 100.0% | 100.0% | 0.0% | 0.0% |
| Execution-match oracle* | 95.3% | 100.0% | 92.1% | 95.9% | 7.9% | 0.0% |

## handwritten (300 pairs: 120 correct, 180 mismatched; 14 mismatches return the same rows as the gold query)

| Method | Accuracy | Precision | Recall | F1 | FAR ↓ | FRR ↓ |
|---|---|---|---|---|---|---|
| Syntax + permission check | 40.0% | 0.0% | 0.0% | 0.0% | 100.0% | 0.0% |
| Execution sanity (non-empty) | 42.3% | 81.8% | 5.0% | 9.4% | 95.0% | 1.7% |
| Similarity: question vs SQL | 59.0% | 66.9% | 62.8% | 64.8% | 37.2% | 46.7% |
| Similarity: question vs SQL meaning | 60.0% | 62.3% | 84.4% | 71.7% | 15.6% | 76.7% |
| Semmai (proof engine only) | 100.0% | 100.0% | 100.0% | 100.0% | 0.0% | 0.0% |
| Semmai (proof + similarity) | 100.0% | 100.0% | 100.0% | 100.0% | 0.0% | 0.0% |
| Execution-match oracle* | 95.3% | 100.0% | 92.2% | 96.0% | 7.8% | 0.0% |

## blind (398 pairs: 160 correct, 238 mismatched; 15 mismatches return the same rows as the gold query)

| Method | Accuracy | Precision | Recall | F1 | FAR ↓ | FRR ↓ |
|---|---|---|---|---|---|---|
| Syntax + permission check | 40.2% | 0.0% | 0.0% | 0.0% | 100.0% | 0.0% |
| Execution sanity (non-empty) | 44.2% | 83.3% | 8.4% | 15.3% | 91.6% | 2.5% |
| Similarity: question vs SQL | 55.3% | 66.7% | 50.4% | 57.4% | 49.6% | 37.5% |
| Similarity: question vs SQL meaning | 61.1% | 63.0% | 84.5% | 72.2% | 15.5% | 73.8% |
| Semmai (proof engine only) | 97.0% | 95.2% | 100.0% | 97.5% | 0.0% | 7.5% |
| Semmai (proof + similarity) | 97.0% | 95.2% | 100.0% | 97.5% | 0.0% | 7.5% |
| Execution-match oracle* | 96.2% | 100.0% | 93.7% | 96.7% | 6.3% | 0.0% |

\* needs the gold query's result, so it cannot be used at run time.

## Detection rate per error type (all test sets)

| Error type | n | Syntax | Exec non-empty | Similarity | Semmai | Exec-match oracle* |
|---|---|---|---|---|---|---|
| AGGREGATION_SWAP | 106 | 0% | 2% | 72% | 100% | 97% |
| COLUMN_SWAP | 98 | 0% | 27% | 90% | 100% | 98% |
| CONSTRAINT_ADD | 112 | 0% | 4% | 71% | 100% | 93% |
| CONSTRAINT_DROP | 107 | 0% | 0% | 91% | 100% | 97% |
| DATE_DRIFT | 67 | 0% | 9% | 84% | 100% | 78% |
| ENTITY_SWAP | 13 | 0% | 0% | 100% | 100% | 100% |
| GROUP_SWAP | 27 | 0% | 0% | 56% | 100% | 100% |
| LIMIT_CHANGE | 34 | 0% | 0% | 59% | 100% | 100% |
| LOGIC_AND_TO_OR | 90 | 0% | 0% | 71% | 100% | 100% |
| NEGATION_FLIP | 103 | 0% | 2% | 69% | 100% | 98% |
| OPERATOR_BOUNDARY | 101 | 0% | 1% | 89% | 100% | 50% |
| OPERATOR_REVERSE | 104 | 0% | 1% | 82% | 100% | 99% |
| ORDER_FLIP | 47 | 0% | 0% | 64% | 100% | 98% |
| PROJECTION_SWAP | 95 | 0% | 2% | 72% | 100% | 98% |
| VALUE_DRIFT | 103 | 0% | 9% | 83% | 100% | 97% |

## Acceptance rate of meaning-preserving rewrites

| Rewrite | n | Similarity | Semmai |
|---|---|---|---|
| GOLD | 405 | 39% | 99% |
| REWRITE_ALIASES | 105 | 43% | 99% |
| REWRITE_BETWEEN | 3 | 0% | 100% |
| REWRITE_BOUNDARY_EQUIVALENT | 27 | 7% | 96% |
| REWRITE_COUNT_KEY | 21 | 38% | 100% |
| REWRITE_IN_SINGLE | 58 | 29% | 98% |
| REWRITE_LITERAL_LEFT | 34 | 24% | 100% |
| REWRITE_LOWERCASE | 119 | 45% | 98% |
| REWRITE_REORDER_CONDITIONS | 36 | 28% | 97% |
| REWRITE_YEAR_FUNCTION | 2 | 0% | 100% |

Mean verification latency: 2.55 ms (median 2.02 ms, p95 5.70 ms).