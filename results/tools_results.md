# Multilingual tool-action verification results

892 training utterances, 703 test utterances (held-out templates; 259 contain slot values never seen in training) -> 2812 action pairs (1406 correct, 1406 mismatched). Thresholds tuned on cross-validated training data: intent 0.10, similarity 0.296.

| Method | Accuracy | Precision | Recall | F1 | FAR ↓ | FRR ↓ |
|---|---|---|---|---|---|---|
| Schema validation | 50.0% | 0.0% | 0.0% | 0.0% | 100.0% | 0.0% |
| Similarity (utterance vs call) | 58.3% | 62.4% | 41.5% | 49.9% | 58.5% | 25.0% |
| Semmai: intent check only | 57.1% | 80.9% | 18.7% | 30.4% | 81.3% | 4.4% |
| Semmai: slot checks only | 95.0% | 100.0% | 89.9% | 94.7% | 10.1% | 0.0% |
| Semmai without numeric-coverage rule | 93.4% | 95.4% | 91.3% | 93.3% | 8.7% | 4.4% |
| Semmai (full) | 93.9% | 95.4% | 92.2% | 93.8% | 7.8% | 4.4% |

## Per language (Semmai full)

| Locale | Intent model accuracy | Accuracy | F1 | FAR ↓ | FRR ↓ | Similarity F1 |
|---|---|---|---|---|---|---|
| de-DE | 84.3% | 94.3% | 94.3% | 7.0% | 4.3% | 51.4% |
| en-US | 82.4% | 93.8% | 93.7% | 7.6% | 4.8% | 23.7% |
| es-ES | 89.7% | 94.6% | 94.5% | 7.3% | 3.4% | 49.7% |
| fr-FR | 75.9% | 92.2% | 92.3% | 6.9% | 8.6% | 46.4% |
| hi-IN | 85.3% | 94.2% | 94.0% | 9.1% | 2.6% | 60.8% |
| ta-IN | 75.7% | 94.3% | 94.2% | 8.7% | 2.6% | 57.1% |

## Per error type (detection rate; acceptance rate for correct calls)

| Type | n | Schema | Similarity | Intent only | Slots only | No numeric rule | Semmai |
|---|---|---|---|---|---|---|---|
| GOLD | 703 | 100% | 75% | 96% | 100% | 96% | 96% |
| REWRITE_FORMAT | 703 | 100% | 75% | 96% | 100% | 96% | 96% |
| INTENT_SWAP | 281 | 0% | 25% | 75% | 77% | 83% | 83% |
| SLOT_VALUE_DRIFT | 281 | 0% | 65% | 5% | 100% | 100% | 100% |
| SLOT_DROP | 281 | 0% | 69% | 5% | 78% | 78% | 83% |
| SLOT_ADD | 282 | 0% | 30% | 4% | 100% | 100% | 100% |
| SLOT_TYPE_SWAP | 281 | 0% | 19% | 5% | 95% | 96% | 96% |