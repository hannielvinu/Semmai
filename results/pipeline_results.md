# End-to-end pipeline experiment (Simulated LLM, 3 seeds)

| LLM error rate | Pipeline | Correct answers | Wrong answers executed ↓ | Rejected (no answer) |
|---|---|---|---|---|
| 10% | Generate → Execute | 90.2% | 9.8% | 0.0% |
| 10% | Generate → Verify → Execute/Reject | 88.5% | 0.0% | 11.5% |
| 10% | … + regenerate (≤3 attempts) | 98.1% | 0.0% | 1.9% |
| 20% | Generate → Execute | 80.8% | 19.2% | 0.0% |
| 20% | Generate → Verify → Execute/Reject | 79.5% | 0.0% | 20.5% |
| 20% | … + regenerate (≤3 attempts) | 97.6% | 0.0% | 2.4% |
| 30% | Generate → Execute | 68.5% | 31.5% | 0.0% |
| 30% | Generate → Verify → Execute/Reject | 67.5% | 0.0% | 32.5% |
| 30% | … + regenerate (≤3 attempts) | 96.2% | 0.1% | 3.7% |
| 40% | Generate → Execute | 62.7% | 37.3% | 0.0% |
| 40% | Generate → Verify → Execute/Reject | 61.4% | 0.0% | 38.6% |
| 40% | … + regenerate (≤3 attempts) | 92.4% | 0.2% | 7.4% |