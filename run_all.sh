#!/usr/bin/env bash
# Reproduce every dataset, result table and figure from scratch (~1 minute on a laptop CPU).
set -e
cd "$(dirname "$0")"
python scripts/build_databases.py
python scripts/generate_bird_style.py
python scripts/build_verification_benchmark.py
python scripts/build_verification_benchmark.py --questions data/bird_style/handwritten.json --out data/bird_style/verification_handwritten.json --dev-frac 0
python scripts/build_verification_benchmark.py --questions data/bird_style/blind_test.json --out data/bird_style/verification_blind.json --dev-frac 0
python scripts/generate_massive_style.py
python scripts/evaluate_sql.py
python scripts/evaluate_tools.py
python scripts/run_pipeline_experiment.py
python scripts/evaluate_tamil.py
python scripts/make_figures.py
python -m unittest discover -s tests
python scripts/make_architecture.py
