@echo off
REM Windows version of run_all.sh - rebuilds data, runs all evaluations, figures and tests.
cd /d "%~dp0"
set PYTHONUTF8=1
python scripts\build_databases.py || exit /b 1
python scripts\generate_bird_style.py || exit /b 1
python scripts\build_verification_benchmark.py || exit /b 1
python scripts\build_verification_benchmark.py --questions data\bird_style\handwritten.json --out data\bird_style\verification_handwritten.json --dev-frac 0 || exit /b 1
python scripts\build_verification_benchmark.py --questions data\bird_style\blind_test.json --out data\bird_style\verification_blind.json --dev-frac 0 || exit /b 1
python scripts\generate_massive_style.py || exit /b 1
python scripts\evaluate_sql.py || exit /b 1
python scripts\evaluate_tools.py || exit /b 1
python scripts\run_pipeline_experiment.py || exit /b 1
python scripts\evaluate_tamil.py || exit /b 1
python scripts\make_figures.py || exit /b 1
python scripts\make_architecture.py || exit /b 1
python -m unittest discover -s tests || exit /b 1
echo.
echo All done. Results are in the results folder.
