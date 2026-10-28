@echo off
REM Semmai local demo - double-click this file. Needs only Python 3.9+ (no Anaconda, no pip installs).
cd /d "%~dp0"
set PYTHONUTF8=1
set PY=
where py >nul 2>nul && set PY=py -3
if not defined PY (
  python -c "import sys; sys.exit(0 if sys.version_info >= (3,9) else 1)" >nul 2>nul && set PY=python
)
if not defined PY (
  echo.
  echo  Python 3 was not found on this computer.
  echo  Install it once, then double-click start_demo.bat again:
  echo    1. Open https://www.python.org/downloads/ and click "Download Python".
  echo    2. Run the installer and TICK "Add python.exe to PATH" before clicking Install.
  echo  Or, in a terminal:  winget install Python.Python.3.12
  echo.
  pause
  exit /b 1
)
echo Starting Semmai at http://localhost:8000 ...
%PY% app\server.py
pause
