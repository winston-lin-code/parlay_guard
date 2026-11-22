@echo off
setlocal
cd /d "%~dp0"
if not exist "src\api.py" (
  echo ERROR: Extract this ZIP directly into your parlay_guard project folder.
  echo It should contain src\api.py and install_moneyline_paper_tracker.py.
  pause
  exit /b 1
)
where python >nul 2>nul
if errorlevel 1 (
  echo ERROR: Python was not found on PATH.
  echo Open the PowerShell environment you normally use for Parlay Guard.
  pause
  exit /b 1
)
if not exist "install_moneyline_paper_tracker.py" (
  echo ERROR: Missing installer. Extract all files from the ZIP into the project root.
  pause
  exit /b 1
)
echo.
echo === Parlay Guard: paper-only moneyline tracker setup ===
python install_moneyline_paper_tracker.py
if errorlevel 1 goto :fail
echo.
echo === Running tracker tests ===
python -m pytest tests/test_moneyline_paper_track.py -q
if errorlevel 1 goto :fail
echo.
echo === Frozen strategy report ===
python -m src.moneyline_paper_track report
if errorlevel 1 goto :fail
set "PG_MAX_GAMES=20"
echo.
echo === Starting local Parlay Guard tracker ===
echo Read-only dashboard: http://127.0.0.1:8000/moneyline-tracker
echo Keep this window open to record new pregame paper selections.
echo No real trades will be placed.
echo Press Ctrl+C to stop. If port 8000 is occupied, stop the old server first.
echo.
python -m uvicorn src.api:app --reload --host 127.0.0.1 --port 8000
if errorlevel 1 goto :fail
exit /b 0
:fail
echo.
echo ERROR: Setup, tests, report or server startup failed. See details above.
pause
exit /b 1
