@echo off
setlocal
cd /d "%~dp0"
if not exist "src\api.py" (
  echo ERROR: Extract the upgrade into the Parlay Guard folder where src\api.py exists.
  pause
  exit /b 1
)
python install_same_game_compare.py
if errorlevel 1 (
  echo Install failed. No server was started.
  pause
  exit /b 1
)
python -m pytest tests/test_same_game_paper_track.py -q
if errorlevel 1 (
  echo Tests failed. No server was started.
  pause
  exit /b 1
)
set PG_MAX_GAMES=20
echo.
echo Same-game dashboard: http://127.0.0.1:8000/same-game-comparison
echo Existing moneyline dashboard: http://127.0.0.1:8000/moneyline-tracker
echo Keep this window open to record paper picks.
echo.
python -m uvicorn src.api:app --reload --host 127.0.0.1 --port 8000
pause
