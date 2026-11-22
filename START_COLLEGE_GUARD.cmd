@echo off
setlocal
cd /d "%~dp0"
set "PYTHONIOENCODING=utf-8"
set "PYTHONUTF8=1"
set "PG_MAX_GAMES=20"
echo ===========================================
echo Parlay Guard - College Sports EV Upgrade
echo ===========================================
python install_college_guard.py
if errorlevel 1 goto :failed
python -c "import pyarrow"
if errorlevel 1 (
  echo Installing parquet support for college historical training...
  python -m pip install pyarrow
  if errorlevel 1 goto :failed
)
echo Loading public historical schedules and training independent models...
python -m src.college_train --sports mbb,cfb --refresh-current
if errorlevel 1 echo College training returned an error. Dashboard will clearly show any missing models.
python -m pytest tests/test_college_guard.py -q
if errorlevel 1 goto :failed
echo Visit http://127.0.0.1:8000/college
python -m uvicorn src.api:app --reload --host 127.0.0.1 --port 8000
if errorlevel 1 goto :failed
exit /b 0
:failed
echo Setup or server failed. Error above.
pause
exit /b 1
