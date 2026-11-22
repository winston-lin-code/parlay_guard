@echo off
setlocal
cd /d "%~dp0"
echo =============================================
echo Parlay Guard - Moneyline bankroll dashboard
 echo =============================================
python install_moneyline_bankroll.py
if errorlevel 1 goto :failed
python -m pytest tests/test_moneyline_budget.py -q
if errorlevel 1 goto :failed
set PG_MAX_GAMES=20
python -m uvicorn src.api:app --reload --host 127.0.0.1 --port 8000
if errorlevel 1 goto :failed
goto :end
:failed
echo.
echo Setup or server failed. Read the error above.
pause
:end
endlocal
