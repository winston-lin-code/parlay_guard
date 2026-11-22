@echo off
setlocal
cd /d "%~dp0"
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
set "PG_MAX_GAMES=20"
echo ===============================================
echo Parlay Guard - Multi-Sport EV Portfolio
echo ===============================================
python install_portfolio.py
if errorlevel 1 goto fail
python -m pytest tests/test_portfolio_api.py -q
if errorlevel 1 goto fail
python verify_portfolio.py
if errorlevel 1 goto fail
echo.
echo Open http://127.0.0.1:8000/portfolio
echo.
python -m uvicorn src.api:app --host 127.0.0.1 --port 8000
if errorlevel 1 goto fail
goto end
:fail
echo.
echo Setup or server start failed. See the error above.
pause
:end
endlocal
