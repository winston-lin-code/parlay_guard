@echo off
cd /d "%~dp0"
echo Starting Parlay Guard live dashboard on http://127.0.0.1:8000/
python -m uvicorn src.api:app --reload --reload-dir src --host 127.0.0.1 --port 8000
pause
