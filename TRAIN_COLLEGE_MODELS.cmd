@echo off
cd /d "%~dp0"
set "PYTHONIOENCODING=utf-8"
set "PYTHONUTF8=1"
python -m src.college_train --sports mbb,cfb --refresh-current
pause
