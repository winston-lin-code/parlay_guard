# Run from the Parlay Guard project root in PowerShell:
#   .\run_dev.ps1
# The server stays running until Ctrl+C.
$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
python -m uvicorn src.api:app --reload --reload-dir . --host 127.0.0.1 --port 8000 --no-access-log
