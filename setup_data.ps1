$ErrorActionPreference = "Stop"

Write-Host "Collecting NBA historical data..."
python -m src.collect_prop_history
if ($LASTEXITCODE -ne 0) { throw "Data collection failed" }

Write-Host "Generating V2 features..."
python -m src.player_models.features
if ($LASTEXITCODE -ne 0) { throw "Feature generation failed" }

Write-Host "Data setup complete!"