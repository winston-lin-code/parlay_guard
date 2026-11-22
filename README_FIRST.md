# Parlay Guard — Multi-Sport EV Portfolio Upgrade

This ZIP **adds** a standalone read-only portfolio planner to your existing Parlay Guard project.
It does not replace the NBA dashboard, your college trained models, your frozen moneyline tracker or your same-game paper experiment.

## One-click install on Windows

1. Stop any Uvicorn server currently using port 8000 (Ctrl+C).
2. Extract **all** files in this ZIP directly into your existing project folder:

   `C:\Users\Winston\Desktop\parlay_guard_project\parlay_guard`

3. Double-click `START_PORTFOLIO.cmd` from inside that folder.
4. Open **http://127.0.0.1:8000/portfolio**.

Manual PowerShell alternative (from inside your `parlay_guard` folder):

```powershell
python install_portfolio.py
python -m pytest tests/test_portfolio_api.py -q
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"
$env:PG_MAX_GAMES = "20"
python -m uvicorn src.api:app --reload --host 127.0.0.1 --port 8000
```

The dashboard displays NBA and college matchups from your existing APIs. College basketball/football require the college upgrade's models to be trained and its `/api/college/live` route installed. NHL, NFL, soccer, MLB and WNBA *schedules* load from ESPN without invented EV. Their **EV allocations** require a real independent probability and current Polymarket moneyline ask from the online EV feed or a manual CSV (see `docs/PORTFOLIO_GUIDE.md`).

## Using it

- Select `Today` or `Tomorrow`, choose a sport, and optionally filter for estimated positive EV.
- Tick games you want, select either side's moneyline, and enter a bankroll, exposure percentage, and per-game limit.
- Click **Generate paper allocation** to see suggested dollars and **% of bankroll**; export the output as CSV.
- The default `$40` example has a 5% daily risk budget ($2), 2% max per game ($0.80), quarter-Kelly, 5% minimum estimated EV, and a 1¢ illustrative share-cost allowance. **Unused money stays unallocated.**

No orders are placed. All suggested stakes are research-only and based on uncertain model probabilities, non-guaranteed displayed asks, and an illustrative fee allowance. The tool does not retrieve your real-money balance, validate eligibility for your exchange, or infer that an unreported injury means a player is healthy.
