# Opt-in paper-only moneyline automation

This is a simulation research pilot. It **never** signs or submits exchange orders, uses real balances, or reports verified fills or realized P&L.

The existing live refresh in `src/api.py` produces a snapshot every 10 minutes. When `PG_ENABLE_PAPER_AUTO=1`, `src.paper_auto_execution.on_live_refresh` records eligible *assumed* NBA moneyline fills in `data/paper_trades/auto_execution.sqlite3`. Default is OFF.

## Windows PowerShell quick start

Run from the repository root:

```powershell
python -m pytest -q tests/test_paper_auto_execution.py
$env:PG_ENABLE_PAPER_AUTO="1"
python -m uvicorn src.api:app --host 127.0.0.1 --port 8000
```

The output of `GET http://127.0.0.1:8000/api/live/data` contains `paper_auto_execution` when enabled, after a successful scheduled refresh. This is **not** a live trading switch.

To disable:

```powershell
$env:PG_ENABLE_PAPER_AUTO="0"
```

The optional offline CLI can also process a previously saved JSON snapshot:

```powershell
python -m src.paper_auto_execution --snapshot data/example_live_snapshot.json
```

Only do this with a fresh snapshot with real quote timestamps; a stale example will be skipped. The journal persists independently of the dashboard.

## Initial fixed simulation policy

- $5 assumed stake, $20 maximum simulated stake per UTC day
- single NBA moneyline on the model's predicted winner only
- model probability reduced by 5 percentage points, observed ask increased by 1 cent for a hypothetical fill
- stressed expected ROI must be at least 10%
- quote at most 120 seconds old, forecast at most 6 hours old
- 10 minutes to 3 hours before tipoff
- require actual CLOB token ID and at least $5 at the observed top ask
- reject invalid/missing data, never infer missing quote times
- one entry per event across refreshes and restarts (SQLite unique key)

The simulated price is **not** a confirmed fill; $0.01 is **not** verified fees/slippage. Current Elo estimates are not proven calibrated, and team availability/injury data are not confirmed. Quote top depth can disappear. Nothing here establishes positive expectancy.

## Before any real trading integration

Validate prospective out-of-sample probability calibration; add timestamped injuries and lineups, platform-specific contract and market mapping, no-play/void settlement logic, actual fee schedule, measured order-book VWAP and slippage, balance and position reconciliation, daily loss and exposure caps, kill switch, retries and failure alerts. Confirm authorized API access and jurisdiction-specific eligibility. Do not use the international CLOB for real trading without confirming that access is permitted.

The current journal does not settle its assumed fills or calculate P&L. Keep the pre-existing independent paper trackers for verified outcome evaluation and build explicit settlement reconciliation before relying on any performance statistic.
