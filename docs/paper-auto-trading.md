# Opt-in paper-only moneyline automation

This is a simulation research pilot. It **never** signs or submits exchange orders, uses real balances, or reports verified fills or realized P&L.

The existing live refresh in `src/api.py` produces a snapshot every 10 minutes. When `PG_ENABLE_PAPER_AUTO=1`, `src.paper_auto_execution.on_live_refresh` records eligible *assumed* NBA moneyline fills in `data/paper_trades/auto_execution.sqlite3`. Default is OFF.

## Windows PowerShell quick start

Run from the repository root:

```powershell
python -m pytest -q tests/test_paper_auto_execution.py tests/test_paper_portfolio.py tests/test_live.py
$env:PG_ENABLE_PAPER_AUTO="1"
python -m uvicorn src.api:app --host 127.0.0.1 --port 8000
```

Open **http://127.0.0.1:8000/paper-auto** for the paper portfolio, or choose **Paper bot portfolio** from the live dashboard. The JSON API is **GET /api/paper-auto/summary** (read-only, always available); `GET /api/live/data` also includes `paper_auto_execution` when enabled after a successful scheduled refresh. This is **not** a live trading switch.

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

- $100 starting **simulated** bankroll, $5 assumed stake, $20 maximum simulated stake per UTC day; never spend more than remaining simulated cash
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

## Settlement and performance tracking

The paper portfolio uses `src.paper_portfolio.settle_pending` against cached **paired, completed NBA team box scores**. Only an exact date, home team, and away team with a unique, internally consistent NBA game is accepted. Missing, ambiguous, postponed, or invalid outcomes remain **PENDING_NBA_VERIFICATION**; they are never counted as losses. This does not establish that Polymarket paid out, resolved the same way, or accepted a live order.

Results are recorded separately in `paper_settlements`, never overwrite assumed fills, and are idempotent. A winning simulated binary stake of $5 at an assumed 0.41 share price produces a simulated $12.20 payout; the losing payout is $0.

The dashboard presents:
- `available_cash`: $100 starting balance minus ALL staked capital plus simulated payouts of verified NBA winners;
- `reserved_open_stakes`: total simulated cost of unresolved entries;
- `book_value_paper_equity`: simulated available cash + unresolved cost basis (NOT a live market valuation);
- realized simulated profit/ROI (only settled entries), W/L, and realized-outcome drawdown;
- cost sensitivity at observed best ask, ask +1¢ (the pilot assumption), and ask +2¢. Those prices are not fills or verified fees.

The initial $100 bankroll is frozen in the local `paper_config` table when the journal first initializes. An automated candidate is rejected if its $5 simulated stake would overdraw this balance. No actual balances, exchange fees, mark-to-market P&L, or deposits are handled.

Settlement checks run during the existing refresh loop **only when** `PG_ENABLE_PAPER_AUTO=1`. The page itself is read-only and polls the local JSON API every 30 seconds without placing new entries or processing settlements. Keep the local server running for new captures and checks; it is not a hosted unattended service.

Data is stored locally under `data/paper_trades/`, which is gitignored. Back up the SQLite file if you need to retain the paper history. The old moneyline/joint strategy trackers remain separate; this new journal does not import or count their bets.

### Offline settlement check

To run outcome reconciliation manually without enabling new entries:

```powershell
python -c "from src.paper_portfolio import settle_pending; print(settle_pending())"
```

No live orders are implemented. Before even considering real trades, examine enough **prospectively recorded** games to validate model calibration, missing lineup impacts, realistic fill prices, and risk limits.

