# Historical market-line audit — Parlay Guard

The prior `calibrate_game_lines` upgrade used **synthetic forecast-centered thresholds**. This package instead evaluates point predictions at **recorded real, timestamped bookmaker home spreads and game totals**. Nothing is bought or sold; snapshots in `data/paper_trades/` remain untouched.

## Why real, timestamped odds?

Bookmaker point spreads and totals move before game time. A closing point line is not necessarily what an early prediction could have traded; a same-day odds CSV without quote timestamps can leak future information. This tool requires (a) a source snapshot time, (b) the actual scheduled tipoff, and (c) a strictly pregame lead window. For leakage safety, this tool requires the source snapshot to be from the **same US Eastern calendar date** as the game; quotes recorded earlier may precede games included in the model's prior-day features. An offered reference price is needed only to show *hypothetical* single-contract returns; a legitimate point-line export without prices can still be used to measure Brier score and ECE.

- Source provider example: [The Odds API historical odds v4](https://the-odds-api.com/liveapi/guides/v4/#get-historical-odds). The historical endpoint requires a paid plan; historical snapshots before a requested time are returned and API credits are consumed. The tool makes **no provider calls** unless you explicitly run `fetch-one`. It never loops over dates or secretly spends credits.
- Other documented CSV exports work if all required columns can be supplied with credible provenance. Do not invent missing timestamps, odds, or outcomes.

## Quick start

Run inside your existing `parlay_guard` folder after installing this upgrade and restarting Uvicorn:

```powershell
python -m pytest tests/ -q
python -m src.market_line_validation report
```

`report` should say no historical market-line audit yet until you actually import verified quotes. That is the correct result.

### Option A — import a directory of historical snapshot JSON files

Historical The Odds API response envelope contains `timestamp` and `data` (events); markets must be requested with `oddsFormat=decimal`.

```powershell
python -m src.market_line_validation import-snapshots --input .\data\historical_odds_json --output .\data\historical_market_lines.csv
```

Every JSON must be a genuine historical response. Normalization keeps exactly the **HOME** spread outcome and **OVER** total outcome, with per-bookmaker price, signed point, actual UTC snapshot time, UTC tipoff, game date in US Eastern, and provider event ID.

### Option B — explicitly fetch **one** historical snapshot from The Odds API

Historical access is paid, uses provider credits, and needs an API key. The endpoint cost for two markets and one region is documented as 20 credits per successful timestamp query; verify your plan first. This example is one request only. Put the key in your local shell, not in source control or chat.

```powershell
$env:ODDS_API_KEY = Read-Host 'The Odds API key'
python -m src.market_line_validation fetch-one --at 2025-11-19T22:00:00Z --region us --bookmaker draftkings --confirm-paid-request --output .\data\historical_odds_json\2025-11-19T2200Z.json
```

The response's `timestamp` is what the pipeline records; the requested `--at` is **not** blindly treated as a valid quote time. The command refuses to overwrite an existing archived snapshot. You must intentionally acquire enough validation- and test-period quotes for a meaningful evaluation.

### Option C — normalized CSV from another historical provider

Columns: `game_date` (NBA ET YYYY-MM-DD), `home_team`, `away_team` (NBA abbreviation or explicit full name), `game_start_utc`, `snapshot_utc` (timezone-aware ISO8601), `bookmaker`, `market` (`spreads` or `totals`), `side` (`HOME` or `OVER`), `point`. Optional `decimal_odds` is required for price/ROI analysis (must be decimal odds >1), and optional `market_update_utc` must be timezone-aware when present. Historical lines are signed from the home team's perspective: e.g., home `-5.5` means home must win by at least six points. For totals, `220.5 OVER` means combined final points exceed 220.5.

Missing reference price is permitted for calibration, but ROI will remain unavailable. The evaluation only scores the chosen bookmaker. Use `--bookmaker` to declare it *before* examining the test results. Fix a quote horizon (default 60 minutes or more before tipoff, maximum 36 hours old). It picks the last quote **at or before** that horizon, once per game and market. Do not select the sportsbook or side with best backtested ROI after seeing outcomes.

### Run market-line validation

```powershell
python -m src.market_line_validation validate --bookmaker draftkings --quotes .\data\historical_market_lines.csv
python -m src.market_line_validation report
```

Uses cached NBA results in `data/nba_games_updated.csv` and `data/nba_2026_27_latest_raw.csv`, from the existing game-line pipeline. You can also supply a preverified **one-row-per-game** CSV with `--completed-csv` and columns `GAME_ID, GAME_DATE, HOME_TEAM, AWAY_TEAM, HOME_PTS, AWAY_PTS`.

The validator writes a JSON artifact at `models/market_line_validation_v1.json` and a row-level audit at `data/market_line_validation_audit.csv`. The dashboard's **Historical market-line audit** panel and `GET /api/market-line-validation` show the same JSON. It does not override the live probabilities.

## Chronology and controls

- Game-line model is trained on games **before 2024-10-01**.
- Forecast residual standard deviations are learned from **2024-10-01 to 2025-09-30** (validation, no test outcomes).
- If validation also contains enough historical *market* quotes, an isotonic probability map is fit on those quotes and tested separately on **2025-10-01 to 2026-06-30** (untouched test).
- At least 100 resolved validation and 100 resolved test market quotes **per market** are required by default; supported held-out test quotes must also meet the threshold. If only test quotes exist, baseline test Brier and ECE are still reported but the market calibrator is **not fitted**.
- Reliability estimates use Brier score and 10-bin ECE; quotes from the same slate/game are not statistically independent. The reported per-game ROI is an unfilled **reference-price scenario**, never an account statement or confirmed Polymarket ROI.
- Market offers with missing or untrusted timestamps, games that cannot match a unique NBA matchup/date, quotes after the pregame cutoff, invalid team mappings, duplicate schedule rows, and ambiguous pregame feature identifications are excluded.
- Pushes are 0 ROI and excluded from binary calibration. One HOME spread and one OVER total outcome per game prevents doubling sample sizes through opposite sides.
- **Never** silently transfer a pass on a bookmaker reference dataset into an executable Polymarket edge or a validated Combo probability. Player props, injuries, fees, slippage, bookmaker restrictions, and fill availability remain outside the audit.

If historical quotes cannot be sourced, the scientifically correct result is **not yet validated**. Do not fabricate a historical market feed or backfill pregame odds with postgame values.
