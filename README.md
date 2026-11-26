# Parlay Guard — NBA probability research MVP

An end-to-end Python portfolio project that converts NBA historical game logs into **pregame** features, trains and evaluates a baseline NBA winner model, and compares estimated probabilities with **manually entered** prediction-market prices. It does **not** place trades or guarantee an edge.

## Project layout

```text
parlay_guard/
├── data/                       # Local CSVs (gitignored)
├── models/                     # Saved model and evaluation (model ignored)
├── src/
│   ├── collect.py              # Optional NBA API collection
│   ├── clean.py                # Home/away and outcome validation
│   ├── merge_games.py          # One row per NBA game
│   ├── feature_engineering.py  # Historical rolling form and rest
│   ├── merge_features.py      # Home/away model features
│   ├── train.py               # Chronological evaluation and calibration
│   ├── scan.py                # Manual-price research scanner
│   └── pipeline.py            # Runs offline stages
├── tests/test_features.py
├── requirements.txt
└── README.md
```

## Windows PowerShell setup

```powershell
cd C:\Users\Winston\projects\polyPredict
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

If you extract this ZIP into a `parlay_guard` subfolder, run commands **inside that folder** instead; or copy the ZIP contents directly into your existing `polyPredict` project.

### Existing historical CSV

Place your existing `nba_all_seasons.csv` in `data/` (9,840 team-game rows from your previous work). Then:

```powershell
python -m src.pipeline
pytest -q
```

The pipeline produces:

- `data/nba_games_updated.csv`
- `data/nba_games_merged.csv`
- `data/nba_team_features.csv`
- `data/nba_model_dataset.csv`
- `models/nba_home_win.joblib`
- `models/evaluation.json`

Alternatively, if you already have cleaned/merged CSVs, you can run individual modules:

```powershell
python -m src.feature_engineering
python -m src.merge_features
python -m src.train
```

### Optional: fetch NBA game logs

```powershell
python -m src.collect --seasons 2022-23 2023-24 2024-25 2025-26
```

This replaces `data/nba_all_seasons.csv`. NBA Stats may rate-limit requests; using your existing dataset is preferable. The current 2025–26 season might not match the exact snapshot you collected previously.

## Manual market-price comparison

Example **illustrative** quote; NOT a current Polymarket market:

```powershell
python -m src.scan --home BOS --away NYK --date 2026-11-20 --home-price 0.55 --away-price 0.47
```

Prices are fractions of $1 (`0.55` means 55¢). The scanner outputs model probabilities, probability differences, and `NO BET` or `PAPER-REVIEW`. It intentionally does not interpret raw differences as guaranteed profit, and it does not trade. Prices for opposite outcomes can differ from 1 minus the other quote due to spreads, fees, and market structure.

**Important:** This scanner uses the historical snapshot on disk. It does not fetch fresh NBA games, injuries, lineups, live markets, or market-specific settlement rules. For dates after the snapshot, recent form becomes stale. Always verify data freshness before considering even a paper-trade comparison.

## Modeling decisions and limitations

- Each game's feature values are calculated from **previous** team games (`shift(1)`), not the game's own result.
- 5- and 10-game win rates and days of rest are the initial features. Rolling features carry across seasons, which may be undesirable after major roster changes.
- Missing values for the first 5/10 team games are expected; the training pipeline imputes them using training-set medians and missingness indicators.
- Chronological train/validation/test split prevents training on later games to predict earlier games. The validation period fits a Platt calibrator; the final test period remains unseen during fitting.
- The **baseline** is deliberately simple and may be weak or miscalibrated; check test Brier score and log loss before interpreting any price differences.
- There is no live Polymarket API integration or order execution. `PAPER-REVIEW` means *inspect further*, not a recommendation to wager.
- Before production: add actual scheduled games, lineups/injuries with timestamped availability, more seasons, closing odds, vig/fees/slippage, market liquidity, walk-forward backtests, and monitored data updates.
- Joblib/pickle model files should only be loaded from trusted sources.

## Suggested next engineering improvements

1. Add team offensive/defensive efficiency and strength of schedule using strictly pregame rolling values.
2. Add timestamped injury/lineup data and enforce feature availability at prediction time.
3. Add automated market-data ingestion with market-ID mapping and price timestamps.
4. Add walk-forward evaluation and paper-trading logs including fees and slippage.
5. Expose predictions through a FastAPI backend and a simple frontend.

## Halftime rule

The Live EV Scanner's **Halftime rule** tab ranks regular-season full-game
moneylines in NBA, NFL, FBS football, men's and women's Division I college
basketball, and WNBA. It shows one qualifying side per game using a $2 paper
stake, 60% minimum model probability and +5% minimum screened EV. Other live
games remain visible with their model status.

NBA and NFL use the last five game-clock minutes of the first half, halftime,
and the first five game-clock minutes of the second half. Their in-play models
use the current score and time remaining. The other supported leagues use
the confirmed halftime break.

The added basketball models use verified ESPN halftime scores and venue status.
They were trained with 2024 and 2025 SportsDataverse schedule data and evaluated
on 2026 games. The holdouts evaluate win probabilities. They do not establish
trading returns for the added leagues.

### ITF singles set-break models

The live board uses separate men's and women's fits based on completed-set game
margins. The models use 2022–23 main-draw singles results, with 2024 held out.
The artifact records Brier score and log loss against the score-only baseline
after each completed set. Within-set updates use the live score rules with an
odds adjustment from the most recent trained set break. The holdout measures
set-break accuracy only. First sets and doubles display score-only estimates.

The inputs are Jeff Sackmann's ATP Futures and WTA qualifying/ITF CSVs, as
preserved in the [archived dataset](https://github.com/micahp/tennis-point-by-point-archive/tree/83733587353df8a41f2fd4f516147d5aa83f5a8d)
under CC BY-NC-SA 4.0. Download the 2022–24 files
`atp/atp_matches_futures_YEAR.csv` and `wta/wta_matches_qual_itf_YEAR.csv`.
Run the following with those six files in the working directory.

```sh
python -m src.train_tennis_live --men atp_matches_futures_202{2,3,4}.csv \
  --women wta_matches_qual_itf_202{2,3,4}.csv --holdout-year 2024
```

### ITF pre-match Elo

The pre-match model rates men's and women's main-draw singles independently.
Each player starts at 1500. It updates overall and surface Elo from completed
matches and blends the two ratings at prediction time. The historical CSVs
give tournament start dates. The replay waits 14 days after that date before
applying results, and batches same-date updates so the order of the CSV rows
does not affect predictions.

The model uses 2022 for warm-up, 2023–24 to select the update rate and surface
weight by log loss, and 2025 for a walk-forward holdout with settings fixed.
The saved player ratings incorporate additional 2026 matches present in the
archive. They are intended for predictions after the artifact's data cutoff.
This model covers pre-match singles probabilities.

Use the same Jeff Sackmann archive identified above, with the 2022–26 ATP
Futures and WTA qualifying/ITF files. Train and query the model with these
commands.

```sh
python -m src.train_tennis_elo --men atp_matches_futures_202{2,3,4,5,6}.csv \
  --women wta_matches_qual_itf_202{2,3,4,5,6}.csv --holdout-year 2025
python -m src.tennis_elo ITFME "Robert Strombachs" "Yash Yadav" --surface Hard
```

### ITF live Elo and first-set break

The live singles estimate combines overall Elo and completed-set game
margins when the live feed identifies both score columns and both players
resolve to ratings. The historical set-break model supplies the fallback.
The fit uses 2023–24 match-start ratings and scores, with 2025 held out.
Current point and game scores adjust the estimate within a set. The holdout
measures match start and set breaks only.

The Halftime view marks a late first set using the score and marks the five
minutes following a recently observed transition into set two. Tennis has
no countdown to set end, so the late-set label uses score state. These
estimates appear in the live coverage list.

To refit with the same historical inputs:

```sh
python -m src.train_tennis_halftime --men atp_matches_futures_202{2,3,4,5,6}.csv \
  --women wta_matches_qual_itf_202{2,3,4,5,6}.csv
```

### Local ITF singles wager runner

Install the project requirements and start in paper mode to verify your feeds.
The runner needs both a fresh sports WebSocket score with identifiable sides
and a matching Polymarket US ITF singles moneyline. It prices the late first
set or the first five minutes after an observed set-two transition. It requires
the Elo-adjusted forecast, 60% minimum model probability, and +5% minimum
screened EV using the capped limit price and estimated fees.

```powershell
python -m pip install -r requirements.txt
python -m src.tennis_wager --stake 2
```

Real-order mode requires local Polymarket US credentials in
`POLYMARKET_KEY_ID` and `POLYMARKET_SECRET_KEY` and a UTC daily exposure cap.
The stake is a maximum all-in cost. The daily cap counts reserved exposure
including canceled or unresolved attempts. Each order uses a limit with
fill-or-kill, and the SQLite ledger records one attempt per event and mode.
Use the same ledger path across restarts. The runner stops on an ambiguous
submission response. Reconcile the exchange activity before trading again.

```powershell
python -m src.tennis_wager --live --stake 2 --daily-cap 10
```

Use Ctrl+C to stop. The default ledger is
`data/paper_trades/tennis_orders.sqlite3`. Startup can take a few score
updates, and a score stream with missing identities or stale data is skipped.
