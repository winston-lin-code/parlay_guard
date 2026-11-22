# Parlay Guard prospective moneyline paper experiment

## Locked at installation

- **Every game baseline**: selects the Elo-predicted winner for every game with an eligible fresh pretipoff quote, *including negative estimated EV*. One $1 paper stake per game.
- **EV ≥ 5%**: uses the **identical frozen game, team, quote, and observation time** as the all-game baseline, but takes only estimated EV ≥ 0.05.
- **EV ≥ 10%**: same, with estimated EV ≥ 0.10.
- **Winner confidence ≥ 70%**: same, with Elo prediction ≥ 0.70, ignoring EV.

Rules, thresholds, observation window, stake, and UTC start date are locked once in `data/paper_trades/moneyline_tracker/frozen_strategy_v1.json`, with a SHA256 checksum. A rerun will not reset the start date.

## Quote policy

Source: `src.auto_live` live `moneylines` dictionary, by predicted winner, using Polymarket binary-contract **best ask** and Elo `home_probability`. One snapshot is frozen at most once per exact game date + both teams, 0–3 hours before tipoff, when model forecast is under 6h old and source snapshot under 20 minutes old. No backfills or fabricated asks. Missing prediction, missing moneyline ask, missing exchange market or expired game stays **unrecorded**; games surfaced by the live API may be listed with an unavailable reason in `moneyline_tracking.unavailable_live_games` until prices appear. The script NEVER submits an order.

To cover more games, configure the server with `PG_MAX_GAMES=20`; the underlying Polymarket market scanner may still omit games that have no market. The denominator is **eligible observed games**, not guaranteed all NBA games. Frozen files are append-only/exclusive; correcting errors requires a versioned new experiment and preserving this one's history.

## Settlement

The tracker reads only existing local completed NBA team games from `data/nba_YYYY_YY_latest_raw.csv`. It requires exactly one validated NBA game with both teams, a date match and consistent PTS/WL. An ESPN schedule ID is preserved as source metadata, not falsely treated as an NBA game ID. For numeric official NBA IDs, the ID must match. Incomplete or missing results remain `PENDING`. Settlement files are separate, created once; a conflicting later result raises a clear error instead of revising settled bets silently. Refresh the team's current season source from your existing `predict_today` workflow (or download current LeagueGameFinder team logs) if no score is cached.

## Returns and comparisons

Each paper stake = **$1 in cost**, using a YES binary contract with ask price `a`:

- win: `1/a - 1` units net
- loss: `-1` unit
- selected bets with pending outcomes are excluded from ROI denominator and reported separately
- ROI = sum(net units) / number of settled $1 bets
- maximum drawdown computed from chronological settled games, **not** a maximum possible bankroll loss

This models idealized best-ask execution. **It does not verify exchange fills, available depth for a full stake, fees, slippage, or whether a user could execute at the quoted price.** Polymarket market settlement rules and real-world execution differ; no real money trade is made or authorized.

## Commands (PowerShell, from Parlay Guard root)

```powershell
python install_moneyline_paper_tracker.py
python -m pytest tests/test_moneyline_paper_track.py -q
$env:PG_MAX_GAMES="20"
python -m uvicorn src.api:app --reload --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000/moneyline-tracker` for a read-only comparison. Periodic capture/settlement happens **only while your local server is running and refreshes succeed**. To request a refresh:

```powershell
Invoke-RestMethod -Method Post http://127.0.0.1:8000/api/live/refresh
```

To inspect outside the server:

```powershell
python -m src.moneyline_paper_track report
python -m src.moneyline_paper_track settle
# If local NBA result logs have not updated, explicitly refresh via NBA Stats:
python -m src.moneyline_paper_track refresh-results
```

To check installation without exposing any personal API credentials:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/api/moneyline-tracker | ConvertTo-Json -Depth 6
```

**Research disclosure:** Because this experiment began after the older moneyline return analyses, it may be used as genuinely prospective tracking. A handful of weeks still do not establish positive expected value. Compare all strategy policies by ROI, count, confidence intervals, drawdown, and missing-game coverage over a pre-agreed evaluation horizon; don't change thresholds in response to early outcomes.
