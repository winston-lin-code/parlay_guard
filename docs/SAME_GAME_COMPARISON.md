# Parlay Guard — frozen same-game moneyline vs player props comparison

This is an **additional** prospective PAPER-ONLY experiment. It does not change the existing moneyline tracker, its frozen rules or its records.

## Four frozen strategies — equal per-game budgets

- **Moneyline only:** always select the Elo-predicted game winner for a 1-unit paper stake, provided a fresh observable best ask exists.
- **Prop only:** one 1-unit paper single, choosing the top *positive stressed-EV* existing shortlisted player prop by stressed EV, from **either** NBA team. Skip when no valid player prop exists.
- **Best positive estimated EV:** a 1-unit single on whichever of the captured moneyline and best screened prop has higher raw model EV (`model_probability / yes_ask - 1`), only when that EV is greater than zero. This comparison is sensitive to unequal model calibration and is NOT a proven edge.
- **Both singles:** if the prop is available, split 1 total paper unit into a 0.5-unit moneyline single and a 0.5-unit prop single. **Not a parlay.** Skip when no prop qualifies.

## Capture and interpretation

At the **first valid live refresh within 1 hour before tipoff**, capture one frozen game decision from the same live snapshot. Forecast must be at most 6 hours old; the live snapshot and prop quote must be no older than 20 minutes. `src.auto_live` originally kept only selected props from the predicted winning team. This add-on leaves that list unchanged while exposing another field with all of the existing quality-screened props, regardless of team.

Prospective capture starts only at installation time. Earlier games are never backfilled. One record per matchup is created with exclusive file creation; repeated refreshes cannot revise the pick. The compare price is **an observed best ask, not a verified fill**. Neither model's price-relative EV has been validated for live profitability. No fees/slippage or real trades.

A comparison game **must** have a moneyline quote and pregame Elo. Player prop availability is optional. The prop universe only includes picks that passed the project's existing stressed-EV shortlist; it does not include all possible player props and cannot distinguish missing data from no profitable opportunities. The report shows both **all captured games** and **paired games where both single-contract asks were available**, to separate selection coverage from same-game relative performance. That paired cohort itself depends on the existing prop-screening model.

Player props require a uniquely mapped `PLAYER_ID` from `data/current_player_features.csv`, a half-point line, a supported stat (`PTS`, `REB`, `AST`, `FG3M`), valid observed asks and quote timestamps. They are settled against exactly matched official NBA `GAME_ID` and `PLAYER_ID` box-score rows. Absent/DNP player results stay pending—not assumed under wins. Official NBA team results settle moneyline legs separately even when player logs have not arrived; settlement sidecars may fill a pending player result, but cannot change a final outcome without an error.

Results are stored in `data/paper_trades/same_game_v1/`. Existing `data/paper_trades/moneyline_tracker/` remains untouched. Each strategy may risk at most **one paper unit per game**. Portfolio ROI is net paper return divided by settled stake, excluding still-pending strategy-games; the drawdown is from chronological settled paper returns, not an actual account balance.

## Commands

Extract into your existing Parlay Guard project **root** (where `src/api.py` already exists), then:

```powershell
python install_same_game_compare.py
python -m pytest tests/test_same_game_paper_track.py -q
$env:PG_MAX_GAMES="20"
python -m uvicorn src.api:app --reload --host 127.0.0.1 --port 8000
```

View: `http://127.0.0.1:8000/same-game-comparison` and `http://127.0.0.1:8000/api/same-game-comparison`.

Run after games finish and the new day's completed logs become available:

```powershell
python -m src.same_game_paper_track refresh-results
```

If NBA Stats/player requests fail, update existing NBA result caches using the app's established workflows and retry `python -m src.same_game_paper_track settle`. If your main server is shut down there is **no scheduled cloud capture**; run the local server for the experiment to collect games. Do not change frozen thresholds based on early results.
