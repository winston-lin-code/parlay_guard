# Parlay Guard: save player models and generate pregame projections

These files **do not overwrite** `polymarket_nba_ev.py`.

1. Save `player_prop_common.py`, `train_player_props.py` and `predict_current_props.py` in the project root.
2. Install missing packages if needed: `python -m pip install pandas numpy scikit-learn joblib pyarrow`
3. Run `python train_player_props.py` to keep out-of-sample 2025–26 predictions **and** save production models trained through 2025–26.
4. Run a **fresh** `python polymarket_nba_ev.py --pages 2 --max-games 5` to collect upcoming markets.
5. Check feature parity: `python predict_current_props.py --audit-only`.
6. If audit passes: `python predict_current_props.py`.
7. Research only: rerun `python polymarket_nba_ev.py --pages 2 --max-games 5` to incorporate `data/current_projections.csv`.

The inference script derives next-game rolling features from completed games in `data/processed/player_training_data_v2.csv`. It stops when features do **not** match pregame feature values in your historical training dataset. This is intentional: the original feature-generation source was not provided, so exact parity cannot be assumed.

## Required current data

The player training CSV must contain **current completed games from the 2026–27 NBA season**. Data through 2025–26 alone cannot generate genuinely fresh current predictions. The script skips players with fewer than three completed current-season games, stale data (>7 days from upcoming matchup), unknown teams, and ambiguous player names.

The event slug `nba-gsw-mem-2026-11-19` is treated as **Warriors away, Grizzlies home**. This convention requires independent verification against an NBA schedule before treating the `IS_HOME` feature as correct.

`player_availability_verified=False` for all outputs. The regressors model **conditional production assuming a player appears** and do not correct for injuries, minutes restrictions, or markets that resolve NO when a player sits. It is unsafe to interpret the resulting EV as validated without those controls and further calibration/backtests.

## If feature parity audit fails

Provide the code used to create `player_training_data_v2.csv` (particularly shifted rolling averages, standard deviations, season averages, and `REST_DAYS`). The audit prints example mismatches so we can match the exact formulas instead of silently creating incompatible inputs.
