# Professional pregame EV models

The scanner uses separate score models for NFL, NHL, and MLB regular-season games. Each model uses team Elo, recent scoring and defense, rest, and venue. The models fit score margin and total independently. Residual distributions are estimated from a separate calibration season. Training features use completed games from earlier dates.

| League | Completed holdout games | Enabled markets | Results through |
|---|---:|---|---|
| NFL | 272 | Moneyline, half-point spreads and totals | November 23, 2026 |
| NHL | 1,312 | Moneyline and half-point puck lines | November 24, 2026 |
| MLB | 2,430 | Moneyline, half-point run lines and totals | September 27, 2026 |

The 2024 season (2024–25 for NHL) supplies calibration residuals. The 2025 season (2025–26 for NHL) supplies holdout results. A market must beat a league-average score baseline on both Brier score and log loss. NHL totals failed that gate. NFL spread and total metrics use available archived half-point lines. NHL and MLB use simulated half-point lines. The evaluation does not simulate fills or establish a profitable strategy. The models do not use starters, injuries, or pitcher and goalie assignments.

The pregame scanner checks the league schedule and requires a unique fixture match before pricing. It uses fresh order-book depth and the existing fee and probability stress adjustments. The artifact must contain recent results within 14 days, so the bundled MLB snapshot remains inactive during its offseason and needs a refresh for the next regular season. NFL moneyline estimates incorporate the half payout when the contract explicitly specifies that settlement for ties.

## Rebuild the snapshots

- NFL uses the [nflverse games CSV](https://github.com/nflverse/nfldata/blob/master/data/games.csv).
- NHL uses one [NHL club schedule response](https://api-web.nhle.com/v1/club-schedule-season/BOS/20252026) per club and season. Game IDs are deduplicated during training.
- MLB uses one [MLB schedule response](https://statsapi.mlb.com/api/v1/schedule?sportId=1&season=2025&gameType=R) per regular season.

```sh
python -m src.pro_score_calibration nfl path/to/games.csv
python -m src.pro_score_calibration nhl path/to/nhl-2*.json
python -m src.pro_score_calibration mlb path/to/mlb-2*.json
```

Keep the original historical seasons in each input set and include recent regular-season results. Commit the regenerated files under `models/pro_scores/` to update the scanner.
