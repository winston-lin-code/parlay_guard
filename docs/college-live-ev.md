# College live EV

Parlay Guard updates screened EV from the verified score and regulation clock, available order-book depth, and the applicable fee schedule. It polls each league independently on a 15-second target cadence and rechecks the scoreboard after fetching quotes. It requires a unique schedule match using both team names and a scheduled start within 30 minutes. Missing or stale inputs and inconsistent state suppress picks. The live model uses score and clock, with no live possession, lineup, or injury inputs.

## Coverage

| League | Moneylines | Half-point spreads | Half-point totals |
|---|---|---|---|
| Men's Division I basketball | First half | Both halves | Both halves |
| FBS football | First and second quarters | — | Second through fourth quarters |

Coverage is regular season and ends with two minutes left in regulation. Markets outside the tested line range and overtime stay out of the results. Two-sided full-game contracts only.

## Holdout

| League | Training season and games | Calibration season and games | Holdout season and games |
|---|---|---|---|
| Men's Division I basketball | 2023–24 · 6,031 | 2024–25 · 6,008 | 2025–26 · 6,170 |
| FBS football | 2023 · 745 | 2024 · 737 | 2025 · 762 |

Training uses ESPN play-by-play from SportsDataverse with one post-play snapshot per game, period, and three-minute clock bin. Completed-game schedules supply final score labels. Only market/period combinations that improve both Brier score and log loss over the score-and-clock baseline are enabled. The baseline keeps the current score and adds league-average remaining points proportional to clock time. FBS tests use archived full-game half-point lines. Basketball tests use simulated half-point lines centered on the baseline. No archived live prices or simulated fills enter those results.

Download the three respective seasons of [basketball play-by-play](https://py.sportsdataverse.org/docs/mbb/reference/loaders/other) or [football play-by-play](https://py.sportsdataverse.org/docs/cfb/reference/loaders/pbp) and the matching schedules used by the pregame trainer. Install PyArrow in the training environment.

```sh
python -m src.college_inplay_model mbb --plays mbb-play-202{4,5,6}.parquet --schedules mbb-schedule-202{4,5,6}.parquet
python -m src.college_inplay_model cfb --plays cfb-play-202{3,4,5}.parquet --schedules cfb-schedule-202{3,4,5}.parquet
```

The artifacts under `models/college_guard/` contain the fitted distributions and holdout metrics.
