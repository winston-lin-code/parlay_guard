# WCBB historical feature experiments

The research runner reads separate historical inputs and produces comparison
reports. Existing backtests and model artifacts are read-only. The 2025–26 fold
is an examined development benchmark. Reserve the season ending in 2027 for
prospective evaluation.

## Inputs

Put these WeHoop files for each season ending in 2019 through 2026 in the same
directory. The runner accepts the release Parquet files or CSVs with the same
column names.

| File | Source |
|---|---|
| `wbb_schedule_{season}.parquet` | [WeHoop schedules](https://github.com/sportsdataverse/sportsdataverse-data/releases/tag/espn_womens_college_basketball_schedules) |
| `play_by_play_{season}.parquet` | [WeHoop play-by-play](https://github.com/sportsdataverse/sportsdataverse-data/releases/tag/espn_womens_college_basketball_pbp) |
| `team_box_{season}.parquet` | [WeHoop team box scores](https://github.com/sportsdataverse/sportsdataverse-data/releases/tag/espn_womens_college_basketball_team_boxscores) |
| `division_i_teams.csv` | Season-specific Division I membership with columns `season,team_id` using ESPN IDs from a verified crosswalk. [SportsDataverse reference data](https://github.com/sportsdataverse/sdv-reference-data) |

Save any archived baseline predictions with `game_id,season,seconds_remaining,probability`.
Probability represents a home win. The runner measures these on the intersection
with the validation grid and leaves the original file untouched.

```powershell
python -m src.wcbb_feature_experiments --data .\data\wcbb-history
```

Pass `--baseline-predictions .\data\wcbb-history\original_predictions.csv` for a
paired-cohort overlap table. Parquet input requires `pyarrow`.

## Method and output

The audit checks schedule completion, duplicate game IDs and timestamped
fixtures, Division I membership, play-by-play completeness, opening and final
scores, team identity, clocks, and team box scores. All models see the same game
IDs and game-clock snapshots. A snapshot uses the score before the clock mark.
At quarter boundaries it includes the preceding quarter's final zero-clock plays.
A game with a conflicting sequence or clock regression is excluded.

Historical team and opponent features update from verified earlier final plays
after a 24-hour embargo. The embargo allows time for results and box scores to
appear. Archive wall-clock time records a play's occurrence. Historical source
receipt time is unavailable from these files. Incomplete timestamps prevent a
game from contributing to the pregame history. Offensive and defensive
efficiency use points per 100 estimated possessions within a season.
Strength of schedule averages the pregame Elo of the opponents already faced.
The home-court feature checks the designated home team's recorded home venue.
Neutral-site status is a separate input.

Each fold trains on seasons that precede it. The main table ranks models by
pooled log loss over the fixed times with Brier score as a tie-breaker. Season
and time tables include accuracy, log loss, Brier score, ROC AUC, and calibration.
First qualifying picks are evaluated once per game on the shared one-minute
grid. Paired bootstrap intervals resample whole games within validation seasons
for differences in accuracy, Brier score, and log loss. A negative difference
favors the candidate for loss metrics.

Each run writes to a new subdirectory under `results/wcbb-feature-experiments`.
Key outputs include `ranked_comparison.md`, `folds.csv`, `data_audit.csv`,
`fixed_time_metrics.csv`, `calibration.csv`, `first_qualifying_summary.csv`,
and `paired_game_bootstrap.csv`. The run also exports first-entry predictions
and out-of-fold probabilities.
