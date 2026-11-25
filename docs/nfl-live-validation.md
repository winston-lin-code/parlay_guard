# NFL live EV and quarterback changes

The live model uses the score and regulation clock. It trained on 543 regular-season games from 2022–23, calibrated on 272 games from 2024, and tested on 272 games from 2025. It estimates remaining scoring margin and total using [nflverse public play-by-play](https://github.com/nflverse/nflverse-data/releases/tag/pbp) with one observation per game per three-minute clock interval. Final scores are outcomes only. The baseline carries the current margin forward with a proportional home adjustment and adds league-average scoring proportional to time remaining. Both models use distributions fitted on the calibration season. Only market and clock groups that beat this baseline on both probability metrics enter the scanner. Archived spread and total lines were used as evaluation thresholds. A historical test at executable in-game prices is still needed.

Moneylines passed in Q1, Q3, and Q4. Totals passed in Q1, Q2, and Q3. No spread group passed both checks.

Live EV requires a matching regular-season scoreboard, fresh books, and a repeat scoreboard check. The model skips the final two minutes and overtime. It has no possession or injury features.

## Starting quarterback change audit

The [nflverse schedule data](https://github.com/nflverse/nfldata/blob/master/data/games.csv) includes the actual starting quarterback and an archived spread for each game. In 1,707 NFL regular-season games from 2016–2025, both teams had an established starter, defined as the same quarterback starting at least three of the last four games. Exactly one team changed starters in 376 of those games.

| Outcome for the team that changed quarterback | Adjusted association | Approximate 95% interval |
|---|---:|---:|
| Actual scoring margin | −4.35 points | −5.95 to −2.75 |
| Archived spread expressed as expected margin | −2.78 points | −3.23 to −2.33 |
| Scoring margin relative to archived spread | −1.57 points | −3.13 to −0.02 |

The regression accounts for team-season, week, home field, opponent, and recent scoring margin. It uses HC3 standard errors. Changes include injuries, benchings, rest, and returns. The data does not identify the reason or when the change became public. These are historical associations. The residual against the spread does not measure betting ROI or in-game injury effects.

```sh
python -m src.audit_nfl_qb_changes path/to/games.csv
python -m src.nfl_inplay_model path/to/play_by_play_202{2,3,4,5}.parquet
```

Install PyArrow in the training environment to read the play-by-play files.

## NBA lineup audit

I replayed 5,373 regular-season games from the 2021–22 through 2025–26 seasons using [ESPN player box scores distributed by SportsDataverse](https://py.sportsdataverse.org/docs/nba/reference/loaders/player). A regular starter started at least seven of the team's prior 10 games and averaged at least 28 minutes across those games. There were 203 games with a regular starter out and an explicit injury-related reason in the box score.

| Change per regular starter | Adjusted scoring margin | Approximate 95% interval |
|---|---:|---:|
| Out with an explicit injury-related reason | −1.70 points | −3.55 to +0.14 |
| Absent for another or unknown reason | −1.99 points | −2.57 to −1.40 |
| Played off the bench | −0.43 points | −1.62 to +0.75 |

The regression accounts for both teams' starter changes, team-season effects and recent scoring margin. The injury category uses explicit box-score descriptions such as sprain, fracture or surgery. Other absences can include injuries with missing or unrecognized reasons. The results describe association with final margins and do not measure event-time market moves. None of these findings is used as an automatic EV adjustment.

```sh
python -m src.audit_nba_lineups path/to/player_box_202{2,3,4,5,6}.parquet
```
