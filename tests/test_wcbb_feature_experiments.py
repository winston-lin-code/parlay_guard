import json

import numpy as np
import pandas as pd

from src import train_wcbb_anytime, wcbb_feature_experiments as experiment
from src.wcbb_all_features import AllFeatures, SEASON


def make_game(game_id, season, tip, win):
    home = 4 if win else 3
    away = 3 if win else 4
    schedule = dict(zip(experiment.SCHEDULE,
                        (game_id, season, tip.isoformat(), 101, 102, home, away, True, "post", 4,
                         game_id % 3 == 0, 91 if game_id % 3 else 99, 91)))
    plays = []
    for period, clock, points in [
        (1, "10:00", (0, 0)), (1, "5:00", (1, 0)), (1, "0:00", (1, 1)),
        (2, "10:00", (1, 2)), (2, "4:00", (2, 2)), (2, "0:00", (2, 2)),
        (3, "10:00", (2, 2)), (3, "0:00", (2, 3)),
        (4, "10:00", (2, 3)), (4, "0:00", (home, away)),
    ]:
        step = len(plays) + 1
        wallclock = tip + pd.Timedelta(minutes=10*step)
        plays.append(dict(zip(experiment.PLAYS,
                              (game_id, 100*game_id+step, step, period, clock, *points, wallclock.isoformat(), 101, 102))))
    boxes = [dict(zip(experiment.BOX, (game_id, team, score, 55, 20, 10, 15)))
             for team, score in ((101, home), (102, away))]
    return schedule, plays, boxes


def test_as_of_clock_does_not_read_same_clock_or_next_quarter_state():
    day = pd.Timestamp("2019-01-01T20:00:00Z")
    schedule, plays, boxes = make_game(1, 2019, day, True)
    counts = experiment.Counter()
    plan = experiment.season_plans(pd.DataFrame([schedule]), 2019, {101, 102}, counts)
    games, grids = experiment.snapshots_for_season(pd.DataFrame(plays), plan,
        experiment.box_possessions(pd.DataFrame(boxes), plan, counts), counts)
    assert len(games) == 1
    grid = grids[0].set_index("seconds_remaining")
    assert grid.loc[2100, "margin"] == 0
    assert grid.loc[1800, "margin"] == 0
    assert grid.loc[1800, "source_period"] == 1
    assert grid.loc[1740, "margin"] == -1
    assert games[0]["available_at"] > day + pd.Timedelta(hours=24)


def test_pregame_features_wait_for_final_wallclock_and_embargo():
    start = pd.Timestamp("2019-01-01T19:00:00Z")
    early_game = {"game_id": 1, "season": 2019, "home_id": 1, "away_id": 2, "tip": start,
                  "available_at": start+pd.Timedelta(hours=26), "possessions": 60,
                  "home_final": 72, "away_final": 60, "home_win": 1}
    waiting = {**early_game, "game_id": 2, "tip": start+pd.Timedelta(hours=25), "available_at": pd.NaT}
    later = {**waiting, "game_id": 3, "tip": start+pd.Timedelta(hours=27)}
    rows = experiment.pregame_history([early_game, waiting, later]).set_index("game_id")
    assert rows.loc[2, "elo_diff"] == 0 and np.isnan(rows.loc[2, "home_offense"])
    assert rows.loc[3, "elo_diff"] == 20
    assert rows.loc[3, "home_offense"] == 120 and rows.loc[3, "home_defense"] == 100
    assert rows.loc[3, "home_sos"] == 1500


def test_cli_builds_aligned_ranked_report_and_reserves_new_season(tmp_path):
    sources, destination = tmp_path / "input", tmp_path / "results"
    sources.mkdir()
    members, baseline = [], []
    for year in (*experiment.SEASONS, 2027):
        schedules, plays, boxes = [], [], []
        for i in range(8):
            game_id = year * 10 + i
            tip = pd.Timestamp(f"{year}-01-02T19:00:00Z") + pd.Timedelta(days=i*2)
            schedule, pbp, team_box = make_game(game_id, year, tip, i % 2 == 0)
            schedules.append(schedule)
            plays.extend(pbp)
            boxes.extend(team_box)
            if year == 2026:
                baseline.append({"game_id": game_id, "season": year, "seconds_remaining": 1200, "probability": .6})
        if year == 2026:
            schedules.append(schedules[0].copy())
            schedules[-2]["status_type_completed"] = False
            plays[-21]["home_score"] = 99
        for stem, rows in (("wbb_schedule", schedules), ("team_box", boxes), ("play_by_play", plays)):
            pd.DataFrame(rows).to_csv(sources / f"{stem}_{year}.csv", index=False)
        members.extend({"season": year, "team_id": team} for team in (101, 102))
    pd.DataFrame(members).to_csv(sources / "division_i_teams.csv", index=False)
    baseline_path = sources / "original.csv"
    pd.DataFrame(baseline).to_csv(baseline_path, index=False)
    original = baseline_path.read_bytes()
    assert experiment.main(["--data", str(sources), "--output", str(destination),
                            "--baseline-predictions", str(baseline_path), "--bootstrap", "100"]) == 0
    run, = destination.iterdir()
    assert baseline_path.read_bytes() == original
    ranked = pd.read_csv(run / "ranked_comparison.csv")
    assert len(ranked) == 2 * len(experiment.FEATURES)
    assert ranked.log_loss.is_monotonic_increasing
    assert ranked.games.nunique() == 1
    folds = pd.read_csv(run / "folds.csv")
    assert list(folds.test_season) == [2021, 2022, 2023, 2024, 2025, 2026]
    assert (folds.train_last_season < folds.test_season).all()
    assert folds.iloc[-1].status == "examined_development"
    assert json.loads((run / "config.json").read_text())["prospective_season"] == 2027
    comparison = pd.read_csv(run / "paired_game_bootstrap.csv")
    assert comparison.games.nunique() == 1
    assert ((comparison.ci_low <= comparison.difference) & (comparison.difference <= comparison.ci_high)).all()
    audit = pd.read_csv(run / "data_audit.csv")
    assert audit.query("season == 2026 and reason == 'incomplete_schedule_rows'")["count"].item() == 1


def test_missing_inputs_leave_unique_audit_directories(tmp_path):
    for _ in range(2):
        assert experiment.main(["--data", str(tmp_path / "absent"), "--output", str(tmp_path)]) == 2
    runs = list(tmp_path.iterdir())
    assert len(runs) == 2
    for run in runs:
        assert (run / "missing_inputs.txt").is_file()
        assert "Historical ranking unavailable" in (run / "ranked_comparison.md").read_text()


def test_live_artifact_fits_historical_years_and_releases_current_priors(tmp_path):
    seasons = (*experiment.SEASONS, SEASON)
    pd.DataFrame([(year, team) for year in seasons for team in (101, 102)],
                 columns=["season", "team_id"]).to_csv(tmp_path / "division_i_teams.csv", index=False)
    for year in seasons:
        start = pd.Timestamp(f"{year - 1}-09-01T20:00:00Z")
        fixtures = [make_game(year * 10 + index, year, start + pd.Timedelta(days=3 * index), bool(index))
                    for index in range(2)]
        for position, name in enumerate(("wbb_schedule", "play_by_play", "team_box")):
            records = [row for fixture in fixtures for row in
                       ([fixture[position]] if position == 0 else fixture[position])]
            pd.DataFrame(records).to_csv(tmp_path / f"{name}_{year}.csv", index=False)
    artifact = train_wcbb_anytime.build(tmp_path, threads=1)
    assert artifact["trained_seasons"] == list(experiment.SEASONS)
    assert artifact["training_games"] == 2 * len(experiment.SEASONS)
    model = AllFeatures(artifact)
    released = pd.Timestamp(artifact["context"]["101"][0][0], unit="s", tz="UTC")
    assert model.team("101", released)[1:] == (None, None, None)
    assert all(value is not None for value in model.team("101", released + pd.Timedelta(seconds=1))[1:])
