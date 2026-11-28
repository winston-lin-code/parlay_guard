"""Offline WCBB feature ablations on shared game-clock snapshots."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import heapq
import json
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits


SEASONS = tuple(range(2019, 2027))
FIXED_TIMES = (2100, 1800, 1500, 1200, 900, 600, 300)
ENTRY_TIMES = tuple(range(2340, 0, -60))
PREGAME_DELAY = pd.Timedelta(hours=24)
SEED = 27
BENCHMARK = {"games": 37311, "snapshot_accuracy": .8386, "log_loss": .3397, "first_win_rate": .7744}
SCHEDULE = ("game_id", "season", "start_date", "home_id", "away_id", "home_score", "away_score",
            "status_type_completed", "status_type_state", "format_regulation_periods", "neutral_site",
            "venue_id", "home_venue_id")
PLAYS = ("game_id", "id", "game_play_number", "period_number", "clock_display_value", "home_score",
         "away_score", "wallclock", "home_team_id", "away_team_id")
BOX = ("game_id", "team_id", "team_score", "field_goals_attempted", "free_throws_attempted",
       "offensive_rebounds", "total_turnovers")
ELO = ("elo_diff",)
SCORE = ("margin", "seconds_remaining", "pace_adjusted_margin")
ADDITIONS = {
    "home_court": ("home_court",),
    "neutral_site": ("neutral_site",),
    "offense": ("home_offense", "away_offense"),
    "defense": ("home_defense", "away_defense"),
    "strength_of_schedule": ("home_sos", "away_sos"),
}
FEATURES = {"elo_only": ELO, "score_only": SCORE, "combined": ELO + SCORE}
FEATURES.update({f"combined_plus_{name}": ELO + SCORE + columns for name, columns in ADDITIONS.items()})
FEATURES["combined_plus_all"] = ELO + SCORE + tuple(col for group in ADDITIONS.values() for col in group)


def read_frame(path, columns):
    if path.suffix == ".parquet":
        return pd.read_parquet(path, columns=list(columns))
    return pd.read_csv(path, usecols=list(columns), low_memory=False)


def read_season(root, stem, season, columns):
    for suffix in (".parquet", ".csv"):
        path = root / f"{stem}_{season}{suffix}"
        if path.is_file():
            return read_frame(path, columns)
    raise FileNotFoundError(f"{stem}_{season}.parquet or .csv")


def timestamps(values):
    return pd.to_datetime(values, utc=True, format="mixed", errors="coerce")


def flag(value):
    if pd.isna(value):
        return np.nan
    return {"true": 1., "false": 0., "1": 1., "0": 0., "1.0": 1., "0.0": 0.}.get(str(value).lower(), np.nan)


def season_plans(schedule, season, members, counters):
    rows = schedule.loc[:, SCHEDULE].copy()
    counters["schedule_rows"] = len(rows)
    ids = pd.to_numeric(rows.game_id, errors="coerce")
    rows["game_id"] = ids.where((ids > 0) & (ids % 1 == 0)).astype("Int64")
    rows["tip"] = timestamps(rows.start_date)
    for col in ("home_id", "away_id", "home_score", "away_score"):
        rows[col] = pd.to_numeric(rows[col], errors="coerce")
    original = len(rows)
    rows = rows.drop_duplicates(subset=list(SCHEDULE))
    counters["duplicate_identical_schedule_rows_removed"] = original - len(rows)
    conflict = rows.game_id.notna() & rows.game_id.duplicated(keep=False)
    counters["conflicting_game_ids_excluded"] = rows.loc[conflict, "game_id"].nunique()
    rows = rows[~conflict]
    status = rows.status_type_completed.map(flag).eq(1) & rows.status_type_state.eq("post")
    counters["incomplete_schedule_rows"] = int((~status).sum())
    rows = rows[status]
    for col in ("home_id", "away_id", "home_score", "away_score"):
        rows = rows[rows[col].notna() & (rows[col] % 1 == 0)]
    counters["final_rows_with_missing_or_fractional_id_or_score"] = int(status.sum()) - len(rows)
    rows = rows.astype({col: "int64" for col in ("home_id", "away_id", "home_score", "away_score")})
    valid = (rows.game_id.notna() & rows.tip.notna() & (rows.home_id > 0) & (rows.away_id > 0)
             & (rows.home_id != rows.away_id) & (rows.home_score >= 0) & (rows.away_score >= 0)
             & (rows.home_score != rows.away_score)
             & pd.to_numeric(rows.season, errors="coerce").eq(season)
             & pd.to_numeric(rows.format_regulation_periods, errors="coerce").eq(4)
             & (rows.tip >= pd.Timestamp(f"{season-1}-07-01", tz="UTC"))
             & (rows.tip < pd.Timestamp(f"{season}-07-01", tz="UTC")))
    counters["invalid_completed_schedule_rows"] = int((~valid).sum())
    rows = rows[valid].copy()
    counters["division_i_teams_in_membership"] = len(members)
    counters["division_i_teams_in_schedule"] = len((set(rows.home_id) | set(rows.away_id)) & members)
    both = rows.home_id.isin(members) & rows.away_id.isin(members)
    counters["completed_games_without_two_confirmed_division_i_teams"] = int((~both).sum())
    rows = rows[both].copy()
    # A replacement ESPN ID for the same timestamp and pairing remains ambiguous.
    rows["pair_a"] = np.minimum(rows.home_id, rows.away_id)
    rows["pair_b"] = np.maximum(rows.home_id, rows.away_id)
    duplicates = rows.duplicated(["pair_a", "pair_b", "tip"], keep=False)
    counters["duplicate_fixture_ids_excluded"] = int(duplicates.sum())
    return rows[~duplicates].set_index("game_id", drop=False)


def box_possessions(boxes, plan, counters):
    result = {}
    stats = ["field_goals_attempted", "free_throws_attempted", "offensive_rebounds", "total_turnovers"]
    boxes = boxes.drop_duplicates()
    for game_id, group in boxes.groupby("game_id"):
        if game_id not in plan.index:
            continue
        target = plan.loc[game_id]
        by_team = group.set_index("team_id")
        if len(group) != 2 or set(by_team.index) != {target.home_id, target.away_id}:
            counters["conflicting_or_incomplete_team_box"] += 1
            continue
        aligned = by_team.loc[[target.home_id, target.away_id]]
        values = aligned[stats].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        scores = pd.to_numeric(aligned.team_score, errors="coerce").to_numpy(dtype=float)
        if not np.array_equal(scores, [target.home_score, target.away_score]):
            counters["team_box_final_score_mismatch"] += 1
            continue
        if not np.isfinite(values).all() or np.any(values < 0):
            counters["invalid_possession_inputs"] += 1
            continue
        attempts, free_throws, offensive_rebounds, turnovers = values.T
        possessions = float(np.mean(attempts + .44 * free_throws - offensive_rebounds + turnovers))
        if not 20 <= possessions <= 180:
            counters["invalid_possession_inputs"] += 1
            continue
        result[game_id] = possessions
    counters["games_with_valid_team_boxes"] = len(result)
    return result


def snapshots_for_season(plays, plan, boxes, counters):
    plays = plays[plays.game_id.isin(plan.index)].copy()
    counters["games_with_play_by_play"] = plays.game_id.nunique()
    original = len(plays)
    plays = plays.drop_duplicates()
    counters["duplicate_identical_play_rows_removed"] = original - len(plays)
    target = plays.clock_display_value.astype("string").str.extract(r"^(?:(\d{1,2}):)?(\d{1,2}(?:\.\d+)?)$")
    plays["clock"] = 60 * pd.to_numeric(target[0], errors="coerce").fillna(0) + pd.to_numeric(target[1], errors="coerce")
    plays["wallclock"] = timestamps(plays.wallclock)
    for col in ("home_score", "away_score", "period_number", "game_play_number"):
        plays[col] = pd.to_numeric(plays[col], errors="coerce")
    counters["games_without_play_by_play"] = len(plan) - counters["games_with_play_by_play"]
    games, snapshots = [], []
    times = np.asarray(ENTRY_TIMES, dtype=int)
    for game_id, group in plays.groupby("game_id", sort=False):
        event = plan.loc[game_id]
        if (group.home_team_id.dropna().ne(event.home_id).any()
                or group.away_team_id.dropna().ne(event.away_id).any()):
            counters["play_by_play_team_mismatch"] += 1
            continue
        if (group.id.dropna().duplicated().any() or group.game_play_number.isna().any()
                or group.game_play_number.duplicated().any()):
            counters["conflicting_play_id_or_sequence_excluded"] += 1
            continue
        values = group[["period_number", "clock", "home_score", "away_score"]].to_numpy(dtype=float)
        if (not np.isfinite(values).all() or np.any(values < 0)
                or np.any(values[:, [0, 2, 3]] % 1)
                or np.any(values[:, 0] < 1)
                or np.any(values[:, 1] > np.where(values[:, 0] <= 4, 600, 300))):
            counters["invalid_play_states_excluded"] += 1
            continue
        group = group.sort_values("game_play_number")
        period = group.period_number.to_numpy(dtype=int)
        clock = group.clock.to_numpy()
        home, away = group.home_score.to_numpy(), group.away_score.to_numpy()
        if not set(range(1, 5)).issubset(period) or clock[-1] != 0 or np.any(np.diff(period) < 0):
            counters["incomplete_play_by_play_excluded"] += 1
            continue
        if (home[-1], away[-1]) != (event.home_score, event.away_score):
            counters["play_by_play_final_score_mismatch"] += 1
            continue
        if period[0] != 1 or clock[0] < 540 or home[0] + away[0] > 4:
            counters["unverified_opening_score_excluded"] += 1
            continue
        elapsed = (period - 1) * 600 + (600 - clock)
        # A clock regression makes as-of ordering ambiguous, including later corrections.
        if np.any(np.diff(elapsed) < 0):
            counters["clock_regression_excluded"] += 1
            continue
        # At a boundary, include the preceding quarter's 0:00 plays only.
        # Otherwise require the play clock to precede the snapshot strictly.
        marks = 2400 - times
        groups = np.searchsorted(elapsed, marks, side="left") - 1
        for i in np.flatnonzero(marks % 600 == 0):
            tied = np.flatnonzero((elapsed == marks[i]) & (period == marks[i] // 600))
            if len(tied):
                groups[i] = tied[-1]
        if np.any(groups < 0):
            counters["missing_early_snapshot_state_excluded"] += 1
            continue
        elapsed_source = elapsed[groups]
        selected_period = period[groups]
        assert np.all((elapsed_source < marks) | ((elapsed_source == marks) & (selected_period == marks // 600)))
        wall = group.wallclock.dropna()
        final_wall = group.wallclock.iloc[-1]
        available = pd.NaT
        if (pd.notna(final_wall) and wall.is_monotonic_increasing
                and event.tip <= final_wall <= event.tip + pd.Timedelta(hours=12)):
            available = final_wall + PREGAME_DELAY
        else:
            counters["no_verifiable_final_wallclock_for_pregame_updates"] += 1
        neutral = flag(event.neutral_site)
        court = np.nan
        if neutral == 1.:
            court = 0.
        elif neutral == 0. and pd.notna(event.venue_id) and pd.notna(event.home_venue_id):
            court = float(event.venue_id == event.home_venue_id)
        counters["neutral_site_missing"] += int(np.isnan(neutral))
        counters["home_venue_indicator_missing"] += int(np.isnan(court))
        game = {"game_id": game_id, "season": int(event.season), "tip": event.tip, "available_at": available,
                "home_id": event.home_id, "away_id": event.away_id, "home_final": event.home_score,
                "away_final": event.away_score, "home_win": int(event.home_score > event.away_score),
                "neutral_site": neutral, "home_court": court, "possessions": boxes.get(game_id, np.nan)}
        games.append(game)
        margin = home[groups] - away[groups]
        snapshots.append(pd.DataFrame({"game_id": game_id, "seconds_remaining": times, "margin": margin,
                                       "pace_adjusted_margin": margin / np.sqrt(times / 60),
                                       "source_elapsed": elapsed_source, "source_period": selected_period}))
    counters["eligible_games_with_complete_snapshot_grid"] = len(games)
    return games, snapshots


def pregame_history(games, *, on_release=None, through=None):
    ratings = defaultdict(lambda: 1500.)
    offense = defaultdict(float)
    defense = defaultdict(float)
    poss = defaultdict(float)
    opponent_elo = defaultdict(float)
    counts = Counter()
    pending, features = [], []
    last_update = pd.NaT
    def release(before):
        nonlocal last_update
        while pending and pending[0][0] < before:
            last_update, _, prior, home_before, away_before = heapq.heappop(pending)
            a, b = prior["home_id"], prior["away_id"]
            p = 1 / (1 + 10 ** ((ratings[b] - ratings[a]) / 400))
            change = 20 * (prior["home_win"] - p)
            ratings[a] += change
            ratings[b] -= change
            for team, pts, allowed, opponent in ((a, prior["home_final"], prior["away_final"], away_before),
                                                  (b, prior["away_final"], prior["home_final"], home_before)):
                key = prior["season"], team
                counts[key] += 1
                opponent_elo[key] += opponent
                if np.isfinite(prior["possessions"]):
                    offense[key] += pts
                    defense[key] += allowed
                    poss[key] += prior["possessions"]
                if on_release is not None:
                    on_release(last_update, prior["season"], team, ratings[team],
                               100 * offense[key] / poss[key] if poss[key] else None,
                               100 * defense[key] / poss[key] if poss[key] else None,
                               opponent_elo[key] / counts[key])

    for game in sorted(games, key=lambda row: (row["tip"], row["game_id"])):
        tip = game["tip"]
        release(tip)
        a, b = game["home_id"], game["away_id"]
        feature = {**game, "elo_diff": ratings[a] - ratings[b], "latest_pregame_input_utc": last_update}
        for side, team in (("home", a), ("away", b)):
            key = game["season"], team
            feature[f"{side}_offense"] = 100 * offense[key] / poss[key] if poss[key] else np.nan
            feature[f"{side}_defense"] = 100 * defense[key] / poss[key] if poss[key] else np.nan
            feature[f"{side}_sos"] = opponent_elo[key] / counts[key] if counts[key] else np.nan
        assert pd.isna(last_update) or last_update < tip
        features.append(feature)
        if pd.notna(game["available_at"]):
            assert game["available_at"] > tip
            heapq.heappush(pending, (game["available_at"], game["game_id"], game, ratings[a], ratings[b]))
    if through is not None:
        release(through)
    return pd.DataFrame(features)


def make_model(family):
    # Preprocessing fits inside the fold. Empty features remain available to later folds.
    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    if family == "logistic":
        return make_pipeline(imputer, StandardScaler(), LogisticRegression(max_iter=500, random_state=SEED))
    return make_pipeline(imputer, HistGradientBoostingClassifier(max_iter=100, max_leaf_nodes=15,
                         learning_rate=.08, min_samples_leaf=40, l2_regularization=1.,
                         early_stopping=False, random_state=SEED))


def fit_expanding(data, min_train):
    seasons = sorted(data.season.unique())
    folds, complete = [], []
    for index, season in enumerate(seasons):
        if index < min_train:
            continue
        train = data[data.season < season]
        test = data[data.season == season].copy()
        if train.home_win.nunique() != 2 or test.empty:
            raise ValueError(f"Season {season} cannot be evaluated with the earlier training seasons")
        assert train.tip.max() < test.tip.min()
        assert not set(train.game_id) & set(test.game_id)
        fold_games = train.game_id.nunique(), test.game_id.nunique()
        folds.append({"test_season": int(season), "train_first_season": int(train.season.min()),
                      "train_last_season": int(train.season.max()), "train_games": fold_games[0],
                      "test_games": fold_games[1], "status": "examined_development" if season == 2026 else "validation"})
        predictions = test[["game_id", "season", "seconds_remaining", "home_win"]].copy()
        for family in ("logistic", "gradient_boosting"):
            for label, columns in FEATURES.items():
                model = make_model(family)
                model.fit(train[list(columns)], train.home_win)
                predictions[f"{family}__{label}"] = model.predict_proba(test[list(columns)])[:, 1]
        complete.append(predictions)
        print(f"Season {season}: {fold_games[0]:,} training games, {fold_games[1]:,} evaluated", flush=True)
    return (pd.concat(complete, ignore_index=True) if complete else pd.DataFrame()), pd.DataFrame(folds)


def individual_scores(y, p):
    clipped = np.clip(p, 1e-15, 1 - 1e-15)
    return ((p >= .5) == y).astype(float), (p-y)**2, -(y*np.log(clipped) + (1-y)*np.log1p(-clipped))


def metrics(y, p):
    accuracy, brier, loss = individual_scores(y, p)
    bins = np.minimum(9, (p * 10).astype(int))
    ece = sum(abs(p[bins == b].sum() - y[bins == b].sum()) for b in range(10)) / len(y)
    return {"accuracy": accuracy.mean(), "brier": brier.mean(), "log_loss": loss.mean(),
            "roc_auc": roc_auc_score(y, p) if np.unique(y).size == 2 else np.nan, "ece": ece}


def first_predictions(frame, model, threshold):
    probability = frame[model].to_numpy()
    confidence = np.maximum(probability, 1 - probability)
    eligible = frame.loc[confidence >= threshold, ["game_id", "season", "seconds_remaining", "home_win"]].copy()
    eligible["probability"] = probability[confidence >= threshold]
    eligible.sort_values(["game_id", "seconds_remaining"], ascending=[True, False], inplace=True)
    entries = eligible.drop_duplicates("game_id").copy()
    entries["won"] = ((entries.probability >= .5) == entries.home_win).astype(int)
    entries["entry_quarter"] = ((2399 - entries.seconds_remaining) // 600 + 1).astype(int)
    entries["quarter_seconds_remaining"] = entries.seconds_remaining - (4-entries.entry_quarter)*600
    entries["model"] = model
    return entries


def intervals(fixed, names, repetitions):
    y = fixed.home_win.to_numpy()
    columns = ["accuracy", "brier", "log_loss"]
    scoring = pd.DataFrame({f"{name}|{metric}": values
                            for name in names for metric, values in zip(columns, individual_scores(y, fixed[name].to_numpy()))})
    by_game = scoring.groupby([fixed.season, fixed.game_id], sort=True).mean()
    values = by_game.to_numpy()
    index = {name: i for i, name in enumerate(by_game.columns)}
    rng = np.random.default_rng(SEED)
    draws = np.zeros((repetitions, values.shape[1]))
    seasons = by_game.index.get_level_values("season").to_numpy()
    for season in np.unique(seasons):
        rows = values[seasons == season]
        for start in range(0, repetitions, 16):
            end = min(repetitions, start + 16)
            weights = rng.multinomial(len(rows), np.full(len(rows), 1/len(rows)), size=end-start)
            draws[start:end] += weights @ rows
    draws /= len(values)
    comparisons = []
    for name in names:
        family, label = name.split("__")
        if label != "combined":
            comparisons.append((name, f"{family}__combined"))
        if family == "gradient_boosting":
            comparisons.append((name, f"logistic__{label}"))
    out = []
    for candidate, reference in comparisons:
        for metric in columns:
            i, j = index[f"{candidate}|{metric}"], index[f"{reference}|{metric}"]
            lo, hi = np.quantile(draws[:, i] - draws[:, j], [.025, .975])
            out.append({"candidate": candidate, "reference": reference, "metric": metric, "games": len(values),
                        "difference": values[:, i].mean()-values[:, j].mean(), "ci_low": lo, "ci_high": hi})
    return pd.DataFrame(out)


def write_table(output, name, frame):
    frame.to_csv(output / name, index=False)


def write_report(output, ranked=None, reason=None, eligible=None, min_train=2, threshold=.60):
    lines = ["# WCBB historical feature experiments", "",
             "2019–2026 season files are eligible for research. The 2025–26 fold is an examined development benchmark. "
             "The 2026–27 season is reserved for a separate prospective test and is excluded from this runner.", "",
             "The supplied benchmark covered 37,311 games. Its reported metrics were 83.86% snapshot accuracy, "
             "0.3397 log loss, and 77.44% first qualifying win rate at 60%. The original benchmark is reported "
             "separately because its cohort and timestamps must be verified before a paired comparison.", ""]
    if reason:
        lines.extend(["## Ranking status", "", reason, "", "See data_audit.csv for input and exclusion counts."])
    else:
        lines += [f"Models use the same {eligible:,} eligible games and snapshots. The first {min_train} available "
                  "seasons form the initial training window. Later seasons are evaluated in order. Models are ranked "
                  "by pooled fixed-time log loss with Brier score as the tie-breaker.", "",
                  f"First qualifying picks use a {threshold:.0%} threshold on the one-minute game-clock grid. "
                  "The fixed-time grid uses 35, 30, 25, 20, 15, 10, and 5 minutes remaining.", "",
                  "| Rank | Model | Games | Accuracy | Brier | Log loss | AUC | ECE | First picks | First win rate |",
                  "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for row in ranked.itertuples(index=False):
            win_rate = f"{row.first_win_rate:.2%}" if row.first_games else "—"
            lines.append(f"| {row.rank} | {row.model} | {row.games:,} | {row.accuracy:.2%} | {row.brier:.4f} | "
                         f"{row.log_loss:.4f} | {row.roc_auc:.4f} | {row.ece:.4f} | {row.first_games:,} | {win_rate} |")
        lines += ["", "The bootstrap resamples games within each validation season. Each draw keeps a game's "
                  "snapshots together and measures paired differences on common observations.", "",
                  "Pregame context updates use verified final play wall-clock times plus a 24-hour embargo. "
                  "Historical archived play timestamps provide game-clock order. Live feed receipt times "
                  "and wager execution are outside this prediction experiment."]
    (output / "ranked_comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def collect_reports(predictions, output, threshold, repetitions, baseline):
    meta = ["game_id", "season", "seconds_remaining", "home_win"]
    names = [col for col in predictions if col not in meta]
    fixed = predictions[predictions.seconds_remaining.isin(FIXED_TIMES)]
    ranked, checkpoints, calibration, entries, first = [], [], [], [], []
    for name in names:
        picks = first_predictions(predictions, name, threshold)
        entries.append(picks)
        ranked.append({"model": name, "games": fixed.game_id.nunique(), "snapshots": len(fixed),
                       **metrics(fixed.home_win.to_numpy(), fixed[name].to_numpy()),
                       "first_games": len(picks), "first_win_rate": picks.won.mean()})
        scopes = [("all_validation", fixed), *((str(year), part) for year, part in fixed.groupby("season"))]
        for scope, fold in scopes:
            for checkpoint, part in [("all_fixed", fold), *list(fold.groupby("seconds_remaining"))]:
                y, p = part.home_win.to_numpy(), part[name].to_numpy()
                checkpoints.append({"model": name, "scope": scope, "seconds_remaining": checkpoint,
                                    "games": part.game_id.nunique(), "snapshots": len(part), **metrics(y, p)})
                bins = np.minimum(9, (p * 10).astype(int))
                for group in np.unique(bins):
                    mask = bins == group
                    calibration.append({"model": name, "scope": scope, "seconds_remaining": checkpoint,
                                        "bin_low": group/10, "bin_high": (group+1)/10, "snapshots": int(mask.sum()),
                                        "mean_predicted_probability": p[mask].mean(), "observed_home_win_rate": y[mask].mean()})
            observed = picks if scope == "all_validation" else picks[picks.season.astype(str) == scope]
            first.append({"model": name, "scope": scope, "entry_quarter": "all",
                          "entry_seconds_remaining": "all", "covered_games": len(observed),
                          "available_games": fold.game_id.nunique(), "actual_win_rate": observed.won.mean()})
            for (quarter, remaining), part in observed.groupby(["entry_quarter", "seconds_remaining"]):
                first.append({"model": name, "scope": scope, "entry_quarter": int(quarter),
                              "entry_seconds_remaining": int(remaining), "covered_games": len(part),
                              "available_games": fold.game_id.nunique(), "actual_win_rate": part.won.mean()})
    rank = pd.DataFrame(ranked).sort_values(["log_loss", "brier", "model"]).reset_index(drop=True)
    rank.insert(0, "rank", np.arange(1, len(rank)+1))
    exports = {
        "ranked_comparison.csv": rank,
        "fixed_time_metrics.csv": pd.DataFrame(checkpoints),
        "calibration.csv": pd.DataFrame(calibration),
        "first_qualifying_entries.csv": pd.concat(entries, ignore_index=True),
        "first_qualifying_summary.csv": pd.DataFrame(first),
        "paired_game_bootstrap.csv": intervals(fixed, names, repetitions),
        "out_of_fold_predictions.csv.gz": predictions,
    }
    if baseline is not None:
        cohort = fixed.merge(baseline, on=meta[:3], validate="one_to_one")
        if not cohort.empty:
            exports["baseline_overlap_metrics.csv"] = pd.DataFrame(
                {"model": name, "games": cohort.game_id.nunique(), "snapshots": len(cohort),
                 **metrics(cohort.home_win.to_numpy(), cohort[name].to_numpy())}
                for name in ["archived_baseline", *names])
        else:
            exports["baseline_overlap_metrics.csv"] = pd.DataFrame(
                [{"model": "archived_baseline", "games": 0, "snapshots": 0}])
    for name, frame in exports.items():
        write_table(output, name, frame)
    return rank


def run(args, output):
    missing = []
    counters = defaultdict(Counter)
    member_file = args.membership or args.data / "division_i_teams.csv"
    member_data = None
    if member_file.is_file():
        member_data = read_frame(member_file, ("season", "team_id"))
        member_data = member_data.apply(pd.to_numeric, errors="coerce")
        if (member_data.isna().any().any() or (member_data % 1 != 0).any().any() or (member_data <= 0).any().any()):
            raise ValueError("Membership requires positive integer season and team_id values")
        member_data = member_data.astype(int).drop_duplicates()
    else:
        missing.append(str(member_file))

    def audit():
        rows = [{"season": year, "reason": reason, "count": number}
                for year, counts in sorted(counters.items()) for reason, number in sorted(counts.items())]
        write_table(output, "data_audit.csv", pd.DataFrame(rows, columns=["season", "reason", "count"]))

    for year in SEASONS:
        for stem in ("wbb_schedule", "team_box", "play_by_play"):
            absent = not any((args.data / f"{stem}_{year}{suffix}").is_file() for suffix in (".csv", ".parquet"))
            counters[year][f"missing_{stem}_file"] = int(absent)
            if absent:
                missing.append(f"{stem}_{year}.parquet or .csv")
        counters[year]["missing_division_i_membership"] = int(member_data is None or not member_data.season.eq(year).any())
        if member_data is not None and counters[year]["missing_division_i_membership"]:
            missing.append(f"Division I team IDs for season ending {year}")

    counters[2027]["prospective_season_reserved"] = 1
    audit()
    if missing:
        write_report(output, reason="Historical ranking unavailable. Required source files or Division I membership are missing. "
                     "The supplied benchmark above remains the only result from the original cohort. See missing_inputs.txt.")
        (output / "missing_inputs.txt").write_text("\n".join(missing) + "\n", encoding="utf-8")
        return False

    games, grids = [], []
    for year in SEASONS:
        plan = season_plans(read_season(args.data, "wbb_schedule", year, SCHEDULE), year,
                            set(member_data.loc[member_data.season == year, "team_id"]), counters[year])
        boxes = box_possessions(read_season(args.data, "team_box", year, BOX), plan, counters[year])
        local, grid = snapshots_for_season(read_season(args.data, "play_by_play", year, PLAYS),
                                           plan, boxes, counters[year])
        games.extend(local)
        grids.extend(grid)
        print(f"Season {year}: {len(local):,} games passed cohort audit", flush=True)
    ids = Counter(game["game_id"] for game in games)
    duplicated = {game_id for game_id, count in ids.items() if count > 1}
    for game in games:
        if game["game_id"] in duplicated:
            counters[game["season"]]["game_ids_duplicated_across_seasons"] += 1
    games = [game for game in games if game["game_id"] not in duplicated]
    audit()
    if not games or len({game["season"] for game in games}) <= args.min_train_seasons:
        write_report(output, reason="Historical ranking unavailable. The audited cohorts contain too few distinct training "
                     "and evaluation seasons. See data_audit.csv.")
        return False

    priors = pregame_history(games)
    for season, group in priors.groupby("season"):
        for feature in [column for addition in ADDITIONS.values() for column in addition]:
            counters[season][f"missing_{feature}"] = int(group[feature].isna().sum())
    audit()
    context_cols = ["game_id", "season", "tip", "home_win", "latest_pregame_input_utc",
                    *(c for c in FEATURES["combined_plus_all"] if c not in SCORE)]
    data = pd.concat(grids, ignore_index=True).merge(priors[context_cols], on="game_id", validate="many_to_one")
    assert not data.duplicated(["game_id", "seconds_remaining"]).any()
    assert data.groupby("game_id").size().eq(len(ENTRY_TIMES)).all()
    assert data.season.max() <= 2026
    assert (data.latest_pregame_input_utc.isna() | (data.latest_pregame_input_utc < data.tip)).all()
    baseline = None
    if args.baseline_predictions:
        baseline = read_frame(args.baseline_predictions, ("game_id", "season", "seconds_remaining", "probability"))
        baseline = baseline.rename(columns={"probability": "archived_baseline"})
        if baseline.duplicated(["game_id", "season", "seconds_remaining"]).any() or not baseline.archived_baseline.between(0, 1).all():
            raise ValueError("Baseline predictions need unique game/season/time keys and probabilities in [0,1]")
    with threadpool_limits(limits=args.threads):
        predictions, folds = fit_expanding(data, args.min_train_seasons)
        ranked = collect_reports(predictions, output, args.threshold, args.bootstrap, baseline)
    write_table(output, "folds.csv", folds)
    write_table(output, "pregame_features.csv.gz", priors[context_cols])
    write_report(output, ranked, eligible=len(games), min_train=args.min_train_seasons, threshold=args.threshold)
    return True


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="Directory with 2019–2026 WeHoop season files")
    parser.add_argument("--membership", type=Path, help="Season-specific Division I ESPN team IDs as CSV")
    parser.add_argument("--baseline-predictions", type=Path, help="Optional read-only CSV for an exact overlap comparison")
    parser.add_argument("--output", type=Path, default=Path("results/wcbb-feature-experiments"))
    parser.add_argument("--min-train-seasons", type=int, default=2)
    parser.add_argument("--threshold", type=float, default=.60)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args(argv)
    if not (1 <= args.min_train_seasons < len(SEASONS) and .5 < args.threshold <= 1
            and args.bootstrap >= 100 and args.threads >= 1):
        parser.error("Use 1–7 training seasons, a threshold above 0.5 and at most 1, at least 100 bootstrap draws, and positive threads")
    args.output.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = Path(tempfile.mkdtemp(prefix=f"wcbb-{timestamp}-", dir=args.output))
    config = {**vars(args), "seasons": SEASONS, "prospective_season": 2027, "benchmark": BENCHMARK,
              "fixed_times": FIXED_TIMES, "entry_times": ENTRY_TIMES, "seed": SEED}
    (output / "config.json").write_text(json.dumps(config, indent=2, default=str) + "\n", encoding="utf-8")
    print(f"Output: {output.resolve()}", flush=True)
    try:
        return 0 if run(args, output) else 2
    except (OSError, ValueError, ImportError) as exc:
        write_report(output, reason=f"Historical ranking unavailable. {exc}")
        print(f"Experiment stopped: {exc}", flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
