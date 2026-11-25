"""Fit and evaluate live NBA calibration using team logs and public play-by-play.

Run offline with pandas, scikit-learn and pyarrow. No price data is needed.
"""
import argparse
import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss

from .game_lines_live import ELO_FEATURE, FEATURES, _fit_regression, feature_training
from .inplay_calibration import (ARTIFACT, BAND_EDGES, QUANTILES, VERSION,
                                 apply_calibration, clock_band)
from .inplay_scan import probability_frame
from .predict_today import clean_team_history, load_history, make_completed_games

CHECKPOINTS = (60, 120, *range(180, 2880, 180))
REFERENCE_START = pd.Timestamp("2023-10-01")
CALIBRATION_START = pd.Timestamp("2024-10-01")
HOLDOUT_START = pd.Timestamp("2025-10-01")
COLUMNS = ["game_id", "game_date", "team_home", "team_away", "hs", "vs", "period", "secs_game", "number_event"]


def historical_priors(history_file):
    history = load_history(history_file)
    completed = make_completed_games(clean_team_history(history, history.iloc[:0], date(2026, 10, 1)))
    rows, _ = feature_training(completed, include_elo=True, include_game_id=True)
    predictions = []
    features = {"home_margin": FEATURES + [ELO_FEATURE], "total": FEATURES}
    for _, month in rows[rows.date >= REFERENCE_START].groupby(rows.date.dt.to_period("M")):
        earlier = rows[rows.date < month.date.min()]
        if len(earlier) < 300:
            continue
        forecast = month[["game_id", "date", "home_margin", "total"]].copy()
        for target, columns in features.items():
            model = _fit_regression(earlier, target, columns)
            forecast[f"predicted_{target}"] = np.round(model.predict(month[columns]), 1)
        predictions.append(forecast)
    forecasts = pd.concat(predictions).merge(
        completed[["GAME_ID", "HOME_TEAM", "AWAY_TEAM", "HOME_PTS", "AWAY_PTS"]],
        left_on="game_id", right_on="GAME_ID", validate="one_to_one",
    )
    anchor = forecasts[forecasts.date < CALIBRATION_START]
    if len(anchor) < 500:
        raise ValueError("At least 500 chronological reference forecasts are required")
    errors = {f"{target}_holdout_mae": float((anchor[target] - anchor[f"predicted_{target}"]).abs().mean())
              for target in features}
    return forecasts.set_index("game_id"), errors


def replay(path, forecasts, errors, start, end):
    plays = pd.read_parquet(path, columns=COLUMNS).sort_values(["game_id", "number_event"])
    samples = []
    matched = 0
    for game_id, events in plays.groupby("game_id", sort=False):
        game_id = str(game_id).zfill(10)
        if game_id not in forecasts.index:
            continue
        forecast = forecasts.loc[game_id]
        final = events.iloc[-1]
        if not start <= forecast.date < end:
            continue
        if (pd.Timestamp(final.game_date).normalize() != forecast.date
                or (final.team_home, final.team_away) != (forecast.HOME_TEAM, forecast.AWAY_TEAM)
                or (final.hs, final.vs) != (forecast.HOME_PTS, forecast.AWAY_PTS)
                or final.secs_game < 2880):
            continue
        regulation = events[events.period <= 4].sort_values(["secs_game", "number_event"])
        if regulation.empty or regulation.secs_game.max() < 2880:
            continue
        clocks = regulation.secs_game.to_numpy()
        positions = np.searchsorted(clocks, 2880 - np.asarray(CHECKPOINTS), side="right") - 1
        if min(positions) < 0:
            continue
        projection = {"home_margin": forecast.predicted_home_margin, "projected_total": forecast.predicted_total}
        matched += 1
        for seconds, pos in zip(CHECKPOINTS, positions):
            snapshot = regulation.iloc[pos]
            frame = probability_frame({"phase": "in", "remaining": seconds,
                                       "home_score": snapshot.hs, "away_score": snapshot.vs},
                                      projection, errors)
            samples.append((seconds, frame, forecast.home_margin, forecast.total))
    if matched < 1000:
        raise ValueError(f"Expected a full regular-season sample, matched only {matched} games in {path.name}")
    return samples, matched


def fit_curves(samples):
    residuals = {kind: [[] for _ in BAND_EDGES[:-1]] for kind in ("margin", "total")}
    for seconds, (_, margin, total), final_margin, final_total in samples:
        band = clock_band(seconds)
        for kind, base, outcome in (("margin", margin, final_margin), ("total", total, final_total)):
            residuals[kind][band].append((outcome - base.mean) / base.stdev)
    return {kind: [np.quantile(values, QUANTILES).tolist() for values in bands]
            for kind, bands in residuals.items()}


def evaluate(samples, artifact):
    scores = {name: ([], [], []) for name in ("moneyline", "spread", "total")}
    for seconds, frame, final_margin, final_total in samples:
        adjusted = apply_calibration(frame, seconds, artifact)
        for values, value in zip(scores["moneyline"], (frame[0], adjusted[0], final_margin > 0)):
            values.append(value)
        for index, name, actual in ((1, "spread", final_margin), (2, "total", final_total)):
            center = np.floor(frame[index].mean) + 0.5
            step = 6 if index == 1 else 12
            for threshold in (center - step, center, center + step):
                for values, value in zip(scores[name], (1 - frame[index].cdf(threshold),
                                                        1 - adjusted[index].cdf(threshold), actual > threshold)):
                    values.append(value)
    return {
        market: {
            "observations": len(actual),
            "raw_brier": round(float(brier_score_loss(actual, raw)), 6),
            "calibrated_brier": round(float(brier_score_loss(actual, calibrated)), 6),
            "raw_log_loss": round(float(log_loss(actual, raw, labels=[False, True])), 6),
            "calibrated_log_loss": round(float(log_loss(actual, calibrated, labels=[False, True])), 6),
        }
        for market, (raw, calibrated, actual) in scores.items()
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--team-history", required=True, type=Path)
    parser.add_argument("--calibration-pbp", required=True, type=Path)
    parser.add_argument("--holdout-pbp", required=True, type=Path)
    parser.add_argument("--output", default=ARTIFACT, type=Path)
    args = parser.parse_args()
    priors, errors = historical_priors(args.team_history)
    calibration, calibration_games = replay(args.calibration_pbp, priors, errors, CALIBRATION_START, HOLDOUT_START)
    holdout, holdout_games = replay(args.holdout_pbp, priors, errors, HOLDOUT_START, pd.Timestamp("2026-07-01"))
    artifact = {"version": VERSION, "band_edges": BAND_EDGES, "reference_errors": errors,
                "calibration_season": "2024–25", "holdout_season": "2025–26",
                "calibration_games": calibration_games, "holdout_games": holdout_games,
                "play_by_play_source": "https://github.com/ramirobentes/nba_data",
                "market_lines": "synthetic_half_point_thresholds",
                "knots": fit_curves(calibration)}
    artifact["holdout"] = evaluate(holdout, artifact)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, separators=(",", ":"), allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in artifact.items() if k != "knots"}, indent=2))


if __name__ == "__main__":
    main()
