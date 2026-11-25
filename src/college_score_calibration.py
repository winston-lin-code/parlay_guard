"""Chronologically validated pregame margin/total distributions for NCAA markets."""

import argparse
import hashlib
import json
import math
from functools import lru_cache
from pathlib import Path
from statistics import NormalDist

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

from .college_train import MODELS, parse_schedule, snapshot_features, training_rows
from .inplay_calibration import EMPIRICAL_WEIGHT, QUANTILES

VERSION = "college-pregame-score-cdf-v1"
TARGETS = ("margin", "total")
SEASONS = {"mbb": (2025, 2026), "cfb": (2024, 2025)}
NEUTRAL_POINTS = {"mbb": 70, "cfb": 26}


def scoring_features(history, home, away, sport):
    default = NEUTRAL_POINTS[sport]
    return [float(np.mean([r[column] for r in history.get(team, [])] or [default]))
            for team in (home, away) for column in (0, 1)]


class CollegeDistribution:
    def __init__(self, mean, params, calibrated=True):
        self.mean = float(mean)
        self.base = NormalDist(self.mean, params["sd"])
        self.knots = params["knots"]
        self.calibrated = calibrated

    def cdf(self, threshold):
        raw = self.base.cdf(threshold)
        if not self.calibrated:
            return raw
        residual = (threshold - self.mean) / self.base.stdev
        empirical = float(np.interp(residual, self.knots, QUANTILES))
        return EMPIRICAL_WEIGHT * empirical + (1 - EMPIRICAL_WEIGHT) * raw


def _features_for_model(game, bundle):
    home, away = (game[k]["id"] for k in ("home", "away"))
    states = bundle["teams"]
    if any(team not in states or states[team]["season"] != game["season"] or states[team]["games"] < 3
           for team in (home, away)):
        return None
    base, _, _ = snapshot_features(states, home, away, game["start_utc"], game["season"], game["neutral"])
    return base + scoring_features(bundle["scores"], home, away, game["sport"])


def predict_game(game, bundle):
    features = _features_for_model(game, bundle)
    if features is None:
        return None
    vectors = []
    for kind in TARGETS:
        model = bundle["targets"][kind]
        mean = model["intercept"] + sum(weight * value for weight, value in zip(model["weights"], features))
        vectors.append(CollegeDistribution(mean, model))
    margin, total = vectors
    return 1 - margin.cdf(0), margin, total


def _fit_inputs(games, sport):
    base, _, states, _ = training_rows(games, sport)
    scores, season_by_team, features, usable = {}, {}, [], []
    for game, row in zip(games, base):
        home, away, season = game["home_id"], game["away_id"], game["season"]
        for team in (home, away):
            if season_by_team.get(team) != season:
                scores[team] = []
                season_by_team[team] = season
        features.append(row + scoring_features(scores, home, away, sport))
        usable.append(min(len(scores[home]), len(scores[away])) >= 3)
        for team, opponent in ((home, away), (away, home)):
            points = game["home_points"] if team == home else game["away_points"]
            allowed = game["home_points"] if opponent == home else game["away_points"]
            scores[team] = (scores[team] + [[points, allowed]])[-10:]
    return np.asarray(features), np.asarray(usable), states, scores


def _diagnostics(means, outcomes, params, offsets, moneyline=False):
    actual, raw_prob, calibrated_prob = [], [], []
    for mean, outcome in zip(means, outcomes):
        raw = CollegeDistribution(mean, params, calibrated=False)
        adjusted = CollegeDistribution(mean, params)
        cutoffs = (0,) if moneyline else tuple(math.floor(mean) + 0.5 + shift for shift in offsets)
        for cutoff in cutoffs:
            actual.append(outcome > cutoff)
            raw_prob.append(1 - raw.cdf(cutoff))
            calibrated_prob.append(1 - adjusted.cdf(cutoff))
    return {
        "observations": len(actual),
        "raw_brier": round(float(brier_score_loss(actual, raw_prob)), 6),
        "calibrated_brier": round(float(brier_score_loss(actual, calibrated_prob)), 6),
        "raw_log_loss": round(float(log_loss(actual, raw_prob, labels=[False, True])), 6),
        "calibrated_log_loss": round(float(log_loss(actual, calibrated_prob, labels=[False, True])), 6),
    }


def fit_artifact(sport, games):
    games = sorted({g["game_id"]: g for g in games}.values(), key=lambda g: (g["date"], g["game_id"]))
    calibration_season, test_season = SEASONS[sport]
    features, eligible, states, scores = _fit_inputs(games, sport)
    seasons = np.asarray([g["season"] for g in games])
    train, calibration, test = [mask & eligible for mask in (
        seasons < calibration_season, seasons == calibration_season, seasons == test_season)]
    if min(train.sum(), calibration.sum(), test.sum()) < 400:
        raise ValueError(f"Insufficient chronological game samples for {sport}")
    if max(g["date"] for g, keep in zip(games, train) if keep) >= min(
            g["date"] for g, keep in zip(games, calibration) if keep):
        raise ValueError("Training overlaps calibration")
    if max(g["date"] for g, keep in zip(games, calibration) if keep) >= min(
            g["date"] for g, keep in zip(games, test) if keep):
        raise ValueError("Calibration overlaps the holdout")
    home = np.asarray([g["home_points"] for g in games])
    away = np.asarray([g["away_points"] for g in games])
    targets = {}
    holdout = {}
    for kind, actual in zip(TARGETS, (home - away, home + away)):
        scale = StandardScaler().fit(features[train])
        model = Ridge(alpha=150).fit(scale.transform(features[train]), actual[train])
        weights = model.coef_ / scale.scale_
        intercept = model.intercept_ - weights @ scale.mean_
        means = features @ weights + intercept
        errors = actual[calibration] - means[calibration]
        sd = float(np.sqrt(np.mean(errors ** 2)))
        params = {"weights": weights.tolist(), "intercept": float(intercept), "sd": sd,
                  "knots": np.quantile(errors / sd, QUANTILES).tolist()}
        targets[kind] = params
        key = "spreads" if kind == "margin" else "totals"
        offset = 6 if kind == "margin" else 12
        holdout[key] = _diagnostics(means[test], actual[test], params, (-offset, 0, offset))
        if kind == "margin":
            holdout["moneyline"] = _diagnostics(means[test], actual[test], params, (), moneyline=True)
    return {"version": VERSION, "sport": sport, "targets": targets, "teams": states, "scores": scores,
            "source": "SportsDataverse ESPN schedules", "market_lines": "synthetic_half_point_thresholds",
            "train_games": int(train.sum()), "calibration_games": int(calibration.sum()), "holdout_games": int(test.sum()),
            "calibration_season": calibration_season, "holdout_season": test_season,
            "team_results_through": games[-1]["date"], "holdout": holdout}


@lru_cache(maxsize=2)
def load_score_artifact(sport):
    try:
        value = json.loads((MODELS / f"{sport}_calibrated_scores.json").read_text(encoding="utf-8"))
        if value["version"] != VERSION or value["sport"] != sport:
            return None
        for key in TARGETS:
            model = value["targets"][key]
            if (len(model["weights"]) != 9 or len(model["knots"]) != len(QUANTILES)
                    or not all(math.isfinite(x) for x in [model["intercept"], model["sd"], *model["weights"], *model["knots"]])
                    or model["sd"] <= 0 or any(a > b for a, b in zip(model["knots"], model["knots"][1:]))):
                return None
        return value
    except (ValueError, OSError, TypeError, KeyError):
        return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sport", choices=tuple(SEASONS))
    parser.add_argument("data", type=Path, nargs="+")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    records = []
    for path in args.data:
        rows = parse_schedule(pd.read_parquet(path), args.sport)
        if not rows:
            raise ValueError(f"No complete games in {path.name}")
        records.extend(rows)
    artifact = fit_artifact(args.sport, records)
    artifact["input_sha256"] = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in args.data}
    output = args.output or MODELS / f"{args.sport}_calibrated_scores.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(artifact, separators=(",", ":"), allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({key: artifact[key] for key in ("train_games", "calibration_games", "holdout_games", "team_results_through", "holdout")}, indent=2))


if __name__ == "__main__":
    main()
