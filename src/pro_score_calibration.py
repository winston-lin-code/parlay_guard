"""Train pregame score distributions from regular-season pro results."""

import argparse
import csv
import json
import math
from collections import defaultdict
from datetime import date
from functools import lru_cache
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.preprocessing import StandardScaler

from .college_score_calibration import CollegeDistribution
from .inplay_calibration import QUANTILES

MODEL_DIR = Path(__file__).resolve().parents[1] / "models" / "pro_scores"
SEASONS = {"nfl": (2024, 2025), "nhl": (2024, 2025), "mlb": (2024, 2025)}
SPORT_AVERAGES = {"nfl": 22, "nhl": 3, "mlb": 4.5}
VERSION = "pro-pregame-scores-v1"


def games_from(path, sport):
    if sport == "nfl":
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if row["game_type"] != "REG" or not row["home_score"] or not row["away_score"]:
                    continue
                yield (row["game_id"], row["gameday"], int(row["season"]), row["home_team"], row["away_team"],
                       int(row["home_score"]), int(row["away_score"]), row["location"] == "Neutral",
                       float(row["spread_line"] or "nan"), float(row["total_line"] or "nan"))
        return
    raw = json.loads(path.read_text())
    if sport == "nhl":
        for g in raw["games"]:
            if g["gameType"] != 2 or g["gameState"] not in ("OFF", "FINAL"):
                continue
            yield (g["id"], g["gameDate"], g["season"] // 10000, g["homeTeam"]["abbrev"], g["awayTeam"]["abbrev"],
                   g["homeTeam"]["score"], g["awayTeam"]["score"], g.get("neutralSite", False), math.nan, math.nan)
        return
    for slate in raw["dates"]:
        for g in slate["games"]:
            if g["gameType"] != "R" or g["status"]["abstractGameState"] != "Final" or g.get("isTie"):
                continue
            home, away = (g["teams"][side] for side in ("home", "away"))
            if "score" not in home or "score" not in away:
                continue
            yield (g["gamePk"], g["officialDate"], int(g["season"]), str(home["team"]["id"]), str(away["team"]["id"]),
                   home["score"], away["score"], False, math.nan, math.nan)


def features(sport, states, home, away, day, neutral):
    values = [float(neutral)]
    default = SPORT_AVERAGES[sport]
    for team in (home, away):
        state = states.get(team, {})
        results = state.get("results", [])
        values.extend((state.get("rating", 1500) - 1500,
                       min(21, (date.fromisoformat(day) - date.fromisoformat(state["day"])).days) if results else 7))
        for index in (0, 1):
            values.append((sum(result[index] for result in results) + 4 * default) / (len(results) + 4))
    return values


def sample(sport, games):
    states, rows = {}, []
    same_day = defaultdict(list)
    for game in games:
        same_day[game[1]].append(game)
    # Every forecast on a given date uses only results from earlier dates.
    for day, slate in sorted(same_day.items()):
        season = slate[0][2]
        participating = {team for g in slate for team in g[3:5]}
        for team in participating:
            old = states.get(team, {})
            if old.get("season") != season:
                states[team] = {"rating": 1500 + .65 * (old.get("rating", 1500) - 1500),
                                "season": season, "day": old.get("day", day), "results": old.get("results", [])[-5:]}
        rows.extend(features(sport, states, home, away, day, neutral) for _, _, _, home, away, _, _, neutral, _, _ in slate)
        for _, _, _, home, away, h, a, neutral, _, _ in slate:
            expected = 1 / (1 + 10 ** ((states[away]["rating"] - states[home]["rating"] - (0 if neutral else 35)) / 400))
            delta = 20 * ((1 if h > a else .5 if h == a else 0) - expected)
            for team, points, against, change in ((home, h, a, delta), (away, a, h, -delta)):
                state = states[team]
                state["rating"] += change
                state["day"] = day
                state["results"] = (state["results"] + [[points, against]])[-(10 if sport == "nfl" else 20):]
    return np.asarray(rows), states


def fit_distribution(actual, means, mask):
    residual = actual[mask] - means[mask]
    sd = float(np.sqrt(np.mean(residual ** 2)))
    return {"sd": sd, "knots": np.quantile(residual / sd, QUANTILES).tolist()}


def compare(actual, means, baseline, params, reference, test, lines):
    outcomes, model, base = [], [], []
    for index in np.flatnonzero(test):
        dist = CollegeDistribution(means[index], params)
        base_dist = CollegeDistribution(baseline[index], reference)
        for cutoff in lines(index):
            outcomes.append(actual[index] > cutoff)
            model.append(1 - dist.cdf(cutoff))
            base.append(1 - base_dist.cdf(cutoff))
    return {"observations": len(outcomes),
            "brier": brier_score_loss(outcomes, model), "baseline_brier": brier_score_loss(outcomes, base),
            "log_loss": log_loss(outcomes, model, labels=[False, True]),
            "baseline_log_loss": log_loss(outcomes, base, labels=[False, True])}


def fit(sport, records):
    games = sorted({g[0]: g for g in records}.values(), key=lambda g: (g[1], g[0]))
    x, states = sample(sport, games)
    seasons = np.array([g[2] for g in games])
    cal_year, test_year = SEASONS[sport]
    train, cal, test = seasons < cal_year, seasons == cal_year, seasons == test_year
    if min(train.sum(), cal.sum(), test.sum()) < 200:
        raise ValueError(f"Insufficient completed games for {sport}")
    if any(games[i][1] >= games[j][1] for i, j in (
            (np.flatnonzero(train)[-1], np.flatnonzero(cal)[0]),
            (np.flatnonzero(cal)[-1], np.flatnonzero(test)[0]))):
        raise ValueError("Season partitions overlap")
    home, away = (np.asarray([g[i] for g in games]) for i in (5, 6))
    tie_rate = float(np.mean(home[train | cal] == away[train | cal]))
    targets, holdout = {}, {}
    for kind, outcome in (("margin", home - away), ("total", home + away)):
        scaler = StandardScaler().fit(x[train])
        reg = Ridge(alpha=50).fit(scaler.transform(x[train]), outcome[train])
        weights = reg.coef_ / scaler.scale_
        intercept = float(reg.intercept_ - weights @ scaler.mean_)
        means = x @ weights + intercept
        baseline = np.full(len(games), float(np.mean(outcome[train])))
        params = fit_distribution(outcome, means, cal)
        reference = fit_distribution(outcome, baseline, cal)
        targets[kind] = {**params, "weights": weights.tolist(), "intercept": intercept}
        index = 8 if kind == "margin" else 9
        offsets = ((-1, 0, 1) if sport != "nfl" else (-6, 0, 6) if kind == "margin" else (-8, 0, 8))

        def thresholds(i):
            line = games[i][index]
            if sport == "nfl":
                return (line,) if math.isfinite(line) and abs(line) % 1 == .5 else ()
            if kind == "margin":
                return (-1.5, 1.5)
            return tuple(math.floor(baseline[i]) + .5 + dx for dx in offsets)

        holdout["spreads" if kind == "margin" else "totals"] = compare(
            outcome, means, baseline, params, reference, test, thresholds)
        if kind == "margin":
            holdout["moneyline"] = compare(outcome, means, baseline, params, reference, test & (outcome != 0),
                                            lambda _: (0,))
    return {"version": VERSION, "sport": sport, "targets": targets, "teams": states, "tie_rate": tie_rate,
            "train_games": int(train.sum()), "calibration_games": int(cal.sum()), "holdout_games": int(test.sum()),
            "calibration_season": cal_year, "holdout_season": test_year, "holdout": holdout,
            "team_results_through": games[-1][1],
            "market_lines": "nflverse archived half-point lines" if sport == "nfl" else "simulated half-point lines"}


@lru_cache(maxsize=3)
def load(sport):
    try:
        model = json.loads((MODEL_DIR / f"{sport}.json").read_text())
        if model["version"] == VERSION and model["sport"] == sport:
            return model
    except (OSError, ValueError, KeyError):
        return None


def predict(game, model):
    teams = model["teams"]
    home, away = (game[side]["id"] for side in ("home", "away"))
    day = game["day"]
    if any(team not in teams or len(teams[team]["results"]) < 3 or teams[team]["day"] >= day for team in (home, away)):
        return None
    x = features(model["sport"], teams, home, away, day, game["neutral"])
    margin, total = [CollegeDistribution(p["intercept"] + sum(w*v for w, v in zip(p["weights"], x)), p)
                     for p in (model["targets"]["margin"], model["targets"]["total"])]
    return 1 - margin.cdf(0), margin, total


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sport", choices=SEASONS)
    parser.add_argument("inputs", nargs="+", type=Path)
    args = parser.parse_args()
    model = fit(args.sport, [game for path in args.inputs for game in games_from(path, args.sport)])
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    (MODEL_DIR / f"{args.sport}.json").write_text(json.dumps(model, separators=(",", ":"), allow_nan=False) + "\n")
    print(json.dumps({key: val for key, val in model.items() if key not in ("teams", "targets")}, indent=2))


if __name__ == "__main__":
    main()
