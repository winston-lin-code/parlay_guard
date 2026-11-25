"""Train NFL live outcome distributions from prior-play scores and regulation time."""

import argparse
import json
import math
import re
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from .college_score_calibration import CollegeDistribution
from .pro_score_calibration import compare, fit_distribution

ARTIFACT = Path(__file__).resolve().parents[1] / "models" / "pro_scores" / "nfl_live.json"
VERSION = "nfl-live-score-clock-v1"
COLUMNS = ("game_id", "season", "season_type", "qtr", "game_seconds_remaining", "total_home_score",
           "total_away_score", "home_score", "away_score", "spread_line", "total_line")


def score_clock(update):
    scores = re.fullmatch(r"(\d+)\s*[-–]\s*(\d+)", update.get("score_text", ""))
    if scores is None:
        return None
    period = str(update.get("period_text") or "").strip().upper()
    if period in ("HT", "HALFTIME", "HALF TIME"):
        return int(scores[1]), int(scores[2]), 1800
    quarter = re.fullmatch(r"Q([1-4])|([1-4])Q", period)
    clock = re.fullmatch(r"(\d{1,2}):([0-5]\d)", str(update.get("clock_text") or "").strip())
    if quarter is None or clock is None:
        return None
    current = int(quarter[1] or quarter[2])
    seconds = int(clock[1]) * 60 + int(clock[2])
    remaining = (4 - current) * 900 + seconds
    if seconds > 900 or not 120 <= remaining <= 3600:
        return None
    return int(scores[1]), int(scores[2]), remaining


def design(home, away, seconds):
    fraction = seconds / 3600
    margin, total = home - away, home + away
    return np.stack([fraction, fraction ** 2, margin * fraction, total * fraction], axis=-1)


def slices(seconds):
    return np.clip((np.asarray(seconds, dtype=int) - 1) // 900, 0, 3)


def load_inputs(inputs):
    frame = pd.concat([pd.read_parquet(path, columns=list(COLUMNS)) for path in inputs], ignore_index=True)
    frame = frame[(frame.season_type == "REG") & frame.qtr.between(1, 4)
                  & frame.game_seconds_remaining.between(120, 3600)].dropna(subset=[
                      "game_seconds_remaining", "home_score", "away_score", "total_home_score", "total_away_score"])
    # Give each game one observation per three-minute regulation interval, avoiding
    # overweighting drives with unusually many plays at the same score and clock.
    frame = frame.assign(interval=(frame.game_seconds_remaining - 1) // 180)
    return frame.drop_duplicates(["game_id", "interval"], keep="first").reset_index(drop=True)


def win_payout(distribution):
    return 1 - .5 * (distribution.cdf(.5) + distribution.cdf(-.5))


def moneyline_diagnostics(actual, means, baseline, params, reference, mask):
    outcomes = np.sign(actual[mask]) * .5 + .5
    probabilities = [np.clip([win_payout(CollegeDistribution(mean, dist)) for mean in group[mask]], 1e-9, 1 - 1e-9)
                     for group, dist in ((means, params), (baseline, reference))]
    def brier(probability):
        return float(np.mean((outcomes - probability) ** 2))

    def loss(probability):
        return -float(np.mean(outcomes * np.log(probability) + (1 - outcomes) * np.log1p(-probability)))

    model, simple = probabilities
    return {"observations": len(outcomes), "brier": brier(model), "baseline_brier": brier(simple),
            "log_loss": loss(model), "baseline_log_loss": loss(simple)}


def fit(frame):
    train = frame.season.to_numpy() <= 2023
    cal = frame.season.to_numpy() == 2024
    holdout = frame.season.to_numpy() == 2025
    counts = [int(frame.loc[mask, "game_id"].nunique()) for mask in (train, cal, holdout)]
    if min(counts) < 240:
        raise ValueError("Each chronological partition needs at least 240 NFL games")
    home, away, seconds = (frame[key].to_numpy() for key in
                           ("total_home_score", "total_away_score", "game_seconds_remaining"))
    home_final, away_final = (frame[key].to_numpy() for key in ("home_score", "away_score"))
    design_matrix = design(home, away, seconds)
    time_slice = slices(seconds)
    targets, holdout_summary = {}, {}
    for kind, current, actual, baseline_addition, line_col in (
        ("margin", home - away, home_final - away_final, 1.5, "spread_line"),
        ("total", home + away, home_final + away_final, 44, "total_line"),
    ):
        model = Ridge(alpha=50, fit_intercept=False).fit(design_matrix[train], (actual - current)[train])
        means = current + model.predict(design_matrix)
        baseline = current + baseline_addition * seconds / 3600
        params, reference = [], []
        for bucket in range(4):
            mask = cal & (time_slice == bucket)
            params.append(fit_distribution(actual, means, mask))
            reference.append(fit_distribution(actual, baseline, mask))
        targets[kind] = {"weights": model.coef_.tolist(), "buckets": params}
        market = "spreads" if kind == "margin" else "totals"
        holdout_summary[market] = []
        if kind == "margin":
            holdout_summary["moneyline"] = []
        lines = frame[line_col].to_numpy()

        def half_point(i):
            line = lines[i]
            return (line,) if math.isfinite(line) and abs(line) % 1 == .5 else ()

        for bucket in range(4):
            mask = holdout & (time_slice == bucket)
            holdout_summary[market].append(compare(actual, means, baseline, params[bucket], reference[bucket],
                                                   mask, half_point))
            if kind == "margin":
                holdout_summary["moneyline"].append(moneyline_diagnostics(
                    actual, means, baseline, params[bucket], reference[bucket], mask))
    return {"version": VERSION, "targets": targets, "holdout": holdout_summary,
            "training_games": counts[0], "calibration_games": counts[1], "holdout_games": counts[2],
            "min_seconds_remaining": 120, "holdout_season": 2025}


@lru_cache(maxsize=1)
def load():
    try:
        artifact = json.loads(ARTIFACT.read_text())
        if artifact["version"] == VERSION:
            return artifact
    except (OSError, ValueError, KeyError):
        return None


def forecast(state, artifact):
    home, away, seconds = state
    bucket = slices(seconds)
    x = design(home, away, seconds)
    distributions = []
    for kind, scored in (("margin", home - away), ("total", home + away)):
        target = artifact["targets"][kind]
        mean = scored + x @ target["weights"]
        distributions.append(CollegeDistribution(mean, target["buckets"][bucket]))
    margin, total = distributions
    return (win_payout(margin), margin, total), bucket


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("parquets", nargs="+", type=Path)
    args = parser.parse_args()
    artifact = fit(load_inputs(args.parquets))
    ARTIFACT.write_text(json.dumps(artifact, separators=(",", ":"), allow_nan=False) + "\n")
    print(json.dumps({key: value for key, value in artifact.items() if key != "targets"}, indent=2))


if __name__ == "__main__":
    main()
