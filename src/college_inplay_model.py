"""Fit live college game distributions using archive scores and regulation clocks."""

import argparse
import json
import re
from functools import lru_cache

import numpy as np
import pandas as pd
from scipy.special import ndtr
from sklearn.linear_model import Ridge

from .college_score_calibration import CollegeDistribution
from .college_train import MODELS, parse_schedule
from .inplay_calibration import EMPIRICAL_WEIGHT, QUANTILES
from .nfl_inplay_model import fit_distribution

# Regulation periods, seconds per period, baseline margin, and baseline total.
RULES = {"mbb": (2, 1200, 3, 145), "cfb": (4, 900, 3, 52)}
VERSION = "college-live-score-clock-v1"
COLUMNS = {
    "mbb": {"period_number": "period", "clock_display_value": "clock", "home_score": "home",
            "away_score": "away", "season_type": "season_type"},
    "cfb": {"period.number": "period", "clock.displayValue": "clock", "homeScore": "home",
            "awayScore": "away", "seasonType": "season_type", "homeTeamSpread": "home_line", "overUnder": "total_line"},
}


def score_clock(update, sport):
    periods, duration, _, _ = RULES[sport]
    scores = re.fullmatch(r"(\d+)\s*[-–]\s*(\d+)", str(update.get("score_text", "")))
    period = re.fullmatch(r"[QH]?(\d+)", str(update.get("period_text") or ""))
    clock = re.fullmatch(r"(\d{1,2}):([0-5]\d)(?:\.\d+)?", str(update.get("clock_text") or ""))
    if scores is None or period is None or clock is None:
        return None
    period_number = int(period[1])
    seconds = int(clock[1]) * 60 + int(clock[2])
    remaining = (periods - period_number) * duration + seconds
    if not 1 <= period_number <= periods or seconds > duration or remaining < 120:
        return None
    return int(scores[1]), int(scores[2]), remaining, period_number


def design(sport, home, away, seconds):
    count, duration, _, _ = RULES[sport]
    fraction = seconds / (count * duration)
    margin = home - away
    return np.stack([fraction, fraction ** 2, margin * fraction, (home + away) * fraction,
                     np.abs(margin) * fraction], axis=-1)


def inputs(sport, plays, schedules):
    names = COLUMNS[sport]
    finals = []
    for path in schedules:
        frame = pd.read_parquet(path)
        if sport == "mbb":
            frame = frame[pd.to_numeric(frame.season_type, errors="coerce") == 2]
        else:
            frame = frame[pd.to_numeric(frame.season_type_id, errors="coerce") == 2]
        finals.extend(parse_schedule(frame, sport))
    results = pd.DataFrame(finals)[["game_id", "season", "home_points", "away_points"]]
    results["game_id"] = results.game_id.astype("int64")
    results = results.drop_duplicates("game_id")
    frames = []
    count, duration, _, _ = RULES[sport]
    for path in plays:
        rows = pd.read_parquet(path, columns=["game_id", "game_play_number", *names]).rename(columns=names)
        rows["period"] = pd.to_numeric(rows.period, errors="coerce")
        clock = rows.clock.astype(str).str.extract(r"^(\d{1,2}):([0-5]\d)(?:\.\d+)?$").astype(float)
        seconds = clock[0] * 60 + clock[1]
        rows["remaining"] = (count - rows.period) * duration + seconds
        rows = rows[(pd.to_numeric(rows.season_type, errors="coerce") == 2)
                    & rows.period.between(1, count) & seconds.between(0, duration)
                    & (rows.remaining >= 120)].dropna(subset=["home", "away", "game_id"])
        rows["interval"] = np.maximum(0, seconds.loc[rows.index] - 1) // 180
        # Sample the first post-play score per game, period and three-minute clock bin.
        rows = rows.sort_values("game_play_number").drop_duplicates(["game_id", "period", "interval"])
        frames.append(rows)
    rows = pd.concat(frames, ignore_index=True).merge(results, on="game_id", validate="many_to_one")
    return rows[(rows.home <= rows.home_points) & (rows.away <= rows.away_points)
                & (rows.home >= 0) & (rows.away >= 0)]


def diagnostics(actual, means, baseline, params, reference, threshold):
    event = actual[:, None] > threshold
    valid = np.isfinite(threshold)

    def evaluate(center, settings):
        residual = (threshold - center[:, None]) / settings["sd"]
        cumulative = EMPIRICAL_WEIGHT * np.interp(residual, settings["knots"], QUANTILES)
        prob = np.clip(1 - cumulative - (1 - EMPIRICAL_WEIGHT) * ndtr(residual), 1e-9, 1 - 1e-9)[valid]
        label = event[valid]
        return float(np.mean((label - prob) ** 2)), float(-np.mean(np.where(label, np.log(prob), np.log1p(-prob))))

    brier, loss = evaluate(means, params)
    base_brier, base_loss = evaluate(baseline, reference)
    return {"observations": int(valid.sum()), "brier": brier, "log_loss": loss,
            "baseline_brier": base_brier, "baseline_log_loss": base_loss}


def fit(sport, frame):
    years = sorted(frame.season.unique())
    if len(years) != 3:
        raise ValueError("Provide three distinct complete chronological seasons")
    train, cal, test = (frame.season.to_numpy() == year for year in years)
    counts = [int(frame.loc[mask, "game_id"].nunique()) for mask in (train, cal, test)]
    if min(counts) < 400:
        raise ValueError("Too few complete games")
    home, away, seconds = (frame[key].to_numpy() for key in ("home", "away", "remaining"))
    periods, duration, margin_prior, total_prior = RULES[sport]
    x = design(sport, home, away, seconds)
    targets, metrics = {}, {}
    for kind, samples, scored, prior in (
        ("margin", frame.home_points - frame.away_points, home - away, margin_prior),
        ("total", frame.home_points + frame.away_points, home + away, total_prior),
    ):
        final = samples.to_numpy()
        model = Ridge(alpha=50, fit_intercept=False).fit(x[train], (final - scored)[train])
        means = scored + model.predict(x)
        baseline = scored + prior * seconds / (periods * duration)
        buckets = []
        market = "spreads" if kind == "margin" else "totals"
        metrics[market] = []
        if kind == "margin":
            metrics["moneyline"] = []
        for period in range(1, periods + 1):
            phase = frame.period.to_numpy() == period
            holdout = test & phase
            params = fit_distribution(final, means, cal & phase)
            reference = fit_distribution(final, baseline, cal & phase)
            buckets.append(params)
            if sport == "cfb":
                lines = (-frame.home_line if kind == "margin" else frame.total_line).to_numpy()[holdout]
                threshold = np.where(np.abs(lines) % 1 == .5, lines, np.nan)[:, None]
            else:
                shift = np.asarray([-6, 0, 6] if kind == "margin" else [-12, 0, 12])
                threshold = np.floor(baseline[holdout, None]) + .5 + shift
            metrics[market].append(diagnostics(final[holdout], means[holdout], baseline[holdout],
                                                params, reference, threshold))
            if kind == "margin":
                metrics["moneyline"].append(diagnostics(final[holdout], means[holdout], baseline[holdout],
                                                         params, reference, np.zeros((int(holdout.sum()), 1))))
        targets[kind] = {"weights": model.coef_.tolist(), "buckets": buckets}
    return {"version": VERSION, "sport": sport, "seasons": [int(year) for year in years],
            "games": counts, "targets": targets, "holdout": metrics}


@lru_cache(maxsize=2)
def load(sport):
    try:
        data = json.loads((MODELS / f"{sport}_live_scores.json").read_text(encoding="utf-8"))
        if data["version"] == VERSION and data["sport"] == sport:
            return data
    except (OSError, ValueError, KeyError):
        return None


def forecast(state, artifact):
    home, away, remaining, period = state
    x = design(artifact["sport"], home, away, remaining)
    distributions = []
    for kind, points in (("margin", home - away), ("total", home + away)):
        target = artifact["targets"][kind]
        distributions.append(CollegeDistribution(points + x @ target["weights"], target["buckets"][period - 1]))
    margin, total = distributions
    return 1 - margin.cdf(0), margin, total


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sport", choices=RULES)
    parser.add_argument("--plays", nargs="+", required=True)
    parser.add_argument("--schedules", nargs="+", required=True)
    args = parser.parse_args()
    artifact = fit(args.sport, inputs(args.sport, args.plays, args.schedules))
    (MODELS / f"{args.sport}_live_scores.json").write_text(json.dumps(artifact, separators=(",", ":"), allow_nan=False) + "\n")
    print(json.dumps({key: value for key, value in artifact.items() if key != "targets"}, indent=2))


if __name__ == "__main__":
    main()
