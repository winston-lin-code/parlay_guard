"""Train separate full-game moneyline probabilities from basketball halftime scores."""

import argparse
import ast
from functools import lru_cache
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss

from .common import MODELS

HALF_PERIOD = {"CBB": 1, "WCBB": 2, "WNBA": 2}
ARTIFACT = MODELS / "halftime_basketball.json"
VERSION = "basketball-halftime-moneyline-v1"


def features(home, away, neutral):
    return np.stack(((np.asarray(home) - away) / 10, 1 - np.asarray(neutral, dtype=float)), axis=-1)


def halftime_score(periods, total, count, half):
    if not isinstance(periods, str):
        return None
    try:
        rows = ast.literal_eval(re.sub(r"}\s*{", "}, {", periods))
        if len(rows) < count or any(row.get("period", i) != i for i, row in enumerate(rows, 1)):
            return None
        points = [row["value"] for row in rows]
        if sum(points) == total and all(p >= 0 and int(p) == p for p in points):
            return int(sum(points[:half]))
    except (SyntaxError, ValueError, TypeError, KeyError, OverflowError):
        pass
    return None


def schedule_rows(paths, league):
    cols = ["game_id", "season", "game_date", "season_type", "status_type_completed", "neutral_site",
            "notes_headline", "home_id", "away_id", "home_score", "away_score", "home_linescores", "away_linescores"]
    if league != "WNBA":
        cols.extend(("home_conference_id", "away_conference_id"))
    rows = pd.concat([pd.read_parquet(path, columns=cols) for path in paths], ignore_index=True)
    rows = rows[(rows.season_type == 2) & rows.status_type_completed & (rows.home_score != rows.away_score)]
    rows = rows[~rows.notes_headline.fillna("").str.contains("All.Star|Commissioner.s Cup Championship", case=False)]
    if league != "WNBA":
        rows = rows.dropna(subset=["home_conference_id", "away_conference_id"])
        rows = rows[(rows.home_conference_id > 0) & (rows.away_conference_id > 0)]
    half = HALF_PERIOD[league]
    for side in ("home", "away"):
        rows = rows.assign(**{"half_" + side: [
            halftime_score(lines, final, half * 2, half)
            for lines, final in zip(rows[side + "_linescores"], rows[side + "_score"])
        ]})
    return rows.dropna(subset=["half_home", "half_away"]).drop_duplicates("game_id")


def fit(rows):
    seasons = sorted(rows.season.unique())
    if len(seasons) != 3:
        raise ValueError("Supply two training seasons and one held-out season")
    train, holdout = rows.season < seasons[-1], rows.season == seasons[-1]
    if min(int(train.sum()), int(holdout.sum())) < 200:
        raise ValueError("Insufficient verified halftime scores")
    if rows.loc[train, "game_date"].max() >= rows.loc[holdout, "game_date"].min():
        raise ValueError("Training and holdout dates overlap")
    x = features(rows.half_home, rows.half_away, rows.neutral_site)
    won = (rows.home_score > rows.away_score).to_numpy()
    model = LogisticRegression(C=10, fit_intercept=False).fit(x[train], won[train])
    pred = model.predict_proba(x[holdout])[:, 1]
    # Compare to a fixed home-win rate estimated from training seasons.
    baseline = np.full(holdout.sum(), float(won[train].mean()))
    held = won[holdout]
    eligible = (pred >= .6) | (pred <= .4)
    brier, baseline_brier = brier_score_loss(held, pred), brier_score_loss(held, baseline)
    loss, baseline_loss = log_loss(held, pred), log_loss(held, baseline)
    if brier >= baseline_brier or loss >= baseline_loss:
        raise ValueError("Halftime probability failed the chronological holdout")
    return {"intercept": 0.0, "weights": model.coef_[0].tolist(),
            "teams": sorted({int(t) for t in pd.concat([rows.home_id, rows.away_id])}),
            "training_seasons": [int(y) for y in seasons[:-1]], "holdout_season": int(seasons[-1]),
            "holdout": {"games": int(holdout.sum()), "brier": round(brier, 6), "log_loss": round(loss, 6),
                        "baseline_brier": round(baseline_brier, 6), "baseline_log_loss": round(baseline_loss, 6),
                        "p60_games": int(eligible.sum()),
                        "p60_accuracy": round(float(((pred[eligible] >= .5) == held[eligible]).mean()), 6),
                        "p60_predicted": round(float(np.maximum(pred[eligible], 1 - pred[eligible]).mean()), 6)}}


@lru_cache(maxsize=1)
def load_models():
    try:
        artifact = json.loads(ARTIFACT.read_text(encoding="utf-8"))
        if artifact["version"] == VERSION:
            return artifact["leagues"]
    except (OSError, ValueError, KeyError):
        pass
    return {}


def win_probability(model, home_score, away_score, neutral):
    return float(expit(model["intercept"] + features(home_score, away_score, neutral) @ model["weights"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for league in HALF_PERIOD:
        parser.add_argument(f"--{league.lower()}", nargs=3, required=True, type=Path)
    args = parser.parse_args()
    models = {league: fit(schedule_rows(getattr(args, league.lower()), league)) for league in HALF_PERIOD}
    ARTIFACT.write_text(json.dumps({"version": VERSION, "leagues": models}, separators=(",", ":"), allow_nan=False) + "\n")
    print(json.dumps({league: model["holdout"] for league, model in models.items()}, indent=2))


if __name__ == "__main__":
    main()
