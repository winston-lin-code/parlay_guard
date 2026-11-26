"""Fit ITF singles set-break priors from historical match scores."""

import argparse
import csv
import json
import re
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss

from .tennis_live import ARTIFACT, _completed_set


def completed_itf_singles(row):
    qualifying = row["round"].startswith("Q") and row["round"] != "QF"
    if "-ITF-" not in row["tourney_id"] or row["best_of"] != "3" or qualifying:
        return None
    score = row["score"]
    if not re.fullmatch(r"\d-\d(?:\(\d+\))?(?: \d-\d(?:\(\d+\))?){1,2}", score):
        return None
    sets = [tuple(map(int, pair.split("(")[0].split("-"))) for pair in score.split()]
    if any(not _completed_set(*pair) for pair in sets):
        return None
    margins = [a - b for a, b in sets]
    if margins[-1] <= 0 or sum(margin > 0 for margin in margins) != 2:
        return None
    return margins


def load_matches(paths):
    matches = {}
    for path in paths:
        with path.open(encoding="utf-8-sig", newline="") as file:
            for row in csv.DictReader(file):
                margins = completed_itf_singles(row)
                if margins is None:
                    continue
                date = row["tourney_date"]
                key = row["tourney_id"], row["match_num"]
                matches[key] = (date, margins)
    return sorted(matches.values())


def boundary_examples(matches, completed_sets):
    rows = [margins[:completed_sets] for _, margins in matches if len(margins) > completed_sets]
    features = np.asarray(rows)
    # Source rows put the eventual winner first. Mirror the scores to fit both orientations.
    return np.concatenate((features, -features)), np.repeat([1, 0], len(rows))


def fit_tour(matches, holdout_year):
    train = [match for match in matches if int(match[0][:4]) < holdout_year]
    holdout = [match for match in matches if int(match[0][:4]) == holdout_year]
    stages = {}
    for completed_sets in (1, 2):
        x, y = boundary_examples(train, completed_sets)
        test_x, test_y = boundary_examples(holdout, completed_sets)
        if min(len(y), len(test_y)) < 400:
            raise ValueError("Each stage needs at least 200 training and 200 holdout matches")
        model = LogisticRegression(fit_intercept=False).fit(x, y)
        probability = model.predict_proba(test_x)[:, 1]
        baseline = np.where(test_x[:, 0] > 0, .75, .25) if completed_sets == 1 else np.full(len(test_y), .5)
        metrics = {}
        for name, metric in (("brier", brier_score_loss), ("log_loss", log_loss)):
            metrics[name] = round(float(metric(test_y, probability)), 6)
            metrics["baseline_" + name] = round(float(metric(test_y, baseline)), 6)
            if metrics[name] >= metrics["baseline_" + name]:
                raise ValueError(f"Set {completed_sets + 1} failed the holdout {name} comparison")
        stages[str(completed_sets)] = {
            "weights": model.coef_[0].tolist(), "training_matches": len(y) // 2,
            "holdout_matches": len(test_y) // 2, **metrics,
        }
    return {"training_years": sorted({int(date[:4]) for date, _ in train}),
            "holdout_year": holdout_year, "set_breaks": stages}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--men", type=Path, nargs="+", required=True)
    parser.add_argument("--women", type=Path, nargs="+", required=True)
    parser.add_argument("--holdout-year", type=int, required=True)
    parser.add_argument("--output", type=Path, default=ARTIFACT)
    args = parser.parse_args()
    tours = {league: fit_tour(load_matches(paths), args.holdout_year)
             for league, paths in (("ITFME", args.men), ("ITFWO", args.women))}
    args.output.write_text(json.dumps({"version": 1, "tours": tours}, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
