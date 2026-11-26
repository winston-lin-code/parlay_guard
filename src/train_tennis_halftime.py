"""Train ITF Elo-adjusted probabilities at first- and second-set breaks."""

import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss

from .tennis_elo import ARTIFACT as ELO_ARTIFACT
from .tennis_live import ELO_BREAK_ARTIFACT, load_models
from .train_tennis_elo import empty_player, read_matches, walk_forward


def fit(matches, elo, score_model):
    players = defaultdict(empty_player)
    rows = {stage: ([], []) for stage in (0, 1, 2)}
    holdout_year = elo["holdout_year"]
    validation_years = elo["validation_years"]
    for day, games in walk_forward(matches, elo["k"], players):
        if day.year not in (*validation_years, holdout_year):
            continue
        split = int(day.year == holdout_year)
        for match in games:
            a, b = (players.get(player_id) for player_id in (match.winner, match.loser))
            if not a or not b:
                continue
            gap = (a["overall"] - b["overall"]) / 400
            for stage in range(len(match.margins)):
                rows[stage][split].append((gap, *match.margins[:stage]))
    output = {}
    for stage, (training, test) in rows.items():
        x, held = (np.asarray(group) for group in (training, test))
        if min(len(x), len(held)) < 200:
            raise ValueError("Insufficient rated matches at set boundary")
        model = LogisticRegression(fit_intercept=False).fit(np.concatenate([x, -x]), np.repeat([1, 0], len(x)))
        p = model.predict_proba(np.concatenate([held, -held]))[:, 1]
        y = np.repeat([1, 0], len(held))
        if stage == 0:
            baseline_log_odds = held[:, 0] * np.log(10)
        else:
            baseline_log_odds = held[:, 1:] @ score_model["set_breaks"][str(stage)]["weights"]
        baseline = 1 / (1 + np.exp(-np.concatenate([baseline_log_odds, -baseline_log_odds])))
        metrics = {"matches": len(test)}
        for label, score in (("brier", brier_score_loss), ("log_loss", log_loss)):
            metrics[label] = round(float(score(y, p)), 6)
            metrics["baseline_" + label] = round(float(score(y, baseline)), 6)
        if stage and (metrics["brier"] >= metrics["baseline_brier"]
                      or metrics["log_loss"] >= metrics["baseline_log_loss"]):
            raise ValueError(f"Elo failed to improve set break {stage}: {metrics}")
        output[str(stage)] = {"weights": model.coef_[0].tolist(), "holdout": metrics}
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--men", type=Path, nargs="+", required=True)
    parser.add_argument("--women", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, default=ELO_BREAK_ARTIFACT)
    args = parser.parse_args()
    elo = json.loads(ELO_ARTIFACT.read_text(encoding="utf-8"))["tours"]
    score_models = load_models()
    tours = {code: fit(read_matches(paths), elo[code], score_models[code])
             for code, paths in (("ITFME", args.men), ("ITFWO", args.women))}
    artifact = {"version": 1, "holdout_year": elo["ITFME"]["holdout_year"], "tours": tours}
    args.output.write_text(json.dumps(artifact, indent=2, allow_nan=False) + "\n")
    print(json.dumps(artifact, indent=2))


if __name__ == "__main__":
    main()
