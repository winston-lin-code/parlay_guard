"""Estimate the historical association between regular-season QB changes and margins."""

import argparse
import json
from collections import Counter, defaultdict, deque

import numpy as np
import pandas as pd


def observations(games):
    history = defaultdict(lambda: deque(maxlen=4))
    rows = []
    for game in games.sort_values(["gameday", "game_id"]).itertuples():
        sides = [(getattr(game, f"{side}_team"), getattr(game, f"{side}_qb_id"))
                 for side in ("home", "away")]
        states = [history[game.season, team] for team, _ in sides]
        if all(len(state) == 4 for state in states) and all(pd.notna(qb) for _, qb in sides):
            incumbents = [Counter(result[0] for result in state).most_common(1)[0] for state in states]
            if min(count for _, count in incumbents) >= 3 and pd.notna(game.spread_line):
                changes = [int(side[1] != recent[0]) for side, recent in zip(sides, incumbents)]
                form = [sum(result[1] for result in state) / 4 for state in states]
                rows.append((game.season, game.week, changes[0], changes[1], sides[0][0], sides[1][0],
                             form[0] - form[1], game.location != "Neutral", game.result, game.spread_line))
        states[0].append((game.home_qb_id, game.result))
        states[1].append((game.away_qb_id, -game.result))
    return pd.DataFrame(rows, columns=("season", "week", "home_changed", "away_changed", "home",
                                      "away", "form_diff", "home_field", "margin", "spread"))


def regression(x, outcomes, variables):
    """OLS coefficients of interest with HC3 standard errors."""
    x_inv = np.linalg.pinv(x)
    leverage = np.einsum("ij,ji->i", x, x_inv)
    metrics = {}
    for name, y in outcomes.items():
        y = np.asarray(y, dtype=float)
        coef = x_inv @ y
        residual = y - x @ coef
        errors = x_inv[:len(variables)] * residual / np.maximum(1e-8, 1 - leverage)
        metrics[name] = {}
        for column, variable in enumerate(variables):
            se = np.linalg.norm(errors[column])
            point = coef[column]
            metrics[name][variable] = {"points": round(float(point), 2),
                                       "approx_95_ci": [round(float(point - 1.96*se), 2),
                                                        round(float(point + 1.96*se), 2)]}
    return metrics


def team_effects(rows):
    away = pd.get_dummies(rows.season.astype(str) + "/" + rows.away.astype(str), dtype=float)
    home = pd.get_dummies(rows.season.astype(str) + "/" + rows.home.astype(str), dtype=float)
    return home.sub(away, fill_value=0)


def estimate(rows):
    change = rows.home_changed - rows.away_changed
    x = np.column_stack([change, np.ones(len(rows)), rows.week, rows.form_diff, rows.home_field, team_effects(rows)])
    metrics = regression(x, {"actual_margin": rows.margin, "archived_spread": rows.spread,
                             "margin_minus_archived_spread": rows.margin - rows.spread}, ["changed"])
    return {"games": len(rows), "games_with_one_qb_change": int((change != 0).sum()),
            "games_with_both_qbs_changed": int(((rows.home_changed + rows.away_changed) == 2).sum()),
            **{name: estimates["changed"] for name, estimates in metrics.items()}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("schedule_csv")
    args = parser.parse_args()
    games = pd.read_csv(args.schedule_csv, low_memory=False)
    games = games[games.season.between(2016, 2025) & (games.game_type == "REG") & games.result.notna()]
    print(json.dumps(estimate(observations(games)), indent=2))


if __name__ == "__main__":
    main()
