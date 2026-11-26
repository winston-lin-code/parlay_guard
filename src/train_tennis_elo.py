"""Fit and validate ITF singles Elo using dated tournament results."""

import argparse
from collections import defaultdict, deque
import csv
from dataclasses import dataclass
from datetime import datetime, timedelta
from itertools import groupby, product
import json
import math
from pathlib import Path

from .tennis_elo import ARTIFACT, START_RATING, SURFACES, blended_rating, win_probability
from .train_tennis_live import completed_itf_singles

RESULT_LAG = timedelta(days=14)


@dataclass(frozen=True)
class Match:
    date: datetime
    surface: str
    winner: str
    loser: str
    winner_name: str
    loser_name: str
    margins: tuple[int, ...] = ()


def read_matches(paths):
    matches = {}
    for path in paths:
        with path.open(encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                margins = completed_itf_singles(row)
                if margins is None:
                    continue
                winner, loser = row["winner_id"], row["loser_id"]
                if not winner.isdecimal() or not loser.isdecimal() or winner == loser:
                    continue
                day = datetime.strptime(row["tourney_date"], "%Y%m%d")
                match = Match(day, row["surface"].title(), winner, loser, row["winner_name"], row["loser_name"], tuple(margins))
                key = row["tourney_id"], row["match_num"]
                if key in matches and matches[key] != match:
                    raise ValueError(f"Conflicting tournament match: {key}")
                matches[key] = match
    return sorted(matches.values(), key=lambda match: match.date)


def empty_player():
    return {"overall": START_RATING, "surfaces": {}, "matches": 0, "name": ""}


def apply_results(matches, players, k):
    overall_updates = defaultdict(float)
    surface_updates = defaultdict(float)
    for match in matches:
        winner, loser = players[match.winner], players[match.loser]
        overall_change = k * (1 - win_probability(winner["overall"], loser["overall"]))
        overall_updates[match.winner] += overall_change
        overall_updates[match.loser] -= overall_change
        if match.surface in SURFACES:
            winner_surface = winner["surfaces"].get(match.surface, START_RATING)
            loser_surface = loser["surfaces"].get(match.surface, START_RATING)
            surface_change = k * (1 - win_probability(winner_surface, loser_surface))
            surface_updates[match.winner, match.surface] += surface_change
            surface_updates[match.loser, match.surface] -= surface_change
        for player, name in ((winner, match.winner_name), (loser, match.loser_name)):
            player["matches"] += 1
            player["name"] = name
    for player_id, change in overall_updates.items():
        players[player_id]["overall"] += change
    for (player_id, surface), change in surface_updates.items():
        surfaces = players[player_id]["surfaces"]
        surfaces[surface] = surfaces.get(surface, START_RATING) + change


def walk_forward(matches, k, players, *, retain_ratings=False):
    pending = deque()
    for day, group in groupby(matches, key=lambda match: match.date):
        while pending and pending[0][0] <= day:
            _, available = pending.popleft()
            apply_results(available, players, k)
        games = list(group)
        yield day, games
        # The CSV date is a tournament start, so withhold the entire batch for two weeks.
        pending.append((day + RESULT_LAG, games))
    if retain_ratings:
        for _, games in pending:
            apply_results(games, players, k)


def replay(matches, k, weight, evaluation_years, *, retain_ratings=False):
    players = defaultdict(empty_player)
    predictions = []
    for day, games in walk_forward(matches, k, players, retain_ratings=retain_ratings):
        if day.year in evaluation_years:
            for game in games:
                a, b = (players.get(player_id) or empty_player() for player_id in (game.winner, game.loser))
                prob = win_probability(blended_rating(a, game.surface, weight),
                                       blended_rating(b, game.surface, weight))
                predictions.append((prob, bool(a["matches"] and b["matches"])))
    return summarize(predictions), players


def summarize(predictions):
    if not predictions:
        raise ValueError("No matches in evaluation seasons")
    n = len(predictions)
    return {
        "matches": n,
        "both_players_rated": sum(rated for _, rated in predictions),
        "accuracy": sum(1 if p > .5 else .5 if p == .5 else 0 for p, _ in predictions) / n,
        "brier": sum((1 - p) ** 2 for p, _ in predictions) / n,
        "log_loss": -sum(math.log(max(p, 1e-15)) for p, _ in predictions) / n,
    }


def train(matches, holdout_year):
    seasons = sorted({match.date.year for match in matches})
    if holdout_year not in seasons or sum(y < holdout_year for y in seasons) < 2:
        raise ValueError("Supply a warm-up season, validation seasons and a holdout season")
    validation_years = [y for y in seasons[1:] if y < holdout_year]
    tuning_history = [m for m in matches if m.date.year < holdout_year]
    candidates = []
    for k, weight in product((12, 20, 32, 48), (0, .25, .5, .75, 1)):
        metrics, _ = replay(tuning_history, k, weight, validation_years)
        candidates.append((metrics["log_loss"], k, weight, metrics))
    _, k, weight, validation = min(candidates)
    _, baseline_k, _, _ = min(row for row in candidates if row[2] == 0)
    holdout, players = replay(matches, k, weight, [holdout_year], retain_ratings=True)
    overall, _ = replay(matches, baseline_k, 0, [holdout_year])
    return {
        "warmup_year": seasons[0], "validation_years": validation_years, "holdout_year": holdout_year,
        "last_tournament_start": matches[-1].date.date().isoformat(),
        "k": k, "surface_weight": weight, "validation": validation, "holdout": holdout,
        "overall_only_holdout": {"k": baseline_k, **overall}, "players": dict(players),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--men", type=Path, nargs="+", required=True)
    parser.add_argument("--women", type=Path, nargs="+", required=True)
    parser.add_argument("--holdout-year", type=int, required=True)
    parser.add_argument("--output", type=Path, default=ARTIFACT)
    args = parser.parse_args()
    tours = {}
    for league, paths in (("ITFME", args.men), ("ITFWO", args.women)):
        tours[league] = train(read_matches(paths), args.holdout_year)
        print(league, json.dumps({key: value for key, value in tours[league].items() if key != "players"}), flush=True)
    payload = {"version": 1, "ratings_released_after_days": RESULT_LAG.days, "tours": tours}
    args.output.write_text(json.dumps(payload, separators=(",", ":"), sort_keys=True, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
