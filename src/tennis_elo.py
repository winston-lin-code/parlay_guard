"""Pre-match Elo probabilities for men's and women's ITF singles."""

import argparse
import json
from pathlib import Path
import unicodedata

ARTIFACT = Path(__file__).resolve().parents[1] / "models" / "itf_elo.json"
START_RATING = 1500.0
SURFACES = ("Hard", "Clay", "Grass", "Carpet")


def win_probability(first_rating, second_rating):
    return 1 / (1 + 10 ** ((second_rating - first_rating) / 400))


def blended_rating(player, surface, weight):
    overall = player["overall"]
    court = player["surfaces"].get(surface, overall)
    return overall + weight * (court - overall)


def player_key(name):
    return " ".join(unicodedata.normalize("NFKC", name).casefold().split())


def resolve_player(players, query):
    if query in players:
        return query
    matches = [player_id for player_id, player in players.items()
               if player_key(player["name"]) == player_key(query)]
    if len(matches) != 1:
        detail = "ambiguous; provide a player ID" if matches else "missing from the ratings"
        raise ValueError(f"{query}: {detail}")
    return matches[0]


def predict(model, first, second, surface=None):
    if surface is not None and surface not in SURFACES:
        raise ValueError(f"Unsupported surface: {surface}")
    players = model["players"]
    first_id, second_id = (resolve_player(players, query) for query in (first, second))
    if first_id == second_id:
        raise ValueError("Choose two distinct players")
    ratings = [blended_rating(players[player_id], surface, model["surface_weight"])
               for player_id in (first_id, second_id)]
    probability = win_probability(*ratings)
    return {
        "ratings_through_tournament_start": model["last_tournament_start"],
        "surface": surface,
        "players": [
            {"id": player_id, "name": players[player_id]["name"], "matches": players[player_id]["matches"],
             "elo": round(rating, 1), "win_probability": round(p, 6)}
            for player_id, rating, p in zip((first_id, second_id), ratings, (probability, 1 - probability))
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tour", choices=("ITFME", "ITFWO"))
    parser.add_argument("first", help="Player ID or exact player name")
    parser.add_argument("second", help="Player ID or exact player name")
    parser.add_argument("--surface", choices=SURFACES)
    parser.add_argument("--artifact", type=Path, default=ARTIFACT)
    args = parser.parse_args()
    try:
        model = json.loads(args.artifact.read_text(encoding="utf-8"))["tours"][args.tour]
        result = predict(model, args.first, args.second, args.surface)
    except (OSError, KeyError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
