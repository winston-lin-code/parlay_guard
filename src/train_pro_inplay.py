"""Train MLB inning-break and NHL regulation win models from official results."""

import argparse
from calendar import monthrange
from concurrent.futures import ThreadPoolExecutor
from datetime import date
import json
from pathlib import Path
from random import Random
from time import sleep

import numpy as np
import requests
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss

from .pro_inplay_model import MODEL_DIR, VERSION, baseball_state, features
from .pro_live_scores import MLB, NHL

SEASONS = (2023, 2024, 2025)
MLB_FIELDS = ("gamePk,gameData,status,abstractGameState,liveData,linescore,scheduledInnings,plays,allPlays,"
              "about,inning,isTopInning,isComplete,count,outs,matchup,postOnFirst,postOnSecond,postOnThird,"
              "id,result,homeScore,awayScore")


def download(cache, label, url, **params):
    path = cache / f"{label}.json"
    if path.exists():
        return json.loads(path.read_bytes())
    for attempt in range(3):
        try:
            response = requests.get(url, params=params, timeout=(8, 15))
            response.raise_for_status()
            payload = response.json()
            path.write_bytes(response.content)
            return payload
        except (requests.RequestException, ValueError):
            if attempt == 2:
                raise
            sleep(1 + attempt)


def mlb_games(cache, season, pool):
    def month(m):
        return download(cache, f"mlb-{season}-{m}", f"{MLB}/v1/schedule", sportId=1, gameType="R",
                        startDate=date(season, m, 1), endDate=date(season, m, monthrange(season, m)[1]),
                        hydrate="linescore")["dates"]
    for slates in pool.map(month, range(3, 10 + 1)):
        for slate in slates:
            yield from slate["games"]


def mlb_rows(game):
    line = game.get("linescore") or {}
    innings = line.get("innings") or []
    if (game.get("gameType") != "R" or game["status"].get("abstractGameState") != "Final"
            or line.get("scheduledInnings") != 9 or len(innings) < 9):
        return []
    final_away, final_home = (line["teams"][s].get("runs") for s in ("away", "home"))
    if None in (final_away, final_home) or final_home == final_away:
        return []
    home = away = 0
    snapshots = []
    for index, inning in enumerate(innings, 1):
        if inning["num"] != index or "runs" not in inning["away"]:
            return []
        away += inning["away"]["runs"]
        if index < 9 or (index == 9 and home <= away):
            snapshots.append(baseball_state((away, home), index, "middle"))
        if "runs" not in inning["home"]:
            if index < 9:
                return []
        else:
            home += inning["home"]["runs"]
        if index < 9:
            snapshots.append(baseball_state((away, home), index, "end"))
    return [(game["gamePk"], snapshot, final_home > final_away) for snapshot in snapshots] \
        if (home, away) == (final_home, final_away) else []


def mlb_play_rows(payload, inning_breaks):
    if not payload or payload.get("gamePk") != inning_breaks[0][0]:
        return []
    game_id, _, home_won = inning_breaks[0]
    plays = payload["liveData"]["plays"]["allPlays"]
    rows = list(inning_breaks)
    for play in plays[:-1]:
        state, tally = play["about"], play["result"]
        if not state.get("isComplete") or type(state.get("isTopInning")) is not bool:
            continue
        matchup = play.get("matchup")
        if not isinstance(matchup, dict) or not all(type(tally.get(key)) is int for key in ("awayScore", "homeScore")):
            continue
        bases = [field in matchup for field in ("postOnFirst", "postOnSecond", "postOnThird")]
        sample = baseball_state((tally["awayScore"], tally["homeScore"]), state["inning"],
                                "top" if state["isTopInning"] else "bottom", play["count"].get("outs"), bases)
        if sample:
            rows.append((game_id, sample, home_won))
    return rows if len(rows) > len(inning_breaks) else []


def get_mlb_plays(cache, game_id):
    try:
        return download(cache, f"mlb-plays-{game_id}", f"{MLB}/v1.1/game/{game_id}/feed/live", fields=MLB_FIELDS)
    except (requests.RequestException, ValueError) as exc:
        print(f"MLB {game_id} unavailable ({type(exc).__name__})", flush=True)
        return None


def nhl_games(cache, season, pool, sample_size):
    season_id = f"{season - 1}{season}"
    index = download(cache, f"nhl-{season}-index", "https://api.nhle.com/stats/rest/en/game",
                     cayenneExp=f"season={season_id} and gameType=2", limit=-1)["data"]
    game_ids = sorted(g["id"] for g in index if g["gameStateId"] == 7)
    ids = Random(1729).sample(game_ids, min(sample_size, len(game_ids)))

    def get_game(game_id):
        try:
            return download(cache, f"nhl-{game_id}", f"{NHL}/gamecenter/{game_id}/landing")
        except (requests.RequestException, ValueError) as exc:
            print(f"NHL {game_id} unavailable ({type(exc).__name__})", flush=True)
            return None
    yield from pool.map(get_game, ids)


def nhl_rows(game):
    if not game or game["gameState"] not in ("OFF", "FINAL") or game.get("gameType") != 2:
        return []
    final_away, final_home = (game[side]["score"] for side in ("awayTeam", "homeTeam"))
    if final_home == final_away:
        return []
    regulation, previous = [], (0, 0)
    for period in (game.get("summary") or {}).get("scoring") or []:
        frame = period["periodDescriptor"]
        if frame["periodType"] == "SO":
            continue
        for goal in period["goals"]:
            tally = goal["homeScore"], goal["awayScore"]
            if (tally[0] - previous[0], tally[1] - previous[1]) not in ((1, 0), (0, 1)):
                return []
            previous = tally
            if frame["periodType"] == "REG":
                minutes, seconds = map(int, goal["timeInPeriod"].split(":"))
                regulation.append(((frame["number"] - 1) * 1200 + minutes * 60 + seconds, *tally))
    finishing_gap = (final_home - previous[0], final_away - previous[1])
    finish = (game.get("periodDescriptor") or {}).get("periodType")
    if finishing_gap != (0, 0) and not (finish == "SO" and finishing_gap in ((1, 0), (0, 1))):
        return []
    home = away = goal_index = 0
    snapshots = []
    for elapsed in range(60, 3600, 60):
        while goal_index < len(regulation) and regulation[goal_index][0] <= elapsed:
            _, home, away = regulation[goal_index]
            goal_index += 1
        snapshots.append((game["id"], (home - away, (3600 - elapsed) / 3600, 0), final_home > final_away))
    return snapshots


def fit(league, splits):
    train, calibration, holdout = splits
    game_ids = [{row[0] for row in split} for split in splits]
    if min(map(len, game_ids)) < 100:
        raise ValueError("Each chronological partition needs at least 100 usable games")
    if len(set.union(*game_ids)) != sum(map(len, game_ids)):
        raise ValueError("A game appears in multiple season splits")
    features_for = lambda split: np.array([features(state) for _, state, _ in split])
    outcomes = lambda split: np.array([won for _, _, won in split])
    x, y = features_for(train), outcomes(train)
    baseline = LogisticRegression().fit([[state[0]] for _, state, _ in train], y)
    model = LogisticRegression().fit(x, y)
    calibrator = LogisticRegression(C=100).fit(
        model.decision_function(features_for(calibration)).reshape(-1, 1), outcomes(calibration)
    )
    test_x, test_y = features_for(holdout), outcomes(holdout)
    probabilities = calibrator.predict_proba(model.decision_function(test_x).reshape(-1, 1))[:, 1]
    baseline_p = baseline.predict_proba([[state[0]] for _, state, _ in holdout])[:, 1]
    metrics = {"games": len({row[0] for row in holdout}), "observations": len(test_y),
               "brier": float(brier_score_loss(test_y, probabilities)),
               "log_loss": float(log_loss(test_y, probabilities)),
               "score_only_brier": float(brier_score_loss(test_y, baseline_p)),
               "score_only_log_loss": float(log_loss(test_y, baseline_p))}
    calibration_bins = []
    for lower in np.arange(0, 1, .1):
        mask = (probabilities >= lower) & (probabilities < lower + .1)
        if mask.any():
            calibration_bins.append({"observations": int(mask.sum()), "predicted": float(probabilities[mask].mean()),
                                     "observed": float(test_y[mask].mean())})
    return {"version": VERSION, "league": league, "seasons": SEASONS,
            "train_games": len({row[0] for row in train}), "calibration_games": len({row[0] for row in calibration}),
            "weights": model.coef_[0].tolist(), "intercept": float(model.intercept_[0]),
            "calibration_slope": float(calibrator.coef_[0, 0]),
            "calibration_intercept": float(calibrator.intercept_[0]), "holdout": metrics,
            "holdout_calibration_bins": calibration_bins}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("league", choices=("mlb", "nhl"))
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--mlb-games-per-season", type=int, default=240)
    parser.add_argument("--nhl-games-per-season", type=int, default=192)
    args = parser.parse_args()
    args.cache.mkdir(parents=True, exist_ok=True)
    splits = []
    with ThreadPoolExecutor(max_workers=6) as pool:
        for season in SEASONS:
            if args.league == "mlb":
                state_rows = {game["gamePk"]: rows for game in mlb_games(args.cache, season, pool)
                              if (rows := mlb_rows(game))}
                ids = Random(1729).sample(sorted(state_rows), min(args.mlb_games_per_season, len(state_rows)))
                rows = [row for game_id, payload in zip(ids, pool.map(lambda g: get_mlb_plays(args.cache, g), ids))
                        for row in mlb_play_rows(payload, state_rows[game_id])]
            else:
                rows = [row for game in nhl_games(args.cache, season, pool, args.nhl_games_per_season)
                        for row in nhl_rows(game)]
            splits.append(rows)
            print(season, len({row[0] for row in rows}), "games", flush=True)
    artifact = fit(args.league, splits)
    artifact["requested_games_per_season"] = (
        args.nhl_games_per_season if args.league == "nhl" else args.mlb_games_per_season
    )
    path = MODEL_DIR / f"{args.league}_live.json"
    path.write_text(json.dumps(artifact, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(artifact["holdout"], indent=2))


if __name__ == "__main__":
    main()
