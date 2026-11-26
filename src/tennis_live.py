"""Score-only win probabilities for ITF matches with evenly matched sides."""

import json
import math
import re
from datetime import datetime, timedelta
from functools import lru_cache
from pathlib import Path

from .tennis_elo import ARTIFACT as ELO_ARTIFACT, player_key, resolve_player

ARTIFACT = Path(__file__).resolve().parents[1] / "models" / "itf_set_breaks.json"
ELO_BREAK_ARTIFACT = ARTIFACT.with_name("itf_elo_set_breaks.json")
ITF_LEAGUES = {"ITFME", "ITFWO"}
SET_SCORE = re.compile(r"(\d{1,2})\s*-\s*(\d{1,2})")
BREAK_WINDOW = timedelta(minutes=5)
POINTS = {"0": 0, "15": 1, "30": 2, "40": 3, "AD": 4}


@lru_cache(maxsize=None)
def _race(first, second, target, margin):
    if max(first, second) >= target and abs(first - second) >= margin:
        return float(first > second)
    if first == second:
        return 0.5
    return (_race(first + 1, second, target, margin)
            + _race(first, second + 1, target, margin)) / 2


@lru_cache(maxsize=None)
def _set(first, second):
    if _completed_set(first, second):
        return float(first > second)
    if first == second:
        return 0.5
    return (_set(first + 1, second) + _set(first, second + 1)) / 2


def _completed_set(first, second):
    high, low = sorted((first, second), reverse=True)
    return (high == 6 and low <= 4) or (high == 7 and low in (5, 6))


def live_win_probability(score, period, *, doubles=False, model=None, elo_gap=None, elo_stages=None):
    """Return the first score column's probability, or None for unsupported states."""
    stage = re.fullmatch(r"(S|TB)([1-3])", period.strip().upper())
    if stage is None:
        return None
    segment, set_number = stage[1], int(stage[2])
    games, separator, points = score.upper().partition(":")
    sets = [SET_SCORE.fullmatch(re.sub(r"\(\d+(?:-\d+)?\)$", "", pair.strip()))
            for pair in games.split(",")]
    if len(sets) != set_number or any(pair is None for pair in sets):
        return None
    *completed, (first, second) = [tuple(map(int, pair.groups())) for pair in sets]
    if any(not _completed_set(*pair) for pair in completed):
        return None
    sets_first = sum(a > b for a, b in completed)
    sets_second = len(completed) - sets_first
    if max(sets_first, sets_second) == 2:
        return None

    if segment == "TB":
        tiebreak = SET_SCORE.fullmatch(points.strip())
        if tiebreak is None:
            return None
        deciding_match_tiebreak = set_number == 3 and (first, second) == (0, 0)
        if (first, second) != (6, 6) and not deciding_match_tiebreak:
            return None
        a, b = map(int, tiebreak.groups())
        target = 10 if deciding_match_tiebreak else 7
        if max(a, b) >= target and abs(a - b) >= 2:
            return None
        win_set = _race(a, b, target, 2)
    elif doubles and set_number == 3:
        # The feed may mark the opening of the match tiebreak as S3.
        if (first, second) != (0, 0) or points.strip() not in ("", "0-0"):
            return None
        win_set = 0.5
    else:
        if max(first, second) > 6 or _completed_set(first, second):
            return None
        if not separator:
            win_set = _set(first, second)
        elif (first, second) == (6, 6):
            # Wait for the feed to identify the tiebreak and its point score.
            if points.strip() != "0-0":
                return None
            win_set = 0.5
        else:
            pair = [POINTS.get(value.strip()) for value in points.split("-")]
            if len(pair) != 2 or None in pair:
                return None
            a, b = pair
            if max(a, b) == 4 and (doubles or min(a, b) != 3):
                return None
            win_game = _race(a, b, 4, 1 if doubles else 2)
            win_set = win_game * _set(first + 1, second) + (1 - win_game) * _set(first, second + 1)

    if sets_first > sets_second:
        score_probability = (1 + win_set) / 2
    elif sets_first < sets_second:
        score_probability = win_set / 2
    else:
        score_probability = win_set

    if not 0 < score_probability < 1:
        return score_probability
    margins = [a - b for a, b in completed]
    stage = str(len(completed))
    if elo_gap is not None and elo_stages and stage in elo_stages:
        weights = elo_stages[stage]["weights"]
        fitted_log_odds = sum(weight * value for weight, value in zip(weights, [elo_gap, *margins]))
    elif model and stage in model["set_breaks"]:
        weights = model["set_breaks"][stage]["weights"]
        fitted_log_odds = sum(weight * margin for weight, margin in zip(weights, margins))
    else:
        return score_probability
    boundary_log_odds = math.log(3) * (sets_first - sets_second)
    # Apply the historical shift in log odds to the live scoring baseline.
    adjusted_odds = score_probability / (1 - score_probability) * math.exp(fitted_log_odds - boundary_log_odds)
    return adjusted_odds / (1 + adjusted_odds)


@lru_cache(maxsize=1)
def load_models():
    try:
        artifact = json.loads(ARTIFACT.read_text(encoding="utf-8"))
        return artifact["tours"] if artifact["version"] == 1 else {}
    except (OSError, ValueError, KeyError, TypeError):
        return {}


@lru_cache(maxsize=1)
def load_elo_break_models():
    try:
        fitted = json.loads(ELO_BREAK_ARTIFACT.read_text(encoding="utf-8"))
        ratings = json.loads(ELO_ARTIFACT.read_text(encoding="utf-8"))
        if fitted["version"] == ratings["version"] == 1:
            return fitted["tours"], ratings["tours"]
    except (OSError, ValueError, KeyError, TypeError):
        pass
    return {}, {}


def rating_gap(players, sides, score_order):
    # Require identifiable score columns before attaching player ratings.
    labels = score_order.split(" / ")
    if len(labels) != 2 or {player_key(label) for label in labels} != {player_key(side) for side in sides}:
        return None
    try:
        first, second = (players[resolve_player(players, label)] for label in labels)
    except ValueError:
        return None
    return (first["overall"] - second["overall"]) / 400


def observe_first_set_break(previous, current, parse_time):
    old_period, period = (state.get("period_text", "").strip().upper() for state in (previous, current))
    if period not in ("S2", "TB2") or current.get("phase") != "live":
        return None
    if old_period in ("S2", "TB2"):
        return previous.get("first_set_break_utc")
    if old_period not in ("S1", "TB1") or previous.get("phase") != "live":
        return None
    games = current.get("score_text", "").split(",")
    if len(games) != 2 or games[1].strip() not in ("0-0", "0-0:0-0"):
        return None
    old_time, time = (parse_time(state.get("score_utc")) for state in (previous, current))
    if old_time and time and timedelta(0) < time - old_time <= timedelta(seconds=60):
        return time.isoformat()
    return None


def first_set_checkpoint(score, now):
    period = score.get("period_text", "").strip().upper()
    if period in ("S2", "TB2") and now and score.get("first_set_break_utc"):
        observed = datetime.fromisoformat(score["first_set_break_utc"])
        if timedelta(0) <= now - observed <= BREAK_WINDOW:
            return "Within 5 min of observed set break"
    if period == "TB1":
        return "Late first set · tiebreak"
    if period == "S1":
        games = SET_SCORE.fullmatch(score.get("score_text", "").split(":", 1)[0].strip())
        if games:
            high, low = sorted(map(int, games.groups()), reverse=True)
            if (high == 5 and low < 5) or (high, low) == (6, 5):
                return "Late first set · one game from set"
    return None


def tennis_projection(game, score, now=None):
    league = str(game.get("league") or "").upper()
    prefix = game["event"].split("-", 1)[0].upper()
    if not ({league, prefix} & ITF_LEAGUES) or score.get("phase") != "live":
        return None
    sides = game.get("title", "").split(" vs. ", 1)
    doubles = len(sides) == 2 and all(" - " in side or " / " in side for side in sides)
    code = prefix if prefix in ITF_LEAGUES else league
    model = load_models().get(code) if len(sides) == 2 and not doubles else None
    period = score.get("period_text", "")
    stages, ratings = load_elo_break_models()
    elo = stages.get(code)
    gap = rating_gap(ratings[code]["players"], sides, score.get("score_order", "")) if elo and code in ratings and model else None
    probability = live_win_probability(score.get("score_text", ""), period, doubles=doubles,
                                       model=model, elo_gap=gap, elo_stages=elo)
    if probability is None:
        return None
    return {"first_probability": probability, "second_probability": 1 - probability,
            "elo_prior": gap is not None, "checkpoint": first_set_checkpoint(score, now),
            "historical_prior": bool(model and period.strip()[-1:] in ("2", "3"))}
