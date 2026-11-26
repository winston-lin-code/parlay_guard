"""Score-only win probabilities for ITF matches with evenly matched sides."""

import re
from functools import lru_cache

ITF_LEAGUES = {"ITFME", "ITFWO"}
SET_SCORE = re.compile(r"(\d{1,2})\s*-\s*(\d{1,2})")
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


def live_win_probability(score, period, *, doubles=False):
    """Return the first score column's probability, or None for unsupported states."""
    stage = re.fullmatch(r"(S|TB)([1-3])", period.strip().upper())
    if stage is None:
        return None
    segment, set_number = stage[1], int(stage[2])
    games, separator, points = score.upper().partition(":")
    sets = [SET_SCORE.fullmatch(pair.strip()) for pair in games.split(",")]
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
        return (1 + win_set) / 2
    if sets_first < sets_second:
        return win_set / 2
    return win_set


def tennis_projection(game, score):
    league = str(game.get("league") or "").upper()
    prefix = game["event"].split("-", 1)[0].upper()
    if not ({league, prefix} & ITF_LEAGUES) or score.get("phase") != "live":
        return None
    sides = game.get("title", "").split(" vs. ", 1)
    doubles = len(sides) == 2 and all(" - " in side for side in sides)
    probability = live_win_probability(score.get("score_text", ""), score.get("period_text", ""), doubles=doubles)
    if probability is None:
        return None
    return {"first_probability": probability, "second_probability": 1 - probability}
