"""Reprice live men's college basketball and FBS contracts on verified scores."""

from functools import partial
from math import floor

from .college_inplay_model import RULES, forecast, load, score_clock
from .college_score_calibration import load_score_artifact
from .college_sources import LEAGUES, outcome_team, parse_espn
from .inplay_scan import MAX_AGE, read_json
from .nfl_inplay_scan import live_markets, scan_live


def scoreboard(sport, when, now):
    conf = LEAGUES[sport]
    payload = read_json("GET", f'https://site.api.espn.com/apis/site/v2/sports/{conf["espn"]}/scoreboard',
                        params={"dates": when.replace("-", ""), "groups": conf["group"], "limit": 500},
                        headers={"Cache-Control": "no-cache"}, max_age=MAX_AGE)
    membership = load_score_artifact(sport)
    if not membership:
        return []
    allowed = membership["teams"]
    # The pregame artifacts supply team IDs from completed Division I/FBS schedules.
    # They don't supply the live probability or score.
    payload["events"] = [event for event in payload["events"] if (event.get("season") or {}).get("type") == 2]
    games = parse_espn(payload, sport, when, now, phase="in")
    return [game for game in games if game["espn_id"]
            and all(game[side]["id"] in allowed for side in ("home", "away"))]


def candidates(fixture, game, state, artifact):
    projection = forecast(state, artifact)
    checks = {kind: values[state[3] - 1] for kind, values in artifact["holdout"].items()}
    sport = artifact["sport"]
    bounds = {"spreads": (-35.5, 35.5), "totals": (28.5, 76.5)}
    if sport == "mbb":
        periods, duration, margin, total = RULES[sport]
        home, away, remaining, _ = state
        fraction = remaining / (periods * duration)
        margin_center = floor(home - away + margin * fraction) + .5
        total_center = floor(home + away + total * fraction) + .5
        # The holdout measured thresholds within these distances of the baseline.
        bounds = {"spreads": (margin_center - 6, margin_center + 6),
                  "totals": (total_center - 12, total_center + 12)}
    return live_markets(fixture, game, projection, checks, outcome_team, line_bounds=bounds)


def scan_college_live(sports, sport, *, books_reader=None):
    return scan_live(sports, sport, load(sport), partial(scoreboard, sport),
                     lambda game: score_clock(game, sport), candidates, books_reader=books_reader)
