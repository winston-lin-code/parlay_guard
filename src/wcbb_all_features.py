"""Score WCBB moneylines using the frozen all-feature logistic model."""

from bisect import bisect_left, bisect_right
from datetime import timedelta
from decimal import Decimal
import json
from math import exp, isfinite, sqrt
from pathlib import Path

from .inplay_scan import utc

DEFAULT_PATH = Path("data/wcbb_all_features.json")
MODEL_ID = "logistic__combined_plus_all"
SEASON = 2027
FEATURES = ("elo_diff", "margin", "seconds_remaining", "pace_adjusted_margin", "home_court",
            "neutral_site", "home_offense", "away_offense", "home_defense", "away_defense",
            "home_sos", "away_sos")


def clock_seconds(value):
    if not isinstance(value, str):
        return None
    try:
        parts = value.split(":")
        return sum(float(part) * 60 ** place for place, part in enumerate(reversed(parts))) if len(parts) <= 2 else None
    except ValueError:
        return None


def confirmed_grid(summary, game, quoted, delay):
    """Use the latest elapsed one-minute checkpoint confirmed at the quote."""
    if str(summary.get("header", {}).get("id")) != game["id"] or game["period"] > 4:
        return None
    times, elapsed, states, periods = [], [], [], []
    available = game["tip"]
    halftime = game["checkpoint"] == "Halftime"
    halftime_at = None
    for play in summary.get("plays", []):
        at = utc(play.get("wallclock"))
        period = (play.get("period") or {}).get("number")
        clock = clock_seconds((play.get("clock") or {}).get("displayValue"))
        score = play.get("homeScore"), play.get("awayScore")
        if (at is None or type(period) is not int or not 1 <= period <= 4
                or clock is None or not 0 <= clock <= 600 or any(type(value) is not int for value in score)):
            continue
        available = max(available, at)
        if available > quoted:
            break
        mark = period * 600 - clock
        if elapsed and mark < elapsed[-1]:
            return None
        times.append(available)
        elapsed.append(mark)
        states.append(score)
        periods.append(period)
        if period == 2 and clock == 0 and "end of" in (play.get("text") or "").lower():
            halftime_at = available
    confirmed = bisect_right(times, quoted - timedelta(seconds=delay)) - 1
    expected = game["scores"]["home"], game["scores"]["away"]
    if confirmed < 0 or states[-1] != expected or periods[-1] != game["period"]:
        return None
    grid = int(elapsed[confirmed] // 60) * 60
    if not 60 <= grid <= 2340 or elapsed[-1] - grid >= 60:
        return None
    if halftime:
        if halftime_at is None or halftime_at > quoted - timedelta(seconds=delay) or grid != 1200:
            return None
    elif (quoted - times[-1]).total_seconds() > 120 or grid == 1200 and game["period"] == 2:
        return None

    start = bisect_left(elapsed, grid)
    prior = start - 1
    if grid % 600 == 0:
        while prior < confirmed and elapsed[prior + 1] == grid and periods[prior + 1] == grid // 600:
            prior += 1
        # The model includes final zero-clock plays at a period boundary.
        if prior < len(times) - 1 and elapsed[prior + 1] == grid and periods[prior + 1] == grid // 600:
            return None
    if prior < 0 or states[prior] != expected:
        return None
    return 2400 - grid, times[confirmed]


class AllFeatures:
    def __init__(self, artifact):
        if (artifact.get("model") != MODEL_ID or artifact.get("version") != 1 or artifact.get("season") != SEASON
                or artifact.get("features") != list(FEATURES)):
            raise ValueError("Unexpected WCBB model format or context season")
        self.terms = list(zip(*(artifact[key] for key in ("median", "mean", "scale", "weights")), strict=True))
        self.intercept = artifact["intercept"]
        if (len(self.terms) != len(FEATURES) or not isfinite(self.intercept)
                or any(not all(map(isfinite, row)) or row[2] <= 0 for row in self.terms)):
            raise ValueError("Invalid WCBB model coefficients")
        self.members = {str(value) for value in artifact["members"]}
        self.history = artifact["context"]
        self.elo = artifact["historical_elo"]
        self.venues = artifact["home_venues"]
        self.generated = utc(artifact.get("generated_utc"))
        if not self.members or not self.history or self.generated is None:
            raise ValueError("Missing WCBB team context")
        self.times = {team: [row[0] for row in rows] for team, rows in self.history.items()}
        for team, rows in self.history.items():
            if (self.times[team] != sorted(self.times[team])
                    or any(len(row) != 5 or not isfinite(row[0]) or row[0] >= self.generated.timestamp()
                           or any(value is not None and not isfinite(value) for value in row[1:]) for row in rows)):
                raise ValueError("Invalid WCBB team context chronology")

    @classmethod
    def load(cls, path):
        return cls(json.loads(path.read_text(encoding="utf-8")))

    def team(self, team, tip):
        index = bisect_left(self.times.get(team, ()), tip.timestamp()) - 1
        if index >= 0:
            return self.history[team][index][1:]
        return self.elo.get(team, 1500.0), None, None, None

    def forecast(self, summary, game, quoted, *, delay):
        tip = game["tip"]
        if tip.year + (tip.month >= 7) != SEASON:
            return None
        home, away = (str(game["teams"][side]["id"]) for side in ("home", "away"))
        if home == away or not {home, away} <= self.members:
            return None
        grid = confirmed_grid(summary, game, quoted, delay)
        if grid is None:
            return None
        seconds, score_at = grid
        h_elo, h_off, h_def, h_sos = self.team(home, tip)
        a_elo, a_off, a_def, a_sos = self.team(away, tip)
        margin = game["scores"]["home"] - game["scores"]["away"]
        neutral = game.get("neutral_site")
        venue, home_venue = game.get("venue"), self.venues.get(home)
        home_court = (0 if neutral is True else int(venue == home_venue)
                      if neutral is False and venue and home_venue else None)
        values = (h_elo - a_elo, margin, seconds, margin / sqrt(seconds / 60), home_court, neutral,
                  h_off, a_off, h_def, a_def, h_sos, a_sos)
        logit = self.intercept + sum(((median if value is None else value) - mean) / scale * weight
                                    for value, (median, mean, scale, weight) in zip(values, self.terms))
        probability = 1 / (1 + exp(-logit)) if logit >= 0 else exp(logit) / (1 + exp(logit))
        return Decimal(str(probability)), score_at
