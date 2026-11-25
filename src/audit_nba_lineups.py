"""Estimate how NBA margins differ when established starters miss games."""

import argparse
import json
import re
from collections import Counter, defaultdict, deque

import numpy as np
import pandas as pd

from .audit_nfl_qb_changes import regression, team_effects

COLUMNS = ("game_id", "season", "season_type", "game_date_time", "team_id", "home_away",
           "team_score", "athlete_id", "starter", "minutes", "did_not_play", "reason")
INJURY = re.compile(r"injur|sprain|strain|fractur|surg|concussion|contusion|\btear\b|\btorn\b|ruptur|soreness", re.I)
VARIABLES = ("injury_absent", "other_absent", "moved_to_bench")


def status(box, recent):
    starts, minutes = Counter(), Counter()
    for _, lineup in recent:
        for player, (starter, played) in lineup.items():
            starts[player] += starter
            minutes[player] += played
    usual = {player for player, count in starts.items() if count >= 7 and minutes[player] >= 280}
    current = {player.athlete_id: player for player in box.itertuples()}
    count = dict.fromkeys(VARIABLES, 0)
    for player in usual:
        entry = current.get(player)
        if entry is None or entry.did_not_play or not entry.minutes:
            key = "injury_absent" if entry and INJURY.search(str(entry.reason)) else "other_absent"
            count[key] += 1
        elif not entry.starter:
            count["moved_to_bench"] += 1
    return count


def games_rows(frame):
    teams = defaultdict(lambda: deque(maxlen=10))
    rows = []
    for _, box in frame.sort_values("game_date_time").groupby("game_id", sort=False):
        sides = {side: data for side, data in box.groupby("home_away") if data.team_id.nunique() == 1}
        if set(sides) != {"home", "away"}:
            continue
        home, away = sides["home"], sides["away"]
        if any(side.team_score.nunique() != 1 or side.starter.sum() != 5 for side in (home, away)):
            continue
        season, h_id, a_id = int(home.season.iat[0]), home.team_id.iat[0], away.team_id.iat[0]
        h_hist, a_hist = teams[season, h_id], teams[season, a_id]
        margin = int(home.team_score.iat[0] - away.team_score.iat[0])
        if len(h_hist) == len(a_hist) == 10:
            h, a = status(home, h_hist), status(away, a_hist)
            prior = (sum(row[0] for row in h_hist) - sum(row[0] for row in a_hist)) / 10
            rows.append({"season": season, "home": h_id, "away": a_id, "margin": margin, "recent_margin": prior,
                         **{name: h[name] - a[name] for name in VARIABLES},
                         "injury_absences": h["injury_absent"] + a["injury_absent"]})
        for side, recent, diff in ((home, h_hist, margin), (away, a_hist, -margin)):
            recent.append((diff, {r.athlete_id: (r.starter, r.minutes or 0) for r in side.itertuples()}))
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("parquets", nargs="+")
    args = parser.parse_args()
    frames = [pd.read_parquet(path, columns=list(COLUMNS)) for path in args.parquets]
    frame = pd.concat(frames, ignore_index=True).drop_duplicates(["game_id", "team_id", "athlete_id"])
    frame = frame[(frame.season_type == 2) & frame.team_score.notna() & frame.athlete_id.notna()].copy()
    frame["minutes"] = frame.minutes.fillna(0)
    rows = games_rows(frame)
    x = np.column_stack([rows[list(VARIABLES)], np.ones(len(rows)), rows.recent_margin, team_effects(rows)])
    estimate = regression(x, {"scoring_margin": rows.margin}, VARIABLES)
    print(json.dumps({"games": len(rows), "games_with_labeled_starter_injury": int((rows.injury_absences > 0).sum()),
                      **estimate}, indent=2))


if __name__ == "__main__":
    main()
