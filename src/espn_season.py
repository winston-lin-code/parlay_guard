"""Build team-game history from date-specific ESPN NBA final scores."""

import json
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta
from pathlib import Path
from tempfile import NamedTemporaryFile
from zoneinfo import ZoneInfo

import pandas as pd

ET = ZoneInfo("America/New_York")


def final_team_games(events, day, season_year, normalize_team):
    rows = []
    for event in events:
        season = event["season"]
        if int(season["year"]) != season_year or int(season["type"]) != 2:
            continue
        played = datetime.fromisoformat(event["date"].replace("Z", "+00:00"))
        if played.astimezone(ET).date() != day:
            continue
        competition, = event["competitions"]
        status = competition["status"]["type"]
        if status["state"] != "post" or not status["completed"]:
            continue
        teams = {entry["homeAway"]: entry for entry in competition["competitors"]}
        home, away = teams["home"], teams["away"]
        scores = [int(entry["score"]) for entry in (home, away)]
        if min(scores) <= 0 or scores[0] == scores[1]:
            raise ValueError(f"Invalid ESPN final score for game {event['id']}")
        abbreviations = [normalize_team(entry["team"]["abbreviation"])
                         for entry in (home, away)]
        for index, entry in enumerate((home, away)):
            rows.append({
                "GAME_ID": f"ESPN-{event['id']}",
                "GAME_DATE": day.isoformat(),
                "TEAM_ID": str(entry["team"]["id"]),
                "TEAM_ABBREVIATION": abbreviations[index],
                "MATCHUP": f"{abbreviations[1]} @ {abbreviations[0]}",
                "WL": "W" if scores[index] > scores[1 - index] else "L",
                "PTS": scores[index],
            })
    return rows


def fetch_espn_season(season: str, today: date, cache_dir: Path, fetch_scoreboard,
                     normalize_team) -> pd.DataFrame:
    start_year = int(season.split("-")[0])
    # NBA regular-season games begin in October.
    start = date(start_year, 10, 1)
    end = min(today, date(start_year + 1, 6, 30))
    cache_dir.mkdir(parents=True, exist_ok=True)

    def fetch_day(day: date) -> list[dict]:
        cached = cache_dir / f"{day.isoformat()}.json"
        settled_day = day < today - timedelta(days=1)
        if settled_day and cached.is_file():
            return json.loads(cached.read_text(encoding="utf-8"))
        records = final_team_games(fetch_scoreboard(day), day, start_year + 1, normalize_team)
        if settled_day:
            with NamedTemporaryFile(dir=cache_dir, mode="w", encoding="utf-8", delete=False) as staged:
                temporary = Path(staged.name)
                try:
                    json.dump(records, staged)
                    staged.close()
                    temporary.replace(cached)
                finally:
                    temporary.unlink(missing_ok=True)
        return records

    days = (start + timedelta(days=i) for i in range((end - start).days + 1))
    with ThreadPoolExecutor(max_workers=6) as worker:
        rows = [row for day_rows in worker.map(fetch_day, days) for row in day_rows]
    if not rows:
        raise ValueError(f"ESPN NBA results feed had no final games for {season}")
    return pd.DataFrame(rows)
