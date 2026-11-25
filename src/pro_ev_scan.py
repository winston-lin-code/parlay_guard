"""Regular-season fixture verification for the pro pregame EV models."""

from .college_sources import canonical, parse_espn
from .inplay_scan import utc

PRO_SERIES = {"12185": "nfl", "10346": "nhl", "3": "mlb"}
NFL_ABBREVIATIONS = {"WSH": "WAS", "LAR": "LA"}


def _team(identifier, name, **names):
    return {"id": str(identifier), "name": name, **names}


def _fixture(raw_id, start, day, neutral, home, away):
    return {"espn_id": str(raw_id), "start_utc": start, "day": day, "neutral": neutral,
            "home": home, "away": away}


def fetch_pro_scoreboard(session, sport, day, now):
    if sport == "nfl":
        response = session.get("https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard",
                               params={"dates": day.replace("-", "")}, timeout=(6, 10))
        response.raise_for_status()
        payload = response.json()
        payload["events"] = [event for event in payload.get("events", [])
                             if (event.get("season") or {}).get("type") == 2]
        games = parse_espn(payload, sport, day, now)
        for game in games:
            game["day"] = day
            for side in ("home", "away"):
                team = game[side]
                team["id"] = NFL_ABBREVIATIONS.get(team["abbr"], team["abbr"])
        return games
    if sport == "nhl":
        response = session.get(f"https://api-web.nhle.com/v1/score/{day}", timeout=(6, 10))
        response.raise_for_status()
        return [_fixture(g["id"], g["startTimeUTC"], g["gameDate"], g.get("neutralSite", False),
                         *(_team(g[side]["abbrev"], g[side]["name"]["default"],
                                 abbr=g[side]["abbrev"], location="Utah" if g[side]["abbrev"] == "UTA" else "")
                           for side in ("homeTeam", "awayTeam")))
                for g in response.json()["games"] if g["gameType"] == 2 and g["gameState"] in ("FUT", "PRE")
                and g["gameScheduleState"] == "OK" and utc(g["startTimeUTC"]) > now]

    response = session.get("https://statsapi.mlb.com/api/v1/schedule",
                           params={"sportId": 1, "date": day, "gameType": "R", "hydrate": "team"},
                           timeout=(6, 10))
    response.raise_for_status()
    games = []
    for slate in response.json()["dates"]:
        for g in slate["games"]:
            if (g["gameType"] != "R" or g["status"]["codedGameState"] not in ("S", "P")
                    or g["status"].get("startTimeTBD") or utc(g["gameDate"]) <= now):
                continue
            home, away = (g["teams"][side]["team"] for side in ("home", "away"))
            teams = [_team(t["id"], t.get("teamName", ""), display_name=t["name"],
                           short_name=t.get("shortName", ""), abbr=t.get("abbreviation", ""))
                     for t in (home, away)]
            games.append(_fixture(g["gamePk"], g["gameDate"], g["officialDate"], False, *teams))
    return games


def resolve_pro_team(label, home, away):
    key = canonical(label)
    if not key:
        return None
    ids = [team["id"] for team in (home, away) if key in {
        canonical(team.get(field, "")) for field in ("name", "abbr", "display_name", "short_name", "location")
    } | {canonical((team.get("location") or "") + " " + team["name"])}]
    return ids[0] if len(ids) == 1 else None
