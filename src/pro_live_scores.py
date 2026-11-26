"""Read-only Polymarket US score and game-state verification for MLB and NHL."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
import json
import re
from time import monotonic, sleep

import requests

from .inplay_scan import read_json, utc
from .predict_today import ET
from .us_wager import SLUG, name, public, team_names

LEAGUES = ("mlb", "nhl")
SPORTS = {"mlb": "baseball", "nhl": "hockey"}
NHL = "https://api-web.nhle.com/v1"
MLB = "https://statsapi.mlb.com/api"
START_TOLERANCE = timedelta(minutes=30)
DATA_ERRORS = (requests.RequestException, ValueError, KeyError, TypeError)


def now():
    return datetime.now(timezone.utc)


def league_json(url, **params):
    return read_json("GET", url, max_age=20, params=params, headers={"Cache-Control": "no-cache"})


def league_listings(league):
    for offset in range(0, 1000, 50):
        page = public(f"/v2/leagues/{league}/events?limit=50&offset={offset}", fresh_score=True)["events"]
        for event in page:
            slug = event.get("slug") or ""
            if SLUG.fullmatch(slug) and slug.startswith(league + "-"):
                yield slug, event, now()
        if len(page) < 50:
            return
    raise ValueError(f"{league.upper()} market catalog exceeded its page limit")


def market_events(league):
    return {slug: (event, received) for slug, event, received in league_listings(league)
            if event.get("live") is True}


def official_fixtures(league, day):
    if league == "nhl":
        games = league_json(f"{NHL}/score/{day}")["games"]
        return [(g["id"], utc(g["startTimeUTC"]), g, now()) for g in games if g.get("gameScheduleState") == "OK"]
    slates = league_json(f"{MLB}/v1/schedule", sportId=1, date=day, hydrate="team")["dates"]
    return [(g["gamePk"], utc(g["gameDate"]), g, now())
            for slate in slates for g in slate["games"] if not g["status"].get("startTimeTBD")]


def market_teams(league, event):
    outcome_pairs = [m.get("marketSides") or [] for m in event.get("markets") or []
                     if m.get("marketType") == "moneyline"
                     and m.get("sportsMarketType") == f"{SPORTS[league]}_team_full_game_winner"]
    if len(outcome_pairs) != 1 or len(outcome_pairs[0]) != 2:
        return {}
    sides = {(side.get("team") or {}).get("ordering"): team_names(side) for side in outcome_pairs[0]}
    return sides if set(sides) == {"home", "away"} and all(sides.values()) else {}


def official_teams(league, game):
    clubs = {}
    for side in ("home", "away"):
        if league == "nhl":
            team = game[side + "Team"]
            location = (team.get("placeName") or {}).get("default", "")
            nickname = (team.get("name") or team.get("commonName") or {}).get("default", "")
            values = [team.get("abbrev"), nickname, f"{location} {nickname}"]
        else:
            team = game["teams"][side]["team"]
            values = [team.get(field) for field in ("abbreviation", "name", "teamName", "shortName")]
        clubs[side] = {name(value) for value in values} - {""}
    return clubs


def fixture_matches(league, event, fixtures):
    start = utc(event.get("startTime"))
    clubs = market_teams(league, event)
    if not start or not clubs:
        return []
    matches = []
    for row in fixtures:
        _, official_start, game, _ = row
        if not official_start or abs(start - official_start) > START_TOLERANCE:
            continue
        sides = official_teams(league, game)
        direct = all(clubs[side] & sides[side] for side in sides)
        swapped = bool(clubs["away"] & sides["home"] and clubs["home"] & sides["away"])
        if direct and not swapped:
            matches.append(row)
    return matches


def scores(away, home):
    return [away, home] if all(type(value) is int and value >= 0 for value in (away, home)) else None


def nhl_state(game):
    status = game.get("gameState")
    period = (game.get("periodDescriptor") or {}).get("number")
    clock = game.get("clock") or {}
    phase = "final" if status in ("OFF", "FINAL") else "unavailable"
    if status in ("LIVE", "CRIT") and period and type(clock.get("inIntermission")) is bool:
        phase = "intermission" if clock["inIntermission"] else "in_play"
    return {
        "score_away_home": scores(game["awayTeam"].get("score"), game["homeTeam"].get("score")),
        "phase": phase, "period": period, "clock": clock.get("timeRemaining"),
        "seconds_remaining": clock.get("secondsRemaining"),
        "period_type": (game.get("periodDescriptor") or {}).get("periodType"),
    }


def mlb_state(payload):
    status = payload["gameData"]["status"]
    live = payload["liveData"]
    line = live["linescore"]
    teams = line["teams"]
    inning_state = (line.get("inningState") or "").lower()
    phase = "unavailable"
    if status.get("abstractGameState") == "Final":
        phase = "final"
    elif status.get("detailedState") == "In Progress" and inning_state in ("top", "middle", "bottom", "end"):
        phase = inning_state
    stamp = (payload.get("metaData") or {}).get("timeStamp")
    published = datetime.strptime(stamp, "%Y%m%d_%H%M%S").replace(tzinfo=timezone.utc) if stamp else None
    plays = (live.get("plays") or {}).get("allPlays") or []
    offense = line.get("offense") or {}
    bases = [bool(offense.get(base)) for base in ("first", "second", "third")] if offense.get("team") else None
    return {
        "score_away_home": scores(teams["away"].get("runs"), teams["home"].get("runs")),
        "phase": phase, "period": line.get("currentInning"), "outs": line.get("outs"), "bases": bases,
        "scheduled_innings": line.get("scheduledInnings"),
        "source_updated_at": published.isoformat() if published else None,
        "last_completed_play_at": next((p["about"].get("endTime") for p in reversed(plays)
                                       if p.get("about", {}).get("isComplete") and p["about"].get("endTime")), None),
    }


def market_state(league, event):
    state = event.get("eventState") or {}
    if state.get("type") != SPORTS[league]:
        return {}
    raw_score = re.fullmatch(r"(\d+)\s*-\s*(\d+)", str(state.get("score") or ""))
    raw_period = " ".join(str(state.get("period") or "").upper().split())
    phase, period, outs, bases = "unavailable", None, None, None
    if state.get("ended") is True and raw_period in ("FT", "VFT", "FINAL", "F"):
        phase = "final"
    elif state.get("live") is True:
        if league == "nhl":
            match = re.fullmatch(r"(END\s+)?P([1-3])", raw_period)
            if match:
                phase, period = ("intermission" if match[1] else "in_play"), int(match[2])
        else:
            baseball = state.get("baseballState") or {}
            outs = baseball.get("outs")
            occupancy = [baseball.get(field) for field in ("onFirst", "onSecond", "onThird")]
            if all(type(value) is bool for value in occupancy):
                bases = occupancy
            match = re.fullmatch(r"(TOP|T|BOTTOM|BOT|B|MIDDLE|MID|M|END|E)\s*(\d{1,2})", raw_period)
            if match:
                labels = {"T": "top", "TOP": "top", "B": "bottom", "BOT": "bottom", "BOTTOM": "bottom",
                          "MID": "middle", "MIDDLE": "middle", "M": "middle", "END": "end", "E": "end"}
                phase, period = labels[match[1]], int(match[2])
            elif raw_period.isdecimal() and baseball.get("inningHalf") in ("T", "B"):
                phase, period = ("top" if baseball["inningHalf"] == "T" else "bottom"), int(raw_period)
    stamp = utc(state.get("updatedAt"))
    return {
        "score_away_home": list(map(int, raw_score.groups())) if raw_score else None,
        "phase": phase, "period": period, "period_text": raw_period, "outs": outs, "bases": bases,
        "source_updated_at": stamp.isoformat() if stamp else None,
    }


def agreement(official, market):
    left, right = official.get("score_away_home"), market.get("score_away_home")
    if left is None or right is None:
        return "score_unavailable"
    if left != right:
        return "score_mismatch"
    if "unavailable" in (official["phase"], market["phase"]):
        return "score_agrees_phase_unverified"
    if official["phase"] != market["phase"]:
        return "phase_mismatch"
    if official["phase"] == "final":
        return "final_score_agrees"
    if official.get("period") != market.get("period"):
        return "period_mismatch"
    if all(type(s.get("outs")) is int for s in (official, market)) and official["outs"] != market["outs"]:
        return "outs_mismatch"
    if official["phase"] in ("top", "bottom") and all(s.get("bases") is not None for s in (official, market)):
        if official["bases"] != market["bases"]:
            return "base_runners_mismatch"
    return "score_and_phase_agree"


def received_state(state, received):
    published = utc(state.get("source_updated_at"))
    age = (received - published).total_seconds() if published else None
    return {**state, "received_at": received.isoformat(), "state_age_seconds": round(age, 2) if age is not None else None}


def assess(league, events):
    dates = {start.astimezone(ET).date() for event, _ in events.values() if (start := utc(event.get("startTime")))}
    fixtures = {}
    for day in sorted(dates | {day - timedelta(days=1) for day in dates}):
        fixtures.update((row[0], row) for row in official_fixtures(league, day.isoformat()))
    results = []
    for slug, (event, market_received) in events.items():
        matched = fixture_matches(league, event, fixtures.values())
        if len(matched) != 1:
            results.append({"league": league, "event": slug, "status": "unmatched_fixture",
                            "matching_official_games": len(matched)})
            continue
        game_id, _, game, official_received = matched[0]
        if league == "mlb":
            official = mlb_state(league_json(f"{MLB}/v1.1/game/{game_id}/feed/live"))
            official_received = now()
        else:
            official = nhl_state(game)
        market = market_state(league, event)
        results.append({
            "league": league, "event": slug, "official_game_id": game_id, "status": agreement(official, market),
            "official": received_state(official, official_received),
            "polymarket_us": received_state(market, market_received),
        })
    return {"league": league, "listings_checked": len(events), "games": results}


def poll(league, event):
    try:
        if event:
            listing = public(f"/v1/events/slug/{event}", fresh_score=True)["event"]
            if listing.get("slug") != event:
                raise ValueError("Polymarket US returned a different fixture")
            events = {event: (listing, now())}
        else:
            events = market_events(league)
        return assess(league, events)
    except DATA_ERRORS as exc:
        return {"league": league, "status": "feed_unavailable", "error": str(exc)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--leagues", nargs="+", choices=LEAGUES)
    parser.add_argument("--event", help="Compare a specific MLB or NHL listing, including a completed game")
    parser.add_argument("--watch", action="store_true", help="Keep checking live listings")
    parser.add_argument("--interval", type=int, default=5, help="Seconds between polls, minimum 3")
    args = parser.parse_args()
    if args.interval < 3:
        parser.error("--interval must be at least 3")
    if args.event:
        league = args.event.split("-", 1)[0]
        if not SLUG.fullmatch(args.event) or league not in LEAGUES or args.leagues:
            parser.error("--event requires an MLB or NHL event slug and selects its league")
        leagues = [league]
    else:
        leagues = list(dict.fromkeys(args.leagues or LEAGUES))
    try:
        with ThreadPoolExecutor(max_workers=len(leagues)) as pool:
            while True:
                started = monotonic()
                for result in pool.map(lambda league: poll(league, args.event), leagues):
                    print(json.dumps(result), flush=True)
                if not args.watch:
                    break
                sleep(max(0, args.interval - (monotonic() - started)))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
