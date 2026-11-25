"""Reprice verified regular-season NFL contracts from scores, clock and order books."""

import re
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone

import requests

from .college_ev_scan import MAX_CATALOG_AGE, match_fixture
from .college_sources import ET, parse_espn
from .inplay_scan import MAX_AGE, _books, array, as_num, fee_schedule, market_probabilities, priced_side, read_json, utc
from .nfl_inplay_model import forecast, load, score_clock
from .pro_ev_scan import resolve_pro_team

NFL_SCOREBOARD = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard"
FEED_ERRORS = (requests.RequestException, OSError, ValueError, KeyError, TypeError)


def scoreboard(when, now):
    payload = read_json("GET", NFL_SCOREBOARD, params={"dates": when.replace("-", "")},
                        headers={"Cache-Control": "no-cache"}, max_age=MAX_AGE)
    payload["events"] = [event for event in payload["events"] if (event.get("season") or {}).get("type") == 2]
    return parse_espn(payload, "nfl", when, now, phase="in")


def live_markets(fixture, game, projection, checks, resolve_team, *, nfl=False, line_bounds=None):
    home, away = game["home"], game["away"]

    def resolve(label):
        team = resolve_team(label, home, away)
        return team.lower() if team is not None else None

    start = utc(game["start_utc"])
    for market in fixture["college_markets"]:
        kind = market["sportsMarketType"]
        result = checks.get(kind)
        if not result:
            continue
        if line_bounds and kind in line_bounds:
            line = as_num(market.get("line"))
            lower, upper = line_bounds[kind]
            if line is None:
                continue
            if kind == "spreads":
                labels = array(market.get("outcomes"))
                if len(labels) != 2:
                    continue
                line = -line if resolve(labels[0]) == home["id"].lower() else line
            if not lower <= line <= upper:
                continue
        if result["brier"] >= result["baseline_brier"] or result["log_loss"] >= result["baseline_log_loss"]:
            continue
        if nfl and kind == "moneyline" and not re.search(
                r"ends in a tie.{0,40}resolve 50-50", market.get("description") or "", re.I):
            continue
        scheduled = utc(market.get("gameStartTime")) or utc(fixture["start_utc"])
        if scheduled is None or abs(scheduled - start) > timedelta(minutes=30):
            continue
        sides = market_probabilities(market, {"home": home["id"], "away": away["id"]}, projection,
                                     resolve_team=resolve)
        fees = fee_schedule(market)
        if sides and fees is not None:
            yield market, sides, fees


def eligible_markets(fixture, game, state, artifact):
    projection, bucket = forecast(state, artifact)
    checks = {kind: values[bucket] for kind, values in artifact["holdout"].items()}
    return live_markets(fixture, game, projection, checks, resolve_pro_team, nfl=True)


def scan_live(sports, sport, artifact, get_scoreboard, parse_clock, market_candidates):
    started = datetime.now(timezone.utc)
    result = {"updated_utc": None, "games": [], "opportunities": [], "warnings": []}
    catalog = utc(sports.get("observed_utc"))
    if artifact is None or catalog is None or not timedelta(seconds=-5) <= started - catalog <= MAX_CATALOG_AGE:
        return result
    fixtures = {}
    for fixture in sports.get("games", []):
        start = utc(fixture.get("start_utc"))
        if ((fixture.get("pro") or fixture.get("college")) == sport and fixture.get("college_markets") and start
                and started - timedelta(hours=7) <= start <= started):
            day = start.astimezone(ET).date().isoformat()
            fixtures.setdefault(day, []).append(fixture)
    for day, slate in fixtures.items():
        try:
            schedule = get_scoreboard(day, started)
            candidates = []
            for fixture in slate:
                game = match_fixture(fixture, schedule)
                state = parse_clock(game) if game else None
                if state is None:
                    continue
                view = {k: v for k, v in fixture.items() if k not in ("college_markets", "moneyline", "sports_game_id")}
                picks = list(market_candidates(fixture, game, state, artifact))
                candidates.append((game, state, view, picks))
            tokens = list(dict.fromkeys(token for _, _, _, picks in candidates
                                        for _, sides, _ in picks for token, _, _ in sides))
            books = {}
            if tokens:
                batches = [tokens[i:i + 100] for i in range(0, len(tokens), 100)]
                with ThreadPoolExecutor(max_workers=4) as pool:
                    for batch in pool.map(_books, batches):
                        books.update(batch)
            latest = {g["espn_id"]: g for g in get_scoreboard(day, datetime.now(timezone.utc))} if candidates else {}
        except FEED_ERRORS as exc:
            result["warnings"].append(f"{sport.upper()} live scan unavailable: {type(exc).__name__}")
            continue
        now = datetime.now(timezone.utc)
        if now - started > timedelta(seconds=MAX_AGE):
            break
        for game, state, view, picks in candidates:
            verified = latest.get(game["espn_id"])
            current = parse_clock(verified) if verified else None
            if current is None or current[:2] != state[:2] or current[3:] != state[3:] or not -2 <= state[2] - current[2] <= 15:
                continue
            home, away = state[:2]
            view.update(phase="live", score_text=f"{away}–{home}",
                        score_order=f"{game['away']['abbr']} / {game['home']['abbr']}",
                        period_text=verified["period_text"], clock_text=verified["clock_text"],
                        priced_sides=0, modeled_markets=len(picks), live_model=True)
            for market, sides, fees in picks:
                for token, label, probability in sides:
                    priced = priced_side(books.get(token, {}), probability, fees, now)
                    if not priced:
                        continue
                    view["priced_sides"] += 1
                    if priced["screened_ev"] > 0:
                        result["opportunities"].append({
                            "event": view["event"], "matchup": view["title"], "phase": "live",
                            "score": [away, home], "market": market["question"], "market_type": market["sportsMarketType"], "line": market.get("line"),
                            "side": label, "token_id": token,
                            "probability": probability, "half_payout_tie": sport == "nfl" and market["sportsMarketType"] == "moneyline", **priced,
                        })
            result["games"].append(view)
            result["updated_utc"] = now.isoformat()
    return result


def scan_nfl(sports):
    return scan_live(sports, "nfl", load(), scoreboard, score_clock, eligible_markets)
