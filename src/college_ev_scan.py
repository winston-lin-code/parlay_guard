"""Price verified NCAA and professional pregame contracts using score forecasts."""

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone

import re

import requests

from . import pro_score_calibration
from .pro_ev_scan import fetch_pro_scoreboard, resolve_pro_team
from .college_score_calibration import load_score_artifact, predict_game
from .college_sources import ET, fetch_scoreboard, outcome_team
from .inplay_scan import MAX_AGE, _books, array, fee_schedule, market_probabilities, priced_side, utc

MAX_MODEL_AGE = timedelta(days=14)
MAX_CATALOG_AGE = timedelta(minutes=3)


def match_fixture(fixture, schedule):
    """A full-game moneyline identifies both teams before any alt line is priced."""
    start = utc(fixture["start_utc"])
    if start is None:
        return None
    found = {}
    resolve_team = resolve_pro_team if fixture.get("pro") else outcome_team
    for game in schedule:
        tipoff = utc(game["start_utc"])
        if not tipoff or abs(tipoff - start) > timedelta(minutes=30):
            continue
        home, away = game["home"], game["away"]
        team_ids = {home["id"], away["id"]}
        for market in fixture["college_markets"]:
            if market["sportsMarketType"] != "moneyline":
                continue
            labels = array(market["outcomes"])
            if len(labels) == 2 and {resolve_team(label, home, away) for label in labels} == team_ids:
                found[game["espn_id"]] = game
    return next(iter(found.values())) if len(found) == 1 else None


def candidates_for_fixture(fixture, game, model):
    pro = fixture.get("pro")
    projection = pro_score_calibration.predict(game, model) if pro else predict_game(game, model)
    if projection is None:
        return []
    home, away = game["home"], game["away"]

    resolve_team = resolve_pro_team if pro else outcome_team

    def resolve(label):
        team = resolve_team(label, home, away)
        return team.lower() if team is not None else None

    result = []
    start = utc(game["start_utc"])
    for market in fixture["college_markets"]:
        kind = market["sportsMarketType"]
        evaluation = model["holdout"].get(kind)
        if not evaluation:
            continue
        if pro:
            if (evaluation["brier"] >= evaluation["baseline_brier"]
                    or evaluation["log_loss"] >= evaluation["baseline_log_loss"]):
                continue
            if kind == "moneyline" and pro == "nfl" and not re.search(
                    r"ends in a tie.{0,40}resolve 50-50", market.get("description") or "", re.I):
                continue
        elif (evaluation["calibrated_brier"] > evaluation["raw_brier"]
              or evaluation["calibrated_log_loss"] > evaluation["raw_log_loss"]):
            continue
        match_start = utc(market.get("gameStartTime")) or utc(fixture["start_utc"])
        if match_start is None or abs(match_start - start) > timedelta(minutes=30):
            continue
        sides = market_probabilities(market, {"home": home["id"], "away": away["id"]}, projection,
                                     resolve_team=resolve)
        if sides is not None and kind == "moneyline" and pro == "nfl":
            # A tied NFL moneyline pays $0.50 per share under these contract rules.
            tie = model["tie_rate"]
            sides = [(token, label, p * (1 - tie) + .5 * tie) for token, label, p in sides]
        fees = fee_schedule(market)
        if sides is not None and fees is not None:
            result.append((market, sides, fees))
    return result


def scan_college(sports):
    now = datetime.now(timezone.utc)
    observed = utc(sports.get("observed_utc"))
    warnings = []
    result = {"updated_utc": None, "games": [], "opportunities": [], "warnings": warnings}
    if observed is None or not timedelta(seconds=-5) <= now - observed <= MAX_CATALOG_AGE:
        return result
    matches = {}
    schedule_requests = set()
    models = {}
    for fixture in sports.get("games", []):
        sport = fixture.get("college") or fixture.get("pro")
        start = utc(fixture.get("start_utc"))
        if not sport or not fixture.get("college_markets") or not start or start <= now + timedelta(minutes=5):
            continue
        if sport not in models:
            model = pro_score_calibration.load(sport) if fixture.get("pro") else load_score_artifact(sport)
            last = utc(model.get("team_results_through")) if model else None
            models[sport] = model if last and timedelta(0) <= now - last <= MAX_MODEL_AGE else None
            if models[sport] is None:
                warnings.append(f"{sport.upper()}: current pregame model unavailable")
        if models[sport] is None:
            continue
        label = (sport, start.astimezone(ET).date().isoformat())
        schedule_requests.add(label)
        matches[fixture["event"]] = fixture, label

    def schedule(label):
        try:
            with requests.Session() as session:
                fetch = fetch_pro_scoreboard if label[0] in pro_score_calibration.SEASONS else fetch_scoreboard
                return fetch(session, *label, now=now)
        except (OSError, ValueError, KeyError, TypeError):
            warnings.append(f"{label[0].upper()} pregame scoreboard unavailable")
            return []

    def quotes(batch):
        try:
            return _books(batch)
        except (OSError, ValueError, KeyError, TypeError):
            warnings.append("Pregame order books unavailable")
            return {}

    with ThreadPoolExecutor(max_workers=6) as workers:
        labels = sorted(schedule_requests)
        schedules = dict(zip(labels, workers.map(schedule, labels)))
        to_price = []
        for fixture, label in matches.values():
            game = match_fixture(fixture, schedules[label])
            if game is None:
                continue
            markets = candidates_for_fixture(fixture, game, models[label[0]])
            public = {k: v for k, v in fixture.items() if k != "college_markets"}
            game_state = {**public, "modeled_markets": len(markets), "priced_sides": 0}
            result["games"].append(game_state)
            to_price.extend((game_state, game, *candidate) for candidate in markets)
        tokens = list(dict.fromkeys(token for _, _, _, sides, _ in to_price for token, _, _ in sides))
        batches = [tokens[i:i + 100] for i in range(0, len(tokens), 100)]
        books = {}
        for values in workers.map(quotes, batches):
            books.update(values)

    observed_now = datetime.now(timezone.utc)
    if result["games"]:
        result["updated_utc"] = observed_now.isoformat()
    if (observed_now - now).total_seconds() > MAX_AGE:
        warnings.append("Pregame scan exceeded the quote freshness window")
        return result
    for state, game, market, sides, fees in to_price:
        if utc(game["start_utc"]) <= observed_now + timedelta(minutes=5):
            continue
        for token, label, probability in sides:
            priced = priced_side(books.get(token, {}), probability, fees, observed_now)
            if priced is None:
                continue
            state["priced_sides"] += 1
            if priced["screened_ev"] > 0:
                result["opportunities"].append({
                    "event": state["event"], "matchup": state["title"], "phase": "pregame",
                    "market": market["question"], "side": label, "token_id": token,
                    "probability": probability, **priced,
                    "half_payout_tie": state.get("pro") == "nfl" and market["sportsMarketType"] == "moneyline",
                })
    result["warnings"] = sorted(set(warnings))
    return result
