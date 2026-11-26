"""Read-only Polymarket US screen for the halftime moneyline paper rule."""

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from decimal import Decimal
import re
from time import time_ns

import requests

from .inplay_scan import (
    ESPN_SCOREBOARD, MAX_AGE, PROBABILITY_HAIRCUT, apply_calibration, game_prior, load_calibration,
    parse_games, probability_frame, read_json, utc,
)
from .college_inplay_model import RULES, forecast as college_forecast, load as load_college
from .college_score_calibration import load_score_artifact
from .college_sources import LEAGUES as COLLEGE_LEAGUES
from .halftime_basketball_model import HALF_PERIOD, load_models, win_probability
from .nfl_inplay_model import forecast as nfl_forecast, load as load_nfl
from .nfl_inplay_scan import NFL_SCOREBOARD
from .predict_today import ET
from .us_wager import BOOK_MAX_AGE, CENT, GATEWAY, ONE, SLUG, STANDARD_FEE, WagerError, contract_url, decimal, name, public, team_names

PAPER_STAKE = Decimal("2")
MIN_PROBABILITY = 0.60
MIN_SCREENED_EV = 0.05
WINDOW_SECONDS = 5 * 60
QUARTER_SECONDS = {"NBA": 12 * 60, "NFL": 15 * 60}
LEAGUES = ("NBA", "NFL", "CBB", "WCBB", "CFB", "WNBA")
OTHER_SCOREBOARDS = {
    "CBB": (COLLEGE_LEAGUES["mbb"]["espn"], COLLEGE_LEAGUES["mbb"]["group"]),
    "CFB": (COLLEGE_LEAGUES["cfb"]["espn"], COLLEGE_LEAGUES["cfb"]["group"]),
    "WCBB": ("basketball/womens-college-basketball", 50),
    "WNBA": ("basketball/wnba", None),
}
FEED_ERRORS = (requests.RequestException, ValueError, KeyError, OSError, TypeError)
US_ALIASES = {
    "GSW": "gs", "NOP": "no", "NYK": "ny", "PHX": "pho", "SAS": "sa", "WSH": "was",
    "JAC": "jax",
}


def game_state(league, event):
    season = (event.get("season") or {}).get("type")
    competitions = event.get("competitions") or []
    tip = utc(event.get("date"))
    if season not in (2, "2") or len(competitions) != 1 or tip is None or not event.get("id"):
        return None
    competition = competitions[0]
    status = competition.get("status") or event.get("status") or {}
    kind = status.get("type") or {}
    if kind.get("state") != "in" or kind.get("completed"):
        return None
    period = HALF_PERIOD.get(league, 2)
    current_period, clock = status.get("period"), status.get("clock")
    detail = str(kind.get("shortDetail") or "")
    status_name = kind.get("name")
    break_label = r"end (?:of )?(?:h1|1st(?: half)?)" if period == 1 else r"end (?:of )?(?:q2|2nd(?: quarter)?)"
    halftime = status_name in ("STATUS_HALFTIME", "STATUS_END_PERIOD") or re.fullmatch(
        rf"(?:half\s*time|ht|{break_label})", detail, re.I)
    quarter_seconds = QUARTER_SECONDS.get(league)
    remaining = None
    if current_period == period and clock == 0 and halftime:
        checkpoint = "Halftime"
        if quarter_seconds:
            remaining = 2 * quarter_seconds
    elif (quarter_seconds is not None and status_name == "STATUS_IN_PROGRESS"
          and type(clock) in (int, float)):
        if current_period == 2 and 0 < clock <= WINDOW_SECONDS:
            checkpoint = "Before halftime"
        elif current_period == 3 and quarter_seconds - WINDOW_SECONDS <= clock <= quarter_seconds:
            checkpoint = "After halftime"
        else:
            return None
        remaining = (4 - current_period) * quarter_seconds + clock
    else:
        return None
    sides = {side.get("homeAway"): side for side in competition.get("competitors") or []}
    if set(sides) != {"home", "away"}:
        return None
    teams, scores = {}, {}
    for side, competitor in sides.items():
        team = competitor.get("team") or {}
        code = team.get("abbreviation")
        score = str(competitor.get("score", ""))
        if not code or not re.fullmatch(r"\d+", score):
            return None
        teams[side] = team
        scores[side] = int(score)
    return {"id": event["id"], "league": league, "teams": teams, "scores": scores, "tip": tip, "raw": event,
            "neutral": bool(competition.get("neutralSite")), "checkpoint": checkpoint,
            "remaining": remaining, "clock_text": status.get("displayClock") or ""}


def scores(league, day):
    # A cached zero-clock score can outlast the end of halftime.
    params = {"dates": day.strftime("%Y%m%d"), "limit": 100, "_": time_ns()}
    endpoint = ESPN_SCOREBOARD if league == "NBA" else NFL_SCOREBOARD
    if league in OTHER_SCOREBOARDS:
        sport, group = OTHER_SCOREBOARDS[league]
        endpoint = f"https://site.api.espn.com/apis/site/v2/sports/{sport}/scoreboard"
        if group is not None:
            params.update(groups=group, limit=500)
    return read_json("GET", endpoint, max_age=MAX_AGE, params=params,
                     headers={"Cache-Control": "no-cache"})["events"]


def home_probability(game, now):
    if game["league"] == "NFL":
        artifact = load_nfl()
        if artifact is None:
            raise ValueError("NFL calibration unavailable")
        (probability, _, _), _ = nfl_forecast((game["scores"]["home"], game["scores"]["away"], game["remaining"]), artifact)
        return probability
    if game["league"] in HALF_PERIOD:
        model = load_models().get(game["league"])
        if model is None:
            raise ValueError("Halftime basketball model unavailable")
        if any(int(team.get("id") or 0) not in model["teams"] for team in game["teams"].values()):
            return None
        return win_probability(model, game["scores"]["home"], game["scores"]["away"], game["neutral"])
    if game["league"] == "CFB":
        artifact, membership = load_college("cfb"), load_score_artifact("cfb")
        if artifact is None or membership is None:
            raise ValueError("FBS calibration unavailable")
        if any(str(team.get("id")) not in membership["teams"] for team in game["teams"].values()):
            return None
        check = artifact["holdout"]["moneyline"][1]
        if check["brier"] >= check["baseline_brier"] or check["log_loss"] >= check["baseline_log_loss"]:
            return None
        count, duration, _, _ = RULES["cfb"]
        return college_forecast((game["scores"]["home"], game["scores"]["away"], count * duration // 2, 2), artifact)[0]
    calibration = load_calibration()
    nba = next(iter(parse_games([game["raw"]], now).values()), None)
    if calibration is None or nba is None:
        raise ValueError("NBA halftime model unavailable")
    projection, _ = game_prior(nba)
    frame = probability_frame(nba, projection, calibration["reference_errors"])
    return apply_calibration(frame, nba["remaining"], calibration)[0]


def event_slugs(game):
    codes = [game["teams"][side]["abbreviation"].upper() for side in ("away", "home")]
    choices = []
    for code in codes:
        aliases = (US_ALIASES.get(code, code).lower(), code.lower())
        choices.append(tuple(dict.fromkeys((*aliases, "la") if game["league"] == "NFL" and code == "LAR" else aliases)))
    league = game["league"].lower()
    day = game["tip"].astimezone(ET).date()
    return (f"{league}-{away}-{home}-{day}" for away in choices[0] for home in choices[1])


def side_assignments(game, sides):
    if len(sides) != 2 or any(type(side.get("long")) is not bool for side in sides):
        return None
    if {side["long"] for side in sides} != {True, False}:
        return None
    assignments = {}
    for key, team in game["teams"].items():
        aliases = {name(team.get(field)) for field in
                   ("abbreviation", "displayName", "shortDisplayName", "location", "name")} - {""}
        matches = []
        for side in sides:
            listed = team_names(side)
            if game["league"] == "WCBB":
                # The exchange appends (W) to some school names.
                listed |= {name(re.sub(r"\s+\(W\)$", "", value, flags=re.I))
                           for value in ((side.get("team") or {}).get(field) or "" for field in ("name", "safeName"))}
            if aliases & listed:
                matches.append(side["long"])
        if len(matches) == 1:
            assignments[key] = matches[0]
    return assignments if len(set(assignments.values())) == 2 else None


def verified_contract(game, event_slug, market):
    slug = f"aec-{event_slug}"
    listed = utc(market.get("gameStartTime"))
    rules = str(market.get("description") or "")
    league = game["league"]
    kind = "football" if league in ("NFL", "CFB") else "basketball"
    scopes = {f"{kind}_team_full_game_winner"}
    if league == "WNBA" and re.search(r"winner of the .* (?:WNBA|women.s (?:professional )?basketball|professional basketball) game", rules, re.I):
        scopes.add("moneyline")
    if (market.get("slug") != slug or not SLUG.fullmatch(slug)
            or market.get("active") is not True or market.get("closed") is not False
            or market.get("marketType") != "moneyline" or market.get("sportsMarketType") not in scopes
            or listed is None or abs(listed - game["tip"]) > timedelta(minutes=30)
            or not re.search(r"overtime is included|including (?:any )?overtime", rules, re.I)
            or re.search(r"regulation only|exclud\w+ overtime", rules, re.I)):
        return None
    if league == "NFL" and not re.search(r"tie.{0,70}(?:50-50|\$0\.50)", rules, re.I):
        return None
    sides = side_assignments(game, market.get("marketSides") or [])
    return ({"slug": event_slug, "seriesSlug": league.lower()}, market, sides) if sides else None


def contract_and_sides(game):
    if game["league"] in OTHER_SCOREBOARDS:
        # ESPN and Polymarket often use different college abbreviations and may
        # list neutral-site opponents in opposite order. Match both teams and start time.
        matches = [listing for slug, market in game.get("listed_moneylines", [])
                   if (listing := verified_contract(game, slug, market))]
        return matches[0] if len(matches) == 1 else None
    for event_slug in event_slugs(game):
        try:
            market = public(f"/v1/markets/slug/aec-{event_slug}")["market"]
        except WagerError:
            continue
        listing = verified_contract(game, event_slug, market)
        if listing:
            return listing
    return None


def live_inventory():
    listings = {league: {} for league in OTHER_SCOREBOARDS}
    live_games = {}
    offset = 0
    while True:
        params = {"active": "true", "closed": "false", "live": "true", "limit": 100, "offset": offset}
        try:
            page = read_json("GET", f"{GATEWAY}/v1/events", max_age=MAX_AGE, params=params)
        except (requests.ConnectionError, requests.Timeout):
            page = read_json("GET", f"{GATEWAY}/v1/events", max_age=MAX_AGE, params=params)
        events = page["events"]
        for event in events:
            slug = event.get("slug") or ""
            league = slug.split("-", 1)[0].upper()
            if event.get("ended") or event.get("live") is not True or not SLUG.fullmatch(slug):
                continue
            title = str(event.get("title") or "").strip()
            if title and len(event.get("teams") or []) >= 2:
                live_games[slug] = {"event": slug, "league": league, "title": title, "phase": "live",
                                    "score_text": str(event.get("score") or ""),
                                    "period_text": str(event.get("period") or "")}
            if league not in listings:
                continue
            for market in event.get("markets") or []:
                if market.get("slug") == f"aec-{slug}":
                    listings[league][slug] = market
        if len(events) < 100:
            break
        offset += len(events)
        if offset >= 1000:
            raise ValueError("Live event catalog is incomplete")
    return {league: list(markets.items()) for league, markets in listings.items()}, list(live_games.values())


def paper_quote(book, slug, long, now):
    observed = utc(book.get("transactTime"))
    if (book.get("marketSlug") != slug or book.get("state") != "MARKET_STATE_OPEN" or observed is None
            or not -5 <= (now - observed).total_seconds() <= BOOK_MAX_AGE):
        return None
    levels = []
    for level in book.get("offers" if long else "bids") or []:
        px = level.get("px") or {}
        if px.get("currency") == "USD":
            price, qty = decimal(px.get("value")), decimal(level.get("qty"))
            if CENT <= price <= ONE - CENT and qty > 0:
                levels.append((price if long else ONE - price, qty))
    remaining = PAPER_STAKE
    shares = premium = Decimal(0)
    for price, qty in sorted(levels):
        all_in = price + STANDARD_FEE * price * (ONE - price)
        spend = min(remaining, qty * all_in)
        taken = spend / all_in
        shares += taken
        premium += taken * price
        remaining -= spend
        if not remaining:
            return {"ask": float(premium / shares), "cost": float(PAPER_STAKE / shares),
                    "quote_utc": observed.isoformat()}
    return None


def price_game(game):
    probability = home_probability(game, datetime.now(timezone.utc))
    if probability is None or not (MIN_PROBABILITY <= probability <= 1 or 0 <= probability <= 1 - MIN_PROBABILITY):
        return None
    selected = "home" if probability >= MIN_PROBABILITY else "away"
    p = probability if selected == "home" else 1 - probability
    listing = contract_and_sides(game)
    if listing is None:
        return None
    event, market, sides = listing
    long = sides[selected]
    side = next(side for side in market["marketSides"] if side["long"] is long)
    if side.get("tradable") is not True:
        return None
    book = public(f"/v1/markets/{market['slug']}/book", fresh_book=True)["marketData"]
    quote = paper_quote(book, market["slug"], long, datetime.now(timezone.utc))
    if quote is None:
        return None
    screened_ev = (p - PROBABILITY_HAIRCUT) / quote["cost"] - 1
    if screened_ev < MIN_SCREENED_EV:
        return None
    away, home = (game["teams"][key] for key in ("away", "home"))
    return {**quote, "event": event["slug"], "league": game["league"],
            "title": f"{away['shortDisplayName']} @ {home['shortDisplayName']}",
            "score_text": f"{away['abbreviation']} {game['scores']['away']} · {home['abbreviation']} {game['scores']['home']}",
            "checkpoint": game["checkpoint"], "clock_text": game["clock_text"],
            "side": game["teams"][selected]["shortDisplayName"], "probability": p,
            "screened_ev": screened_ev, "url": contract_url(event, market, long)}


def screen_halftime():
    started = datetime.now(timezone.utc)
    day = started.astimezone(ET).date()
    days = (day - timedelta(days=1), day) if started.astimezone(ET).hour < 6 else (day,)
    warnings = set()

    def fetch(league, date):
        try:
            return {(league, game["id"]): game for event in scores(league, date)
                    if (game := game_state(league, event)) is not None}
        except FEED_ERRORS:
            warnings.add(f"{league} halftime scoreboard unavailable")
            return {}

    def price(game):
        try:
            return price_game(game)
        except FEED_ERRORS:
            warnings.add(f"{game['league']} halftime pricing partially unavailable")
            return None

    with ThreadPoolExecutor(max_workers=6) as pool:
        keys = [(league, date) for league in LEAGUES for date in days]
        inventory = pool.submit(live_inventory)
        slates = list(pool.map(lambda key: fetch(*key), keys))
        games = {key: game for slate in slates for key, game in slate.items()}
        try:
            listings, live_games = inventory.result()
        except FEED_ERRORS:
            warnings.add("Polymarket US live event listings unavailable")
            listings, live_games = {}, None
        for game in games.values():
            game["listed_moneylines"] = listings.get(game["league"], [])
        picks = dict(zip(games, pool.map(price, games.values())))
        needed = {(games[key]["league"], games[key]["tip"].astimezone(ET).date())
                  for key, pick in picks.items() if pick}
        verified = {key: game for slate in pool.map(lambda key: fetch(*key), sorted(needed))
                    for key, game in slate.items()}
    now = datetime.now(timezone.utc)
    rows = [pick for key, pick in picks.items() if pick and key in verified
            and games[key]["scores"] == verified[key]["scores"]
            and games[key]["checkpoint"] == verified[key]["checkpoint"]
            and (games[key]["remaining"] is None
                 or -2 <= games[key]["remaining"] - verified[key]["remaining"] <= 15)
            and -5 <= (now - utc(pick["quote_utc"])).total_seconds() <= BOOK_MAX_AGE]
    return {"updated_utc": now.isoformat(), "games_at_halftime": len(games), "leagues": list(LEAGUES), "live_games": live_games,
            "picks": sorted(rows, key=lambda pick: (-pick["screened_ev"], -pick["probability"], pick["event"])),
            "warnings": sorted(warnings)}
