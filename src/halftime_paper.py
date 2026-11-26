"""Read-only Polymarket US screen for the halftime moneyline paper rule."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal
import re

import requests

from .inplay_scan import (
    MAX_AGE, PROBABILITY_HAIRCUT, apply_calibration, game_prior, load_calibration,
    parse_games, probability_frame, read_json, scoreboard, utc,
)
from .nfl_inplay_model import forecast as nfl_forecast, load as load_nfl
from .nfl_inplay_scan import NFL_SCOREBOARD
from .predict_today import ET
from .us_wager import BOOK_MAX_AGE, CENT, ONE, STANDARD_FEE, WagerError, contract_url, decimal, name, public, team_names

PAPER_STAKE = Decimal("2")
MIN_PROBABILITY = 0.60
MIN_SCREENED_EV = 0.05
LEAGUES = ("NBA", "NFL")
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
    # Include the scoreboard's end-of-Q2 transition before it publishes Halftime.
    halftime = kind.get("name") in ("STATUS_HALFTIME", "STATUS_END_PERIOD") or re.fullmatch(
        r"(?:half\s*time|end (?:of )?(?:q2|2nd(?: quarter)?))",
        str(kind.get("shortDetail") or ""), re.I,
    )
    if (kind.get("state") != "in" or kind.get("completed") or status.get("period") != 2
            or status.get("clock") != 0 or not halftime):
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
    return {"id": event["id"], "league": league, "teams": teams, "scores": scores, "tip": tip, "raw": event}


def scores(league, day):
    if league == "NBA":
        return scoreboard(day)
    return read_json("GET", NFL_SCOREBOARD, max_age=MAX_AGE,
                     params={"dates": day.strftime("%Y%m%d")}, headers={"Cache-Control": "no-cache"})["events"]


def home_probability(game, now):
    if game["league"] == "NFL":
        artifact = load_nfl()
        if artifact is None:
            raise ValueError("NFL calibration unavailable")
        (probability, _, _), _ = nfl_forecast((game["scores"]["home"], game["scores"]["away"], 1800), artifact)
        return probability
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


def contract_and_sides(game):
    for event_slug in event_slugs(game):
        slug = f"aec-{event_slug}"
        try:
            market = public(f"/v1/markets/slug/{slug}")["market"]
        except WagerError:
            continue
        listed = utc(market.get("gameStartTime"))
        rules = str(market.get("description") or "")
        kind = "basketball" if game["league"] == "NBA" else "football"
        if (market.get("slug") != slug or market.get("active") is not True or market.get("closed") is not False
                or market.get("marketType") != "moneyline" or market.get("sportsMarketType") != f"{kind}_team_full_game_winner"
                or listed is None or abs(listed - game["tip"]) > timedelta(minutes=30)
                or not re.search(r"overtime is included|including (?:any )?overtime", rules, re.I)
                or re.search(r"regulation only|exclud\w+ overtime", rules, re.I)):
            continue
        if game["league"] == "NFL" and not re.search(r"tie.{0,70}(?:50-50|\$0\.50)", rules, re.I):
            continue
        sides = market.get("marketSides") or []
        if len(sides) != 2 or {side.get("long") for side in sides} != {True, False}:
            continue
        assignments = {}
        for key, team in game["teams"].items():
            aliases = {name(team.get(field)) for field in
                       ("abbreviation", "displayName", "shortDisplayName", "location", "name")} - {""}
            matches = [side["long"] for side in sides if aliases & team_names(side)]
            if len(matches) == 1:
                assignments[key] = matches[0]
        if len(set(assignments.values())) == 2:
            return {"slug": event_slug, "seriesSlug": game["league"].lower()}, market, assignments
    return None


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


@dataclass(frozen=True)
class HalftimeSelection:
    event: str
    market: str
    long: bool
    side: str
    probability: float
    url: str


def prepare_selection(game):
    probability = home_probability(game, datetime.now(timezone.utc))
    if not MIN_PROBABILITY <= probability <= 1 and not 0 <= probability <= 1 - MIN_PROBABILITY:
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
    return HalftimeSelection(event["slug"], market["slug"], long, game["teams"][selected]["shortDisplayName"],
                             p, contract_url(event, market, long))


def price_selection(game, selection):
    book = public(f"/v1/markets/{selection.market}/book", fresh_book=True)["marketData"]
    quote = paper_quote(book, selection.market, selection.long, datetime.now(timezone.utc))
    if quote is None:
        return None
    screened_ev = (selection.probability - PROBABILITY_HAIRCUT) / quote["cost"] - 1
    if screened_ev < MIN_SCREENED_EV:
        return None
    away, home = (game["teams"][key] for key in ("away", "home"))
    return {**quote, "event": selection.event, "league": game["league"],
            "title": f"{away['shortDisplayName']} @ {home['shortDisplayName']}",
            "score_text": f"{away['abbreviation']} {game['scores']['away']} · {home['abbreviation']} {game['scores']['home']}",
            "side": selection.side, "probability": selection.probability,
            "screened_ev": screened_ev, "url": selection.url}


def price_game(game):
    selection = prepare_selection(game)
    return price_selection(game, selection) if selection is not None else None


def screen_halftime(*, leagues=LEAGUES, score_feed=None, price_feed=None):
    score_feed = score_feed or scores
    price_feed = price_feed or price_game
    started = datetime.now(timezone.utc)
    day = started.astimezone(ET).date()
    days = (day - timedelta(days=1), day) if started.astimezone(ET).hour < 6 else (day,)
    warnings = set()

    def fetch(league, date):
        try:
            return {game["id"]: game for event in score_feed(league, date)
                    if (game := game_state(league, event)) is not None}
        except FEED_ERRORS:
            warnings.add(f"{league} halftime scoreboard unavailable")
            return {}

    def price(game):
        try:
            return price_feed(game)
        except FEED_ERRORS:
            warnings.add(f"{game['league']} halftime pricing partially unavailable")
            return None

    with ThreadPoolExecutor(max_workers=6) as pool:
        keys = [(league, date) for league in leagues for date in days]
        slates = list(pool.map(lambda key: fetch(*key), keys))
        games = {key: game for slate in slates for key, game in slate.items()}
        picks = dict(zip(games, pool.map(price, games.values())))
        needed = {(games[key]["league"], games[key]["tip"].astimezone(ET).date())
                  for key, pick in picks.items() if pick}
        verified = {key: game for slate in pool.map(lambda key: fetch(*key), sorted(needed))
                    for key, game in slate.items()}
    now = datetime.now(timezone.utc)
    rows = [pick for key, pick in picks.items() if pick and key in verified
            and games[key]["scores"] == verified[key]["scores"]
            and -5 <= (now - utc(pick["quote_utc"])).total_seconds() <= BOOK_MAX_AGE]
    return {"updated_utc": now.isoformat(), "games_at_halftime": len(games),
            "picks": sorted(rows, key=lambda pick: (-pick["screened_ev"], -pick["probability"], pick["event"])),
            "warnings": sorted(warnings)}
