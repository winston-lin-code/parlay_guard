"""Read-only CS2 series prices and Elo estimates for Polymarket US moneylines."""

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from functools import lru_cache
from math import comb
import re
import time

import requests

from . import cs2_ratings
from .cs2_ratings import analyzer_ratings, fragcharts_ratings, predict_ratings, team_key
from .env_config import env_value
from .inplay_scan import PROBABILITY_HAIRCUT, read_json, utc
from .pro_live_ev import best_paper_quote
from .us_wager import GATEWAY, SCANNER_BOOKS, SLUG, ONE, BookRateLimited, decimal, public

BOOK_AGE = 12
SCORE_AGE = 120
ROUND_AGE = 20
PRICE_INTERVAL = 5
RATING_INTERVAL = 30 * 60
RATING_RETRY_INTERVAL = 2 * 60
CITO_INTERVAL = 8
REFERENCE_SIZE = Decimal("5")
SLIPPAGE = Decimal(".01")
CITO = "https://api.citoapi.com/api/v1/cs2/live"
FORMATS = (1, 3, 5)
FEED_ERRORS = (requests.RequestException, OSError, ValueError, TypeError, KeyError)


def now():
    return datetime.now(timezone.utc)


def fresh(at, reference, seconds):
    return at is not None and 0 <= (reference - at).total_seconds() <= seconds


def series_probability(maps, best_of, p_map, current_map=None):
    target = best_of // 2 + 1

    @lru_cache(maxsize=None)
    def play(a, b):
        if a >= target or b >= target:
            return float(a >= target)
        return p_map * play(a + 1, b) + (1 - p_map) * play(a, b + 1)

    a, b = maps
    if current_map is None or max(maps) >= target:
        return play(a, b)
    return current_map * play(a + 1, b) + (1 - current_map) * play(a, b + 1)


def map_probability(rounds, p_round):
    """MR12 regulation with repeated six-round overtime at 12–12."""
    overtime_win = sum(comb(6, k) * p_round**k * (1 - p_round)**(6 - k) for k in (4, 5, 6))
    overtime_tie = 20 * p_round**3 * (1 - p_round)**3
    overtime = overtime_win / (1 - overtime_tie)

    @lru_cache(maxsize=None)
    def block(a, b, target, tie_at, tied_probability):
        if a == b == tie_at:
            return tied_probability
        if a >= target or b >= target:
            return float(a >= target)
        return (p_round * block(a + 1, b, target, tie_at, tied_probability)
                + (1 - p_round) * block(a, b + 1, target, tie_at, tied_probability))

    a, b = rounds
    if min(a, b) < 12:
        return block(a, b, 13, 12, overtime)
    start = 12 + 3 * ((min(a, b) - 12) // 3)
    return block(a - start, b - start, 4, 3, overtime)


def inverse(probability, model):
    low, high = 0., 1.
    for _ in range(28):
        mid = (low + high) / 2
        if model(mid) < probability:
            low = mid
        else:
            high = mid
    return (low + high) / 2


def projections(elo, maps, formats, rounds=None, *, per_map=False):
    # Convert series Elo using a BO3 reference. Map Elo can be used directly.
    expected = 1 / (1 + 10 ** ((elo[1] - elo[0]) / 400))
    p_map = expected if per_map else inverse(expected, lambda p: series_probability((0, 0), 3, p))
    current = None
    if rounds is not None:
        p_round = inverse(p_map, lambda p: map_probability((0, 0), p))
        current = map_probability(rounds, p_round)
    return [series_probability(maps, best_of, p_map, current) for best_of in formats]


def series_score(text):
    match = re.fullmatch(r"\s*(\d+)\s*[-–]\s*(\d+)\s*", str(text or ""))
    if match:
        a, b = map(int, match.groups())
        if max(a, b) <= 3 and a + b <= 5:
            return a, b
    return None


def venue_rounds(state, team_ids, map_number, maps):
    if maps is None or map_number is None or map_number != sum(maps) + 1:
        return None
    entries = [item for item in state.get("periodScores") or []
               if item.get("number") == map_number and re.fullmatch(rf"Map\s*{map_number}", item.get("label") or "", re.I)]
    if len(entries) != 1 or len(entries[0].get("scores") or []) != 2:
        return None
    scores = {str(part.get("competitorId")): part.get("score") for part in entries[0]["scores"]}
    result = tuple(scores.get(team) for team in team_ids)
    return result if all(type(value) is int and 0 <= value <= 100 for value in result) else None


def fixture(event, reference):
    slug = event.get("slug") or ""
    tip = utc(event.get("startTime") or event.get("startDate"))
    if (not SLUG.fullmatch(slug) or not slug.startswith("cs2-") or event.get("live") is not True
            or event.get("ended") or tip is None or tip > reference or reference - tip > timedelta(hours=12)):
        return None
    matches = [m for m in event.get("markets") or []
               if m.get("slug") == "aec-" + slug and m.get("marketType") == "moneyline"
               and m.get("sportsMarketType") == "esports_match_winner" and m.get("status") == "MARKET_STATUS_OPEN"
               and m.get("active") is True and m.get("closed") is False]
    teams = event.get("teams") or []
    if len(matches) != 1 or len(teams) != 2 or any(not team.get("name") or team.get("id") is None for team in teams):
        return None
    market = matches[0]
    sides = market.get("marketSides") or []
    if len(sides) != 2 or any(type(side.get("long")) is not bool for side in sides):
        return None
    ordered = {str(side.get("teamId")): side for side in sides}
    team_ids = tuple(str(team["id"]) for team in teams)
    if set(ordered) != set(team_ids) or {side["long"] for side in sides} != {True, False}:
        return None
    state = event.get("eventState") or {}
    period = str(state.get("period") or event.get("period") or "")
    at = utc(state.get("updatedAt"))
    score = series_score(state.get("score"))
    if (state.get("live") is not True or state.get("ended") or score != series_score(event.get("score"))
            or at is None or at < tip):
        score = None
    named_map = re.fullmatch(r"(?:Map|Game)\s*([1-5])", period, re.I)
    map_number = int(named_map[1]) if named_map else None
    names = tuple(team["name"] for team in teams)
    formats = {int(match[1]) for field in ("title", "description") for match in
               re.finditer(r"\bBO([135])\b", str(event.get(field) or ""), flags=re.I)}
    return {"event": slug, "title": event.get("title") or " vs ".join(names), "market": market, "names": names,
            "sides": tuple(ordered[team_id] for team_id in team_ids), "maps": score, "period": period,
            "map_number": map_number, "best_of": next(iter(formats)) if len(formats) == 1 else None,
            "start": tip, "score_at": at, "rounds": venue_rounds(state, team_ids, map_number, score)}


def live_events():
    result = {}
    for offset in range(0, 600, 100):
        page = read_json("GET", GATEWAY + "/v1/events", max_age=BOOK_AGE, params={
            "active": "true", "closed": "false", "live": "true", "limit": 100, "offset": offset, "tagSlug": "cs2",
        })
        events = page["events"]
        if not isinstance(events, list):
            raise ValueError("Missing event listings")
        for event in events:
            result[event.get("slug")] = event
        if len(events) < 100:
            return result.values()
    raise ValueError("Incomplete CS2 listing page")


def cito_list(key):
    return read_json("GET", CITO, max_age=ROUND_AGE, headers={"x-api-key": key})["data"]


def cito_match(match, live_rows):
    keys = tuple(team_key(name) for name in match["names"])
    rows = [row for row in live_rows if row.get("status") == "live" and
            {team_key(row.get("team1Name")), team_key(row.get("team2Name"))} == set(keys)]
    if len(rows) != 1:
        return None
    row = rows[0]
    return row if type(row.get("bestOf")) is int and row["bestOf"] in FORMATS else None


def cito_rounds(key, match, listing, reference):
    match_id = str(listing.get("matchId") or "")
    if not re.fullmatch(r"cs2-match-\d+", match_id):
        return None
    board = read_json("GET", f"{CITO}/{match_id}/scoreboard", max_age=ROUND_AGE,
                      headers={"x-api-key": key})["data"]
    at = utc((board.get("meta") or {}).get("updatedAt"))
    map_number = board.get("currentGame")
    checked = reference()
    if (board.get("matchId") != match_id or board.get("status") != "live" or board.get("bestOf") != listing["bestOf"]
            or not fresh(at, checked, ROUND_AGE) or at < match["start"]):
        return None
    teams = {str(team.get("id")): team for team in board.get("teams", [])}
    providers = {team_key(listing.get(f"team{i}Name")): str(listing.get(f"team{i}Id")) for i in (1, 2)}
    sides = tuple(teams.get(providers.get(team_key(label))) for label in match["names"])
    if any(side is None for side in sides):
        return None
    maps = tuple(side.get("score") for side in sides)
    target = listing["bestOf"] // 2 + 1
    if (any(type(value) is not int or not 0 <= value < target for value in maps)
            or map_number != sum(maps) + 1):
        return None
    if fresh(match["score_at"], checked, SCORE_AGE) and (
            match["maps"] is not None and match["maps"] != maps or
            match["map_number"] is not None and match["map_number"] != map_number):
        return None
    rounds = tuple(side.get("mapScore") for side in sides)
    if any(type(score) is not int or not 0 <= score <= 100 for score in rounds):
        return None
    return {"maps": maps, "rounds": rounds, "map_number": map_number,
            "map_name": (board.get("game") or {}).get("map"), "at": at}


def reference_quote(book, match, side, probability, reference):
    if (book.get("marketSlug") != match["market"]["slug"] or book.get("state") != "MARKET_STATE_OPEN"
            or not fresh(utc(book.get("transactTime")), reference, BOOK_AGE) or side.get("tradable") is not True):
        return {}
    long = side["long"]
    prices = []
    for level in book.get("offers" if long else "bids") or []:
        if (level.get("px") or {}).get("currency") != "USD":
            continue
        px, size = decimal(level["px"].get("value")), decimal(level.get("qty"))
        if 0 < px < 1 and size > 0:
            prices.append(px if long else ONE - px)
    if not prices:
        return {}
    quote = {"ask": float(min(prices)), "quote_utc": book["transactTime"]}
    if probability is None:
        return quote
    low, high = probability
    depth = best_paper_quote(match["market"], side, low, book, REFERENCE_SIZE, SLIPPAGE, reference)
    if depth:
        quantity, cost = depth["quantity"], float(depth["total_estimate"])
        quote.update(limit=float(depth["limit"]), total_cost=cost, shares=quantity,
                     ev_low=depth["screened_ev"], ev_high=max(0, high - PROBABILITY_HAIRCUT) * quantity / cost - 1)
    return quote


def market_book(slug):
    return SCANNER_BOOKS.read(slug, fetch=public, clock=now, max_age=BOOK_AGE)


def evaluate(match, ratings, cito_rows, key):
    row = {"event": match["event"], "title": match["title"], "period": match["period"], "maps": match["maps"],
           "status": "waiting_for_score", "scores_at": match["score_at"].isoformat() if match["score_at"] else None,
           "sides": [{"name": label} for label in match["names"]]}
    team_keys = tuple(team_key(label) for label in match["names"])
    for side, team in zip(row["sides"], team_keys):
        side["ratings"] = [{"source": source.name, "rating": source.teams[team]}
                           for source in ratings if team in source.teams]
    reference = now()
    outcome = None
    maps, rounds, map_number = match["maps"], None, match["map_number"]
    listing = cito_match(match, cito_rows) if key else None
    supplied = listing["bestOf"] if listing else None
    confirmed = match["best_of"] or supplied
    if match["rounds"] is not None and fresh(match["score_at"], reference, ROUND_AGE):
        rounds = match["rounds"]
        row.update(rounds=rounds, rounds_at=match["score_at"].isoformat())
    if listing and (not match["best_of"] or confirmed == supplied):
        try:
            if round_state := cito_rounds(key, match, listing, now):
                observed = round_state["at"]
                if rounds is None or observed >= match["score_at"]:
                    maps, rounds, map_number = (round_state[field] for field in ("maps", "rounds", "map_number"))
                    row.update(maps=maps, rounds=rounds, rounds_at=observed.isoformat(),
                               period=f"Map {map_number}", map_name=round_state["map_name"])
        except FEED_ERRORS:
            row["rounds_unavailable"] = True
    if maps is not None and (rounds is not None or fresh(match["score_at"], now(), SCORE_AGE)):
        formats = tuple(b for b in ((confirmed,) if confirmed else FORMATS)
                        if b >= (map_number or sum(maps) + 1) and max(maps) < b // 2 + 1)
        if map_number is not None and map_number != sum(maps) + 1:
            formats = ()
        # If two sources explicitly disagree on the format, leave EV blank.
        if match["best_of"] and supplied and match["best_of"] != supplied:
            formats = ()
        primary = next((source for source in ratings if all(team in source.teams for team in team_keys)), None)
        if not formats:
            row["status"] = "format_or_score_unverified"
        elif not primary:
            row["status"] = "rating_missing"
        else:
            row.update(status="waiting_for_book", formats=formats, rating_source=primary.name,
                       model_scope="round_score" if rounds is not None else "series_score")
            probabilities = projections([primary.teams[team] for team in team_keys],
                                        maps, formats, rounds, per_map=primary.per_map)
            low, high = min(probabilities), max(probabilities)
            outcome = ((low, high), (1 - high, 1 - low))
            for side, (lower, upper) in zip(row["sides"], outcome):
                side.update(probability_low=lower, probability_high=upper)
    try:
        book = market_book(match["market"]["slug"])
        reference = now()
        # Only combine price and game state after both were observed in order.
        quoted = utc(book.get("transactTime"))
        score_at = utc(row.get("rounds_at") or row["scores_at"])
        effective = outcome if quoted is not None and score_at is not None and quoted >= score_at else None
        for i, side in enumerate(match["sides"]):
            quote = reference_quote(book, match, side, effective[i] if effective else None, reference)
            row["sides"][i].update(quote)
        if outcome and any("ev_low" in side for side in row["sides"]):
            row["status"] = "priced"
    except BookRateLimited as exc:
        row.update(status="book_rate_limited", retry_after_seconds=exc.retry_after)
    except FEED_ERRORS:
        if outcome:
            row["status"] = "book_unavailable"
    return row


class Reader:
    def __init__(self):
        self.cito_key = env_value("CS2_CITO_API_KEY")
        self.ratings = cs2_ratings.current_ratings([
            *cs2_ratings.load_ratings(cs2_ratings.CACHE_PATH, "cache"),
            *cs2_ratings.load_ratings(cs2_ratings.SEED_PATH, "bundled"),
        ], now())
        self.rating_warnings = []
        self.rating_checked = float("-inf")
        self.rating_interval = RATING_RETRY_INTERVAL
        self.pending_ratings = None
        self.cito_rows = []
        self.cito_checked = float("-inf")
        self.cito_warning = None

    @staticmethod
    def fetch_ratings():
        reference = now()
        available, warnings = [], []
        for name, fetch in (("Fragcharts", fragcharts_ratings), ("CS ELO Analyzer", analyzer_ratings),
                            ("CSGOPredict", predict_ratings)):
            try:
                available.append(fetch(reference))
            except FEED_ERRORS as exc:
                warnings.append(f"{name}: {'dated ranking unavailable' if isinstance(exc, ValueError) else 'provider unavailable'}")
        return available, warnings

    def snapshot(self, pool):
        tick = time.monotonic()
        events_future = pool.submit(live_events)
        if self.pending_ratings is not None and self.pending_ratings.done():
            updates, self.rating_warnings = self.pending_ratings.result()
            self.pending_ratings = None
            self.rating_interval = RATING_INTERVAL if updates else RATING_RETRY_INTERVAL
            self.ratings = cs2_ratings.current_ratings([*self.ratings, *updates], now())
            if updates:
                try:
                    cs2_ratings.save_ratings(cs2_ratings.CACHE_PATH,
                                            [source for source in self.ratings if source.origin != "bundled"])
                except OSError:
                    self.rating_warnings.append("Could not save the local Elo cache.")
        if self.pending_ratings is None and tick - self.rating_checked >= self.rating_interval:
            self.rating_checked = tick
            self.pending_ratings = pool.submit(self.fetch_ratings)
        self.ratings = cs2_ratings.current_ratings(self.ratings, now())
        key = self.cito_key
        if key and tick - self.cito_checked >= CITO_INTERVAL:
            self.cito_checked = tick
            try:
                self.cito_rows = cito_list(key)
                self.cito_warning = None
            except FEED_ERRORS:
                self.cito_rows = []
                self.cito_warning = "Cito round feed unavailable."
        warnings = []
        try:
            reference = now()
            matches = [m for event in events_future.result() if (m := fixture(event, reference))]
        except FEED_ERRORS:
            matches = []
            warnings.append("Polymarket US CS2 listings unavailable.")
        warnings.extend(self.rating_warnings)
        if self.cito_warning:
            warnings.append(self.cito_warning)
        rows = list(pool.map(lambda match: evaluate(match, self.ratings, self.cito_rows, key), matches))
        return {"updated_utc": now().isoformat(), "games": rows, "warnings": warnings, "round_feed": bool(key),
                "rating_sources": [{"name": source.name, "as_of": source.as_of.isoformat(),
                                    "origin": source.origin, "teams": len(source.teams)} for source in self.ratings]}


def read_for_display(snapshot, reference):
    games = []
    price_fields = {"ask", "ev_low", "ev_high", "quote_utc", "total_cost", "shares", "limit"}
    model_fields = {"probability_low", "probability_high", "ev_low", "ev_high"}
    for row in snapshot["games"]:
        rounds_at = utc(row.get("rounds_at"))
        stale = (not fresh(rounds_at, reference, ROUND_AGE) if rounds_at is not None
                 else not fresh(utc(row["scores_at"]), reference, SCORE_AGE))
        sides = [{key: value for key, value in side.items()
                  if (key not in price_fields or fresh(utc(side.get("quote_utc")), reference, BOOK_AGE))
                  and (key not in model_fields or not stale)} for side in row["sides"]]
        status = row["status"]
        if stale:
            status = "waiting_for_score"
        elif status == "priced" and not any("ev_low" in side for side in sides):
            status = "waiting_for_book"
        games.append({**row, "status": status, "sides": sides,
                      "rounds": None if stale else row.get("rounds")})
    return {**snapshot, "games": games}
