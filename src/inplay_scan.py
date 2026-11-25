"""Read-only NBA market screen using ESPN game state and Polymarket order books."""

import math
import re
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from statistics import NormalDist

import pandas as pd
import requests

from .auto_live import CLOB, ET, GAMMA, NICKNAMES, array, as_num, match_team, utc
from .common import DATA
from .elo import INITIAL_ELO
from .game_lines_live import fit_line_models, predict_line
from .predict_today import (clean_team_history, compute_elo, fetch_current_season,
                            load_history, make_completed_games, normalize_schedule_team)

ESPN_SCOREBOARD = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard"
MAX_AGE = 35
REFERENCE_NOTIONAL = 10
PROBABILITY_HAIRCUT = 0.05
SCORING_MARKETS = {"moneyline", "spreads", "totals"}


def read_json(method, url, **kwargs):
    response = requests.request(method, url, timeout=(4, 8), **kwargs)
    response.raise_for_status()
    age = as_num(response.headers.get("Age"))
    if age is not None and age > MAX_AGE and (url == ESPN_SCOREBOARD or url.endswith("/books")):
        raise ValueError("Upstream returned a cached live feed")
    return response.json()


def scoreboard(day):
    data = read_json("GET", ESPN_SCOREBOARD, params={"dates": day.strftime("%Y%m%d"), "limit": 100},
                     headers={"Cache-Control": "no-cache"})
    if not isinstance(data.get("events"), list):
        raise ValueError(f"NBA scoreboard omitted events for {day}")
    return data["events"]


def parse_games(events, now):
    games = {}
    ambiguous = set()
    for event in events:
        tip = utc(event.get("date"))
        competitions = event.get("competitions") or []
        if tip is None or len(competitions) != 1 or (event.get("season") or {}).get("type") not in (2, 3, "2", "3"):
            continue
        competition = competitions[0]
        status = competition.get("status") or event.get("status") or {}
        phase = (status.get("type") or {}).get("state")
        if phase not in ("pre", "in") or (status.get("type") or {}).get("completed"):
            continue
        sides = {entry.get("homeAway"): entry for entry in competition.get("competitors") or []}
        if set(sides) != {"home", "away"}:
            continue
        home, away = (normalize_schedule_team((sides[side].get("team") or {}).get("abbreviation", ""))
                      for side in ("home", "away"))
        if home == away or any(team.lower() not in NICKNAMES for team in (home, away)):
            continue
        day = tip.astimezone(ET).date()
        slug = f"nba-{away.lower()}-{home.lower()}-{day}"
        event_id = event.get("id")
        if not event_id or slug in ambiguous:
            continue
        if slug in games:
            if games[slug]["espn_id"] == event_id:
                continue
            ambiguous.add(slug)
            games.pop(slug, None)
            continue
        period, clock = as_num(status.get("period")), as_num(status.get("clock"))
        hp, ap = (as_num(sides[side].get("score")) for side in ("home", "away"))
        if phase == "pre":
            if not 20 < (tip - now).total_seconds() <= 36 * 3600:
                continue
            remaining = 2880
        else:
            if period not in (1, 2, 3, 4) or clock is None or not 0 <= clock <= 720:
                continue
            if hp is None or ap is None or min(hp, ap) < 0 or not hp.is_integer() or not ap.is_integer():
                continue
            remaining = (4 - int(period)) * 720 + clock
            # This research model has no endgame fouling or overtime component.
            if remaining < 60:
                continue
        games[slug] = {"slug": slug, "espn_id": event_id, "home": home, "away": away, "day": day, "phase": phase,
                       "tip": tip, "period": period, "clock": status.get("displayClock"),
                       "remaining": remaining, "home_score": hp, "away_score": ap}
    return games


@lru_cache(maxsize=4)
def prior_models(day, season, historical_path, current_path, historical_version, current_version):
    old = load_history(historical_path)
    current = fetch_current_season(season, current_path)
    previous = current.loc[pd.to_datetime(current.GAME_DATE) < pd.Timestamp(day)]
    if previous.empty or (day - pd.to_datetime(previous.GAME_DATE).max().date()).days > 3:
        raise ValueError("Fresh completed NBA history is unavailable")
    completed = make_completed_games(clean_team_history(old, current, day))
    models, history, metrics = fit_line_models(completed, include_elo=True)
    return models, history, metrics, compute_elo(completed, season)


def game_prior(game):
    day = game["day"]
    year = day.year if day.month >= 7 else day.year - 1
    season = f"{year}-{(year + 1) % 100:02d}"
    historical_path = DATA / "nba_games_updated.csv"
    current_path = DATA / f"nba_{season.replace('-', '_')}_latest_raw.csv"
    models, history, metrics, ratings = prior_models(
        day, season, historical_path, current_path,
        historical_path.stat().st_mtime_ns, current_path.stat().st_mtime_ns,
    )
    home, away = game["home"], game["away"]
    diff = ratings.get(home, INITIAL_ELO) - ratings.get(away, INITIAL_ELO)
    projection = predict_line(models, history, home=home, away=away, game_day=day, elo_diff=diff)
    if projection is None:
        raise ValueError("Unified NBA forecast is unavailable")
    return projection, metrics


def probability_frame(game, projection, metrics):
    fraction = game["remaining"] / 2880
    margin = projection["home_margin"] * fraction
    total = projection["projected_total"] * fraction
    if game["phase"] == "in":
        hp, ap = game["home_score"], game["away_score"]
        margin += hp - ap
        # Prior scoring pace has the weight of half a game.
        pace = (projection["projected_total"] / 2 + hp + ap) / (1.5 - fraction)
        total = hp + ap + pace * fraction
    margin_sd = max(4, metrics["home_margin_holdout_mae"] * math.sqrt(math.pi * fraction / 2))
    total_sd = max(5, metrics["total_holdout_mae"] * math.sqrt(math.pi * fraction / 2))
    # Ground the moneyline in the calibrated pregame probability and let the
    # model's expected final margin reflect score and time for in-play values.
    if game["phase"] == "pre":
        home_win = projection["home_win_probability"]
    else:
        home_win = 1 - NormalDist(margin, margin_sd).cdf(0)
    return home_win, NormalDist(margin, margin_sd), NormalDist(total, total_sd)


def market_probabilities(market, game, frame):
    kind = market.get("sportsMarketType")
    home, away = game["home"].lower(), game["away"].lower()
    labels = array(market.get("outcomes"))
    tokens = array(market.get("clobTokenIds"))
    if (kind not in SCORING_MARKETS or len(labels) != 2 or len(tokens) != 2
            or any(not str(token).isdigit() for token in tokens) or str(tokens[0]) == str(tokens[1])):
        return None
    question = str(market.get("question") or "")
    rules = question + " " + str(market.get("description") or "")
    if re.search(r"regulation only|first (?:half|quarter)|second half|quarter winner|excludes? overtime", rules, re.I):
        return None
    if kind == "moneyline":
        sides = [match_team(label, away, home) for label in labels]
        if set(sides) != {home, away}:
            return None
        p_first = frame[0] if sides[0] == home else 1 - frame[0]
    else:
        line = as_num(market.get("line"))
        if line is None or abs(line) % 1 != 0.5:
            return None
        if kind == "spreads":
            match = re.fullmatch(r"Spread:\s*(.+?)\s*\(([+-]?\d+(?:\.\d+)?)\)", question, re.I)
            sides = [match_team(label, away, home) for label in labels]
            if (not match or set(sides) != {home, away}
                    or match_team(match[1], away, home) != sides[0] or float(match[2]) != line):
                return None
            p_first = 1 - frame[1].cdf(-line) if sides[0] == home else frame[1].cdf(line)
        else:
            match = re.search(r"O/U\s+(\d+(?:\.\d+)?)$", question, re.I)
            if not match or float(match[1]) != line or {str(x).lower() for x in labels} != {"over", "under"}:
                return None
            over = 1 - frame[2].cdf(line)
            p_first = over if str(labels[0]).lower() == "over" else 1 - over
    return [(str(tokens[0]), str(labels[0]), p_first), (str(tokens[1]), str(labels[1]), 1 - p_first)]


def fee_schedule(market):
    if market.get("feesEnabled") is False:
        return 0, 1
    schedule = market.get("feeSchedule") or {}
    rate, exponent = as_num(schedule.get("rate")), as_num(schedule.get("exponent"))
    if market.get("feesEnabled") is True and rate is not None and exponent == 1 and 0 <= rate <= 1:
        return rate, exponent
    return None


def priced_side(book, probability, fees, now):
    # CLOB's book timestamp is Unix milliseconds.
    timestamp = as_num(book.get("timestamp"))
    if timestamp is None or not -5 <= now.timestamp() - timestamp / 1000 <= MAX_AGE:
        return None
    levels = [(as_num(level.get("price")), as_num(level.get("size"))) for level in book.get("asks") or []]
    valid = sorted((price, size) for price, size in levels
                   if price is not None and size is not None and 0 < price < 1 and size > 0)
    remaining, shares, fee = REFERENCE_NOTIONAL, 0, 0
    rate, exponent = fees
    for price, size in valid:
        take = min(remaining / price, size)
        shares += take
        fee += take * rate * (price * (1 - price)) ** exponent
        remaining -= take * price
        if remaining < 1e-8:
            break
    minimum = as_num(book.get("min_order_size"))
    if remaining >= 1e-8 or minimum is None or minimum <= 0 or shares < minimum:
        return None
    ask = REFERENCE_NOTIONAL / shares
    fee_per_share = fee / shares
    # Use the lower return under cash-fee and fee-in-shares accounting.
    def net_ev(p):
        return min(p / (ask + fee_per_share), p * (1 - fee_per_share / ask) / ask) - 1
    return {"ask": ask, "best_ask": valid[0][0], "shares": shares, "fee_per_share": fee_per_share,
            "ev": net_ev(probability), "screened_ev": net_ev(max(0, probability - PROBABILITY_HAIRCUT)),
            "quote_utc": datetime.fromtimestamp(timestamp / 1000, timezone.utc).isoformat()}


def scan_catalog(stop, progress):
    """Walk the entire active catalog, without requesting unsupported order books."""
    cursor = None
    cursors = set()
    seen = set()
    counts = {"cataloged": 0, "open_order_books": 0}
    while not stop.is_set():
        params = {"active": "true", "closed": "false", "limit": 100}
        if cursor is not None:
            params["after_cursor"] = cursor
        page = read_json("GET", GAMMA + "/markets/keyset", params=params)
        markets = page.get("markets")
        if not isinstance(markets, list):
            raise ValueError("Gamma catalog omitted markets")
        for market in markets:
            market_id = market.get("id")
            if market_id and market_id not in seen:
                seen.add(market_id)
                counts["cataloged"] += 1
                if market.get("acceptingOrders") is True and market.get("enableOrderBook") is True:
                    counts["open_order_books"] += 1
        cursor = page.get("next_cursor")
        if cursor in cursors:
            raise ValueError("Gamma catalog returned a repeated cursor")
        progress({**counts, "status": "scanning" if cursor else "complete"})
        if not cursor:
            return
        cursors.add(cursor)
        stop.wait(0.1)


def event_markets(game):
    try:
        event = read_json("GET", f"{GAMMA}/events/slug/{game['slug']}")
    except requests.HTTPError as exc:
        if exc.response is not None and exc.response.status_code == 404:
            return []
        raise
    if event.get("slug") != game["slug"] or event.get("closed"):
        return []
    contracts = []
    for market in event.get("markets") or []:
        if (market.get("closed") or market.get("active") is not True
                or market.get("acceptingOrders") is not True or market.get("enableOrderBook") is not True):
            continue
        scheduled = utc(market.get("gameStartTime") or event.get("gameStartTime"))
        if scheduled is None or abs((game["tip"] - scheduled).total_seconds()) > 30 * 60:
            continue
        contracts.append(market)
    return contracts


def _books(tokens):
    data = read_json("POST", CLOB + "/books", json=[{"token_id": t} for t in tokens])
    if not isinstance(data, list):
        raise ValueError("CLOB returned no order-book batch")
    return {str(book.get("asset_id")): book for book in data}


def scan_opportunities():
    started = datetime.now(timezone.utc)
    day = started.astimezone(ET).date()
    warnings = []
    coverage = {"games": 0, "live_games": 0, "open_markets": 0, "modeled_markets": 0, "priced_sides": 0}

    def fetch(label, function, arg, default):
        try:
            return function(arg)
        except (requests.RequestException, ValueError, KeyError, OSError, TypeError) as exc:
            warnings.append(f"{label}: {type(exc).__name__}")
            return default

    with ThreadPoolExecutor(max_workers=6) as pool:
        days = [day + timedelta(days=offset) for offset in (-1, 0, 1)]
        boards = pool.map(lambda d: fetch(f"ESPN {d}", scoreboard, d, []), days)
        games = parse_games([event for events in boards for event in events], started)
        coverage["games"] = len(games)
        coverage["live_games"] = sum(game["phase"] == "in" for game in games.values())
        candidates = []
        for game, markets in zip(games.values(), pool.map(
                lambda g: fetch(g["slug"], event_markets, g, []), games.values())):
            coverage["open_markets"] += len(markets)
            if not any(m.get("sportsMarketType") in SCORING_MARKETS for m in markets):
                continue
            prior = fetch("NBA pregame inputs", game_prior, game, None)
            if prior is None:
                continue
            frame = probability_frame(game, *prior)
            for market in markets:
                sides = market_probabilities(market, game, frame)
                fees = fee_schedule(market)
                if sides is not None and fees is not None:
                    coverage["modeled_markets"] += 1
                    candidates.append((game, market, sides, fees))
        tokens = list(dict.fromkeys(token for _, _, sides, _ in candidates for token, _, _ in sides))
        batches = [tokens[i:i + 100] for i in range(0, len(tokens), 100)]
        books = {}
        for batch in pool.map(lambda ids: fetch("CLOB quotes", _books, ids, {}), batches):
            books.update(batch)
        live_days = sorted({g["day"] for g, _, _, _ in candidates if g["phase"] == "in"})
        confirmations = pool.map(lambda d: fetch(f"ESPN verification {d}", scoreboard, d, []), live_days)
        latest = parse_games([event for events in confirmations for event in events], datetime.now(timezone.utc))
    now = datetime.now(timezone.utc)
    opportunities = []
    for original, market, sides, fees in candidates:
        game = original
        if original["phase"] == "in":
            confirmed = latest.get(original["slug"])
            if (confirmed is None or confirmed["phase"] != "in"
                    or confirmed["espn_id"] != original["espn_id"]
                    or any(confirmed[field] != original[field] for field in ("home_score", "away_score", "period"))
                    or not -2 <= original["remaining"] - confirmed["remaining"] <= 15):
                continue
        elif (original["tip"] - now).total_seconds() <= 20:
            continue
        if (now - started).total_seconds() > MAX_AGE:
            warnings.append("Scan exceeded the quote and game-state freshness window")
            break
        for token, label, probability in sides:
            priced = priced_side(books.get(token, {}), probability, fees, now)
            if priced is None:
                continue
            coverage["priced_sides"] += 1
            if priced["screened_ev"] <= 0:
                continue
            opportunities.append({
                "event": game["slug"], "matchup": f"{game['away']} @ {game['home']}",
                "phase": "live" if game["phase"] == "in" else "pregame",
                "period": game["period"], "clock": game["clock"],
                "score": [game["away_score"], game["home_score"]] if game["phase"] == "in" else None,
                "market": market["question"], "side": label, "token_id": token,
                "probability": probability, **priced,
            })
    return {"updated_utc": now.isoformat(), "coverage": coverage,
            "opportunities": sorted(opportunities, key=lambda row: -row["screened_ev"]),
            "warnings": sorted(set(warnings))}
