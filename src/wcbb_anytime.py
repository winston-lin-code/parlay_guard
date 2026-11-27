"""Screen live ESPN and Polymarket US data for the WCBB anytime rule."""

from bisect import bisect_right
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from datetime import datetime, timedelta, timezone
from decimal import Decimal
import json
from pathlib import Path
import sqlite3

from . import halftime_paper
from .inplay_scan import PROBABILITY_HAIRCUT, read_json, utc
from .pro_live_ev import best_order_quote
from .us_wager import (BOOK_MAX_AGE, EVENT_CACHE_MAX_AGE, GATEWAY, ONE, SCANNER_BOOKS, SLUG,
                       BookRateLimited, decimal, public)

DB_PATH = Path("data/paper_trades/wcbb_anytime.sqlite3")
LIVE_DB_PATH = Path("data/live_orders/wcbb_anytime.sqlite3")
LEAGUE = "WCBB"
STAKE = Decimal("5.00")
MIN_PRICE = Decimal(".20")
MIN_PROBABILITY = Decimal(".60")
MIN_EV = Decimal(".05")
SLIPPAGE = Decimal(".01")
SCORE_DELAY = 15
SETTLEMENT_BATCH_SIZE = 50
SUMMARY_URL = ("https://site.api.espn.com/apis/site/v2/sports/basketball/"
               "womens-college-basketball/summary")


def now():
    return datetime.now(timezone.utc)


def game_state(event):
    competitions = event.get("competitions") or []
    tip = utc(event.get("date"))
    if len(competitions) != 1 or tip is None or not event.get("id"):
        return None
    competition = competitions[0]
    status = competition.get("status") or event.get("status") or {}
    kind = status.get("type") or {}
    period, clock = status.get("period"), status.get("clock")
    if (kind.get("state") != "in" or kind.get("completed") or type(period) is not int or period < 1
            or type(clock) not in (int, float) or not 0 <= clock <= (600 if period <= 4 else 300)):
        return None
    teams, scores = {}, {}
    for side in competition.get("competitors", []):
        key, team, score = side.get("homeAway"), side.get("team"), str(side.get("score", ""))
        if key in teams or not team or not team.get("id") or not team.get("abbreviation") or not score.isdecimal():
            return None
        teams[key], scores[key] = team, int(score)
    if set(teams) != {"home", "away"}:
        return None
    is_half = period == 2 and clock == 0 and kind.get("name") in ("STATUS_HALFTIME", "STATUS_END_PERIOD")
    return {"id": str(event["id"]), "league": LEAGUE, "tip": tip, "teams": teams, "scores": scores,
            "period": period, "clock": clock, "clock_text": status.get("displayClock") or "",
            "checkpoint": "Halftime" if is_half else f"Q{period}" if period <= 4 else f"OT{period - 4}"}


def forecast(summary, game, quoted):
    """Read the pre-quote ESPN estimate and reject it when the game state has moved."""
    header = summary.get("header") or {}
    if str(header.get("id")) != game["id"]:
        return None
    expected = (game["scores"]["home"], game["scores"]["away"], game["period"])
    rows = {}
    for play in summary.get("plays", []):
        at = utc(play.get("wallclock"))
        state = (play.get("homeScore"), play.get("awayScore"), (play.get("period") or {}).get("number"))
        if at is not None and all(type(value) is int for value in state):
            rows[at] = state, str(play.get("id"))
    times = sorted(rows)
    latest = bisect_right(times, quoted) - 1
    cutoff = bisect_right(times, quoted - timedelta(seconds=SCORE_DELAY)) - 1
    if cutoff < 0 or rows[times[latest]][0] != expected:
        return None
    values = {str(row.get("playId")): row.get("homeWinPercentage") for row in summary.get("winprobability", [])}
    max_age = 300 if game["checkpoint"] == "Halftime" else 120
    # A scoring play or period transition invalidates any earlier estimate.
    for index in range(latest, -1, -1):
        at = times[index]
        state, play_id = rows[at]
        if state != expected or (quoted - at).total_seconds() > max_age:
            return None
        value = values.get(play_id)
        if index <= cutoff and type(value) in (int, float) and 0 <= value <= 1:
            home = decimal(value)
            return home, at
    return None


def display_team(team):
    return team.get("shortDisplayName") or team.get("displayName") or team["abbreviation"]


def moneyline_url(event, market):
    if isinstance(event, str) and event.startswith("wcbb-") and SLUG.fullmatch(event) and market == "aec-" + event:
        return f"https://polymarket.us/sports/wcbb/{event}?marketSlug={market}"
    return None


def scan_game(game, listings, previous, *, live_orders=None):
    away, home = (game["teams"][key] for key in ("away", "home"))
    row = {"game_id": game["id"], "title": f"{display_team(away)} vs {display_team(home)}",
           "score": f"{away['abbreviation']} {game['scores']['away']} · {home['abbreviation']} {game['scores']['home']}",
           "checkpoint": game["checkpoint"], "clock": game["clock_text"], "status": "unmatched_market"}
    live = live_orders is not None
    entry = previous.get(game["id"])
    retry = live and entry is not None and live_orders.can_retry(entry)
    if entry is not None and not retry:
        row.update(status="live_order_already_attempted" if live else "paper_entry_logged", side=entry["side"],
                   market_url=moneyline_url(entry.get("event"), entry.get("market")))
        if live:
            row["order_state"] = entry["state"]
        return row, None
    try:
        if listings is None:
            return {**row, "status": "market_catalog_unavailable"}, None
        listing = halftime_paper.contract_and_sides({**game, "listed_moneylines": listings})
    except halftime_paper.FEED_ERRORS:
        return {**row, "status": "market_catalog_unavailable"}, None
    if not listing:
        return row, None
    event, contract, assignments = listing
    row["market_url"] = moneyline_url(event["slug"], contract["slug"])
    if retry:
        row["refreshing_order"] = True
        if contract["slug"] != entry["market_slug"]:
            return {**row, "status": "live_order_already_attempted"}, None
    try:
        summary = read_json("GET", SUMMARY_URL, max_age=15, params={"event": game["id"]},
                            headers={"Cache-Control": "no-cache"})
    except halftime_paper.FEED_ERRORS:
        return {**row, "status": "espn_probability_unavailable"}, None
    try:
        if retry:
            book = SCANNER_BOOKS.request(f"/v1/markets/{contract['slug']}/book", public, fresh_book=True)
        else:
            book = SCANNER_BOOKS.read(contract["slug"], fetch=public, clock=now)
    except BookRateLimited as exc:
        return {**row, "status": "book_rate_limited", "retry_after_seconds": exc.retry_after}, None
    except halftime_paper.FEED_ERRORS:
        return {**row, "status": "book_unavailable"}, None
    try:
        quoted = utc(book.get("transactTime"))
        checked = now()
        if book.get("marketSlug") != contract["slug"]:
            return {**row, "status": "book_unavailable"}, None
        if book.get("state") != "MARKET_STATE_OPEN":
            return {**row, "status": "book_not_open"}, None
        if quoted is not None and (age := (checked - quoted).total_seconds()) > BOOK_MAX_AGE:
            return {**row, "status": "book_stale", "book_age_seconds": int(age)}, None
        if quoted is None or quoted < game["tip"] or quoted > checked:
            return {**row, "status": "no_fresh_book"}, None
        if retry and ((previous_at := utc(entry["recorded_at"])) is None or quoted <= previous_at):
            return {**row, "status": "waiting_for_retry_quote"}, None
        estimate = forecast(summary, game, quoted)
        if not estimate:
            return {**row, "status": "waiting_for_probability"}, None
        home_probability, score_at = estimate
        selected = "home" if home_probability >= Decimal(".50") else "away"
        probability = home_probability if selected == "home" else ONE - home_probability
        row.update(side=display_team(game["teams"][selected]), probability=float(probability), status="below_probability",
                   quote_utc=quoted.isoformat())
        if probability <= MIN_PROBABILITY:
            return row, None
        long = assignments[selected]
        market_side = next(side for side in contract["marketSides"] if side["long"] is long)
        if market_side.get("tradable") is not True:
            return {**row, "status": "no_fresh_book"}, None
        levels = (decimal(level["px"]["value"]) for level in book.get("offers" if long else "bids", [])
                  if level["px"].get("currency") == "USD" and decimal(level["qty"]) > 0)
        ask = min((px if long else ONE - px for px in levels), default=None)
        if ask is None or not 0 < ask < 1:
            return {**row, "status": "no_fresh_book"}, None
        row["ask"] = float(ask)
        if ask < MIN_PRICE:
            return {**row, "status": "below_price"}, None
        order_quote = best_order_quote(contract, market_side, float(probability), book, STAKE, SLIPPAGE, now())
        if not order_quote:
            return {**row, "status": "insufficient_depth"}, None
        order, quote = order_quote
        cost = decimal(quote["total_estimate"])
        ev = (probability - decimal(PROBABILITY_HAIRCUT)) * quote["quantity"] / cost - ONE
        row.update(limit=float(quote["limit"]), screened_ev=float(ev), cost=float(cost))
        if ev < MIN_EV:
            return {**row, "status": "below_ev"}, None
        entry = {"game_id": game["id"], "event": event["slug"], "market": contract["slug"],
                 "title": row["title"], "side": row["side"], "long": long, "quote_utc": quoted.isoformat(),
                 "score_utc": score_at.isoformat(), "checkpoint": row["checkpoint"], "clock": row["clock"],
                 "probability": row["probability"], "screened_ev": float(ev), "ask": row["ask"], "limit": quote["limit"],
                 "fee": quote["fee_estimate"], "cost": quote["total_estimate"], "quantity": quote["quantity"],
                 "order": order}
        return {**row, "status": "pending_recheck"}, entry
    except halftime_paper.FEED_ERRORS:
        return {**row, "status": "pricing_unavailable"}, None


def record_entry(path, entry):
    path.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(path, timeout=5)) as db, db:
        db.execute("""CREATE TABLE IF NOT EXISTS wcbb_entries (
            game_id TEXT PRIMARY KEY NOT NULL, event TEXT UNIQUE NOT NULL, recorded_at TEXT NOT NULL,
            detail TEXT NOT NULL, settlement TEXT, settled_at TEXT
        )""")
        db.execute("INSERT OR IGNORE INTO wcbb_entries (game_id, event, recorded_at, detail) VALUES (?, ?, ?, ?)",
                   (entry["game_id"], entry["event"], entry["recorded_at"], json.dumps(entry)))


def read_ledger(path):
    if not path.is_file():
        return []
    with closing(sqlite3.connect(path.absolute().as_uri() + "?mode=ro", uri=True, timeout=5)) as db:
        if not db.execute("SELECT 1 FROM sqlite_master WHERE name='wcbb_entries' AND type='table'").fetchone():
            return []
        return [{**json.loads(detail), "settlement": settlement, "settled_at": settled_at}
                for detail, settlement, settled_at in db.execute(
                    "SELECT detail, settlement, settled_at FROM wcbb_entries ORDER BY recorded_at DESC, game_id")]


def settled_payout(market, long):
    if market.get("status") != "MARKET_STATUS_RESOLVED":
        return None
    sides = market.get("marketSides")
    if not isinstance(sides, list) or len(sides) != 2:
        return None
    prices = {}
    for side in sides:
        if not isinstance(side, dict) or type(side.get("long")) is not bool:
            return None
        try:
            prices[side["long"]] = decimal(side.get("price"))
        except ValueError:
            return None
    if len(prices) != 2 or any(not 0 <= price <= ONE for price in prices.values()) or sum(prices.values()) != ONE:
        return None
    return prices[long]


def settle(path):
    pending = {entry["market"]: entry for entry in read_ledger(path)
               if entry["settlement"] is None and type(entry.get("long")) is bool
               and moneyline_url(entry.get("event"), entry.get("market"))}
    slugs = list(pending)
    for offset in range(0, len(slugs), SETTLEMENT_BATCH_SIZE):
        batch = slugs[offset:offset + SETTLEMENT_BATCH_SIZE]
        # Resolved markets may be absent from /markets/slug/{slug}; the batch feed includes them.
        payload = read_json("GET", f"{GATEWAY}/v1/markets", max_age=EVENT_CACHE_MAX_AGE,
                            params={"slug": batch, "limit": len(batch)})
        if not isinstance(payload, dict) or not isinstance(payload.get("markets"), list):
            raise ValueError("Polymarket US omitted markets from the settlement response.")
        settled_at = now().isoformat()
        updates = []
        for market in payload["markets"]:
            if not isinstance(market, dict) or market.get("slug") not in batch:
                continue
            entry = pending[market["slug"]]
            payout = settled_payout(market, entry["long"])
            if payout is not None:
                updates.append((str(payout), settled_at, entry["game_id"]))
        if updates:
            with closing(sqlite3.connect(path, timeout=5)) as db, db:
                db.executemany("UPDATE wcbb_entries SET settlement=?, settled_at=? WHERE game_id=? AND settlement IS NULL",
                               updates)


def performance(entries):
    wins = losses = other = 0
    spend = pnl = peak = drawdown = Decimal(0)
    for row in sorted((row for row in entries if row["settlement"] is not None),
                      key=lambda row: (row["settled_at"], row["game_id"])):
        payout, cost = decimal(row["settlement"]), decimal(row["cost"])
        profit = payout * row["quantity"] - cost
        row["pnl"] = float(profit)
        row["result"] = "Won" if payout == 1 else "Lost" if payout == 0 else "Other settlement"
        wins += payout == 1
        losses += payout == 0
        other += 0 < payout < 1
        spend += cost
        pnl += profit
        peak = max(peak, pnl)
        drawdown = max(drawdown, peak - pnl)
    return {"wins": wins, "losses": losses, "other": other, "open": len(entries) - wins - losses - other,
            "net_profit": float(pnl), "roi": float(pnl / spend) if spend else None, "max_drawdown": float(drawdown)}


def scoreboards(days):
    events = {}
    for day in days:
        for event in halftime_paper.scores(LEAGUE, day):
            if (game := game_state(event)) is not None:
                events[game["id"]] = game
    return events


def scan(path, *, start_index=0, live_orders=None):
    started = now().astimezone(halftime_paper.ET)
    day = started.date()
    days = (day - timedelta(days=1), day) if started.hour < 6 else (day,)
    # ESPN determines which games are in progress. The venue can be late to mark an open market live.
    earliest = datetime.combine(days[0], datetime.min.time(), tzinfo=halftime_paper.ET) - timedelta(minutes=30)
    latest = datetime.combine(day + timedelta(days=1), datetime.min.time(), tzinfo=halftime_paper.ET) + timedelta(minutes=30)
    start_window = (earliest.astimezone(timezone.utc), latest.astimezone(timezone.utc))
    warnings = []
    ledger = read_ledger(path)
    live = live_orders is not None
    with ThreadPoolExecutor(max_workers=8) as pool:
        inventory = pool.submit(halftime_paper.live_inventory, LEAGUE, start_window=start_window)
        try:
            games = scoreboards(days)
        except halftime_paper.FEED_ERRORS:
            games = {}
            warnings.append("ESPN scoreboard unavailable.")
        try:
            listings, _ = inventory.result()
        except halftime_paper.FEED_ERRORS:
            market_list = None
            warnings.append("Polymarket US listings unavailable.")
        else:
            market_list = listings.get(LEAGUE, [])
        ids = list(games)
        previous = live_orders.attempts(LEAGUE, ids) if live else {row["game_id"]: row for row in ledger}
        offset = start_index % len(ids) if ids else 0
        order = ids[offset:] + ids[:offset]
        # A busy venue can exhaust the first part of a scan. Rotate the
        # request order between passes while preserving the dashboard order.
        completed = dict(zip(order, pool.map(
            lambda key: scan_game(games[key], market_list, previous, live_orders=live_orders), order)))
        results = [completed[key] for key in ids]
    candidates = [entry for _, entry in results if entry]
    verified = {}
    if candidates:
        try:
            verified = scoreboards(sorted({games[entry["game_id"]]["tip"].astimezone(halftime_paper.ET).date()
                                           for entry in candidates}))
        except halftime_paper.FEED_ERRORS:
            warnings.append("ESPN score recheck unavailable.")
    for row, entry in results:
        if not entry:
            continue
        original, current = games[entry["game_id"]], verified.get(entry["game_id"])
        recorded = now()
        if (not current or any(current[key] != original[key] for key in ("scores", "period", "checkpoint"))
                or not -2 <= original["clock"] - current["clock"] <= SCORE_DELAY
                or not 0 <= (recorded - utc(entry["quote_utc"])).total_seconds() <= BOOK_MAX_AGE):
            row["status"] = "state_changed"
            continue
        entry["recorded_at"] = recorded.isoformat()
        payload = entry.pop("order")
        if live:
            attempt = {**entry, "league": LEAGUE, "official_game_id": entry["game_id"],
                       "quote": {"quote_utc": entry["quote_utc"], "total_estimate": entry["cost"]}}
            prior = previous.get(entry["game_id"])
            outcome = live_orders.place(attempt, payload, retry_for=prior["order_id"] if prior else None)
            row["status"] = outcome["status"]
            row["order_state"] = outcome.get("state")
        else:
            record_entry(path, entry)
            row["status"] = "paper_entry_logged"
    if live:
        orders = [{**entry, "market_url": moneyline_url(entry.get("event"), entry.get("market"))}
                  for entry in live_orders.recent()]
        if any(entry["state"] in ("pending", "unknown") for entry in orders):
            warnings.append("A WCBB order has an unknown status. Check your Polymarket US activity.")
    else:
        ledger = read_ledger(path) if candidates else ledger
        orders = []
    return {"status": "feed_unavailable" if warnings else "ready", "updated_utc": now().isoformat(),
            "games": [row for row, _ in results], "performance": performance(ledger),
            "entries": [{**entry, "market_url": moneyline_url(entry.get("event"), entry.get("market"))}
                        for entry in ledger[:25]], "live_ordering": live, "orders": orders, "warnings": warnings}
