"""Local live runner for ITF singles match-winner orders on Polymarket US."""

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing, nullcontext
from datetime import datetime, timezone
from decimal import ROUND_CEILING
import json
import os
from pathlib import Path
import sqlite3
from threading import Event, Lock, Thread
from time import monotonic

import requests

from .game_board_feed import follow_scores
from .halftime_paper import MIN_PROBABILITY, MIN_SCREENED_EV, PAPER_STAKE
from .inplay_scan import utc
from .tennis_live import (
    ITF_LEAGUES, first_set_checkpoint, load_elo_break_models, observe_first_set_break, tennis_projection,
)
from .us_wager import BOOK_MAX_AGE, CENT, SLUG, WagerError, decimal, name, public, quote_order, team_names

SCORE_MAX_AGE = 20
DATA_ERRORS = (requests.RequestException, ValueError, KeyError, TypeError)


def now():
    return datetime.now(timezone.utc)


def fresh(timestamp, max_age):
    value = utc(timestamp)
    return value is not None and -5 <= (now() - value).total_seconds() <= max_age


class Scores:
    def __init__(self):
        self.lock = Lock()
        self.games = {}

    def publish(self, game_id, update):
        league = update["score_league"]
        slug = update["event_slug"]
        if league not in ITF_LEAGUES | {"ITF"} and slug.split("-", 1)[0].upper() not in ITF_LEAGUES:
            return
        with self.lock:
            previous = self.games.get(game_id, {})
            if previous.get("phase") == "final":
                return
            self.games[game_id] = {**update, "first_set_break_utc": observe_first_set_break(previous, update, utc)}

    def get(self, game_id):
        with self.lock:
            return self.games.get(game_id, {})

    def active_games(self):
        with self.lock:
            return [(game_id, state) for game_id, state in self.games.items() if state["phase"] == "live"
                    and fresh(state["score_utc"], SCORE_MAX_AGE)]


def live_catalog():
    events = {}
    for offset in range(0, 1000, 100):
        page = public(f"/v1/events?active=true&closed=false&live=true&limit=100&offset={offset}")["events"]
        for event in page:
            slug = event.get("slug") or ""
            if SLUG.fullmatch(slug) and slug.split("-", 1)[0].upper() in ITF_LEAGUES:
                events[slug] = event
        if len(page) < 100:
            return events
    raise WagerError("Live event catalog exceeds its request limit.")


def catalog_scores(events, previous):
    """Build paper-only scores using the catalog's title order and update timestamp."""
    scores = {}
    for slug, event in events.items():
        title = str(event.get("title") or "")
        columns = title.split(" vs. ")
        teams = event.get("teams") or []
        timestamp = event.get("updatedAt")
        if (len(columns) != 2 or len(teams) != 2 or event.get("live") is not True
                or event.get("ended") is True or not event.get("score") or not event.get("period")
                or any(" / " in side or " - " in side for side in columns)
                or any(name(column) not in team_names({"team": team}) for column, team in zip(columns, teams))
                or not fresh(timestamp, SCORE_MAX_AGE)):
            continue
        score = {"event_slug": slug, "score_league": slug.split("-", 1)[0].upper(),
                 "score_columns": columns, "phase": "live", "score_utc": timestamp,
                 "score_text": event["score"], "period_text": event["period"], "source": "catalog"}
        score["first_set_break_utc"] = observe_first_set_break(previous.get(slug, {}), score, utc)
        scores[slug] = score
    return scores


def catalog_event(events, score):
    slug = score.get("event_slug")
    if slug:
        return events.get(slug)
    league = score["score_league"]
    matches = []
    for event in events.values():
        if league in ITF_LEAGUES and not event["slug"].startswith(league.lower() + "-"):
            continue
        teams = event.get("teams") or []
        if len(teams) != 2:
            continue
        aliases = [team_names({"team": team}) for team in teams]
        first, second = map(name, score["score_columns"])
        if ((first in aliases[0] and second in aliases[1])
                or (first in aliases[1] and second in aliases[0])) and first != second:
            matches.append(event)
    return matches[0] if len(matches) == 1 else None


def match_selection(event, score, report=lambda reason: None):
    slug = event.get("slug") or ""
    first, separator, second = str(event.get("title") or "").partition(" vs. ")
    if (not separator or not SLUG.fullmatch(slug) or (score.get("event_slug") and slug != score["event_slug"])
            or any(" / " in side or " - " in side for side in (first, second))
            or event.get("live") is not True or event.get("ended") is True
            or event.get("active") is not True or event.get("closed") is not False
            or not fresh(score.get("score_utc"), SCORE_MAX_AGE)):
        report("listing closed, score stale, or match unsupported")
        return None
    contracts = [market for market in event.get("markets", [])
                 if market.get("slug") == f"aec-{slug}" and market.get("marketType") == "moneyline"
                 and market.get("sportsMarketType") == "tennis_match_winner"
                 and market.get("active") is True and market.get("closed") is False]
    if len(contracts) != 1:
        report("no unique match-winner market")
        return None
    contract = contracts[0]
    sides = contract.get("marketSides") or []
    if len(sides) != 2 or {side.get("long") for side in sides} != {True, False}:
        report("invalid market sides")
        return None
    ordered = []
    for column in score["score_columns"]:
        matches = [side for side in sides if name(column) in team_names(side)]
        if len(matches) != 1:
            report("score player could not be matched")
            return None
        ordered.append(matches[0])
    if len(ordered) != 2 or ordered[0] is ordered[1]:
        report("score order ambiguous")
        return None
    names = []
    for side in ordered:
        labels = [label for label in (first, second) if name(label) in team_names(side)]
        if len(labels) != 1:
            report("market player could not be matched")
            return None
        names.append(labels[0])
    if len(set(names)) != 2:
        report("market player identity ambiguous")
        return None
    projection = tennis_projection({"event": slug, "title": event["title"]},
                                   {**score, "score_order": " / ".join(names)}, now())
    if not projection:
        report("unsupported score format")
        return None
    if not projection["checkpoint"]:
        report("outside first-set-break window")
        return None
    if not projection["elo_prior"]:
        report("one or both Elo ratings unavailable")
        return None
    first_probability = projection["first_probability"]
    selected = int(first_probability < .5)
    probability = (first_probability, 1 - first_probability)[selected]
    if probability < MIN_PROBABILITY:
        report("model probability below 60%")
        return None
    if ordered[selected].get("tradable") is not True:
        report("selected side paused")
        return None
    return contract, ordered[selected]["long"], {"side": names[selected], "probability": probability,
                                                   "checkpoint": projection["checkpoint"]}


def order_for_stake(pick, contract, long, book, stake, report=lambda reason: None):
    # Find the largest whole-contract quantity that fits the worst-case limit and fee.
    minimum = max(1, int(decimal(contract["minimumTradeQty"]).to_integral_value(rounding=ROUND_CEILING)))
    low, high = minimum, min(int(stake / CENT), 10_000)
    best = None
    quote_available = False
    while low <= high:
        quantity = (low + high) // 2
        try:
            payload, summary = quote_order(pick, contract, long, quantity, book, now())
        except WagerError:
            high = quantity - 1
            continue
        quote_available = True
        if decimal(summary["total_estimate"]) <= stake:
            best = payload, summary
            low = quantity + 1
        else:
            high = quantity - 1
    if best is None:
        report("minimum order exceeds stake" if quote_available or minimum > int(stake / CENT)
               else "no fresh executable quote or insufficient depth")
        return None
    while best:
        payload, summary = best
        if summary["screened_ev"] >= MIN_SCREENED_EV:
            payload["manualOrderIndicator"] = "MANUAL_ORDER_INDICATOR_AUTOMATIC"
            return best
        quantity = payload["quantity"] - 1
        if quantity < minimum:
            break
        best = quote_order(pick, contract, long, quantity, book, now())
    report("screened EV below +5%")
    return None


def ledger(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    store = sqlite3.connect(path, timeout=5)
    store.execute("""CREATE TABLE IF NOT EXISTS tennis_orders (
        mode TEXT, event TEXT, utc_day TEXT, cents INTEGER, state TEXT,
        detail TEXT, PRIMARY KEY (mode, event)
    )""")
    return store


def reserve(store, mode, slug, summary, daily_cap):
    date = now().date().isoformat()
    cents = int(decimal(summary["total_estimate"]) * 100)
    with store:
        store.execute("BEGIN IMMEDIATE")
        if mode == "live" and store.execute(
                "SELECT 1 FROM tennis_orders WHERE mode='live' AND state='pending'").fetchone():
            return False
        spent, = store.execute("SELECT COALESCE(SUM(cents),0) FROM tennis_orders WHERE mode=? AND utc_day=?",
                              (mode, date)).fetchone()
        if daily_cap is not None and spent + cents > int(daily_cap * 100):
            return False
        row = store.execute("INSERT OR IGNORE INTO tennis_orders VALUES (?,?,?,?,?,?)",
                            (mode, slug, date, cents, "pending" if mode == "live" else "paper", json.dumps(summary)))
        return row.rowcount == 1


def submit(exchange, store, slug, payload):
    try:
        result = exchange.orders.create(payload)
        order_id = result.get("id")
        if not isinstance(order_id, str) or not order_id:
            raise ValueError("Missing exchange order ID")
    except Exception as exc:
        raise RuntimeError(f"Order state unknown for {slug}. Check Polymarket US activity before restarting.") from exc
    executions = result.get("executions") or []
    status = (executions[-1].get("order") or {}).get("state", "submitted") if executions else "submitted"
    with store:
        previous, = store.execute("SELECT detail FROM tennis_orders WHERE mode='live' AND event=?", (slug,)).fetchone()
        detail = {**json.loads(previous), "order_id": order_id}
        store.execute("UPDATE tennis_orders SET state=?, detail=? WHERE mode='live' AND event=?",
                      (status, json.dumps(detail), slug))
    return {"order_id": order_id, "state": status}


def price_candidate(candidate, stake):
    game_id, state, event = candidate
    slug = event["slug"]
    reasons = []
    try:
        event = public("/v1/events/slug/" + slug)["event"]
        if state.get("source") == "catalog" and (event.get("score") != state["score_text"]
                                                  or event.get("period") != state["period_text"]):
            return None, "catalog score changed before pricing"
        selection = match_selection(event, state, reasons.append)
        if selection is None:
            return None, reasons[-1]
        contract, long, pick = selection
        book = public("/v1/markets/" + contract["slug"] + "/book", fresh_book=True)["marketData"]
        ticket = order_for_stake(pick, contract, long, book, stake, reasons.append)
        if ticket:
            payload, summary = ticket
            return (game_id, slug, state, payload, {**summary, **pick, "score": state["score_text"],
                                                   "score_source": state.get("source", "websocket")}), None
    except DATA_ERRORS as exc:
        return None, f"pricing error ({type(exc).__name__})"
    return None, reasons[-1]


def run(args, exchange):
    scores, stop = Scores(), Event()
    mode = "live" if exchange else "paper"
    with closing(ledger(args.db)) as store, ThreadPoolExecutor(max_workers=4) as pool:
        if exchange and store.execute("SELECT 1 FROM tennis_orders WHERE mode='live' AND state='pending'").fetchone():
            raise RuntimeError("The ledger has an unresolved live order. Reconcile it with Polymarket US before trading.")
        worker = Thread(target=follow_scores, args=(stop, scores.publish), daemon=True)
        worker.start()
        heartbeat = refresh_catalog = 0
        events, paper_scores = {}, {}
        skipped = Counter()
        try:
            while not stop.is_set():
                if monotonic() >= refresh_catalog:
                    try:
                        events = live_catalog()
                    except DATA_ERRORS as exc:
                        events = {}
                        print(f"ITF market catalog unavailable: {type(exc).__name__}", flush=True)
                    if not exchange:
                        paper_scores = catalog_scores(events, paper_scores)
                    refresh_catalog = monotonic() + (15 if exchange else args.interval)
                active = scores.active_games()
                matched = [(game_id, state, event) for game_id, state in active
                           if (event := catalog_event(events, state)) is not None]
                socket_events = {event["slug"] for _, _, event in matched}
                matched.extend((slug, state, events[slug]) for slug, state in paper_scores.items()
                               if slug not in socket_events and fresh(state["score_utc"], SCORE_MAX_AGE))
                candidates = [(game_id, state, event) for game_id, state, event in matched
                              if not store.execute("SELECT 1 FROM tennis_orders WHERE mode=? AND event=?",
                                                   (mode, event["slug"])).fetchone()]
                if monotonic() >= heartbeat:
                    print(f"{mode.upper()} · {len(events)} ITF listings · {len(active)} fresh WebSocket scores "
                          f"· {len(paper_scores)} fresh paper catalog scores · {len(candidates)} queued for screening",
                          flush=True)
                    if skipped:
                        counts = " · ".join(f"{reason}: {count}" for reason, count in skipped.most_common())
                        print(f"Skipped since last update · {counts}", flush=True)
                        skipped.clear()
                    heartbeat = monotonic() + 30
                results = pool.map(lambda candidate: price_candidate(candidate, args.stake), candidates)
                for result, reason in results:
                    if reason:
                        skipped[reason] += 1
                        continue
                    game_id, slug, state, payload, summary = result
                    newest = paper_scores.get(slug, {}) if state.get("source") == "catalog" else scores.get(game_id)
                    if exchange and state.get("source") == "catalog":
                        continue
                    if (not fresh(state["score_utc"], SCORE_MAX_AGE)
                            or not fresh(summary["quote_utc"], BOOK_MAX_AGE)
                            or first_set_checkpoint(newest, now()) != summary["checkpoint"]
                            or any(newest.get(key) != state.get(key)
                                   for key in ("phase", "score_text", "period_text", "score_columns"))):
                        skipped["score, quote, or checkpoint changed before entry"] += 1
                        continue
                    if not reserve(store, mode, slug, summary, args.daily_cap):
                        skipped["already recorded, pending order, or daily cap"] += 1
                        continue
                    print(json.dumps({"mode": mode, "event": slug, **summary,
                                      **(submit(exchange, store, slug, payload) if exchange else {})}), flush=True)
                stop.wait(args.interval)
        finally:
            stop.set()
            worker.join(timeout=3)


def money(value):
    try:
        amount = decimal(value)
    except WagerError as exc:
        raise argparse.ArgumentTypeError("Enter a positive USD amount") from exc
    if amount <= 0 or amount % CENT:
        raise argparse.ArgumentTypeError("Enter a positive USD amount with at most two decimal places")
    return amount


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true", help="Submit real orders. Default is paper logging.")
    parser.add_argument("--stake", type=money, default=PAPER_STAKE, help="Maximum all-in cost per order in USD")
    parser.add_argument("--daily-cap", type=money, help="Maximum reserved exposure per UTC day in USD")
    parser.add_argument("--interval", type=int, default=5, help="Seconds between scans")
    parser.add_argument("--db", type=Path, default=Path("data/paper_trades/tennis_orders.sqlite3"))
    args = parser.parse_args()
    if args.interval < 3 or (args.daily_cap is not None and args.daily_cap < args.stake):
        parser.error("Use at least a 3-second scan interval and a daily cap at least as large as the stake.")
    if not all(load_elo_break_models()):
        parser.error("The ITF Elo ratings and set-break model artifacts are required.")
    context = nullcontext()
    if args.live:
        key = os.getenv("POLYMARKET_KEY_ID")
        secret = os.getenv("POLYMARKET_SECRET_KEY")
        if not key or not secret or args.daily_cap is None:
            parser.error("--live requires both Polymarket US credential variables and --daily-cap.")
        from polymarket_us import PolymarketUS

        context = PolymarketUS(key_id=key, secret_key=secret, timeout=6, max_retries=0)
    try:
        with context as exchange:
            run(args, exchange)
    except KeyboardInterrupt:
        print("Stopped.", flush=True)
    except (RuntimeError, sqlite3.Error) as exc:
        parser.exit(1, f"Stopped: {exc}\n")


if __name__ == "__main__":
    main()
