"""Local live runner for ITF singles match-winner orders on Polymarket US."""

import argparse
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
from websockets.exceptions import InvalidStatus, WebSocketException
from websockets.sync.client import connect

from .game_board_feed import follow_scores
from .halftime_paper import MIN_PROBABILITY, MIN_SCREENED_EV, PAPER_STAKE
from .inplay_scan import utc
from .pregame_capture import MARKETS_WS, auth_headers, get_credentials
from .tennis_live import (
    ITF_LEAGUES, first_set_checkpoint, load_elo_break_models, next_point_scores, observe_first_set_break,
    tennis_projection,
)
from .us_wager import BOOK_MAX_AGE, CENT, SLUG, WagerError, decimal, name, public, quote_order, team_names

SCORE_MAX_AGE = 20
STREAM_BOOK_MAX_AGE = 3
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


class BookStream:
    def __init__(self, market, credentials, wake):
        self.market = market
        self.credentials = credentials
        self.wake = wake
        self.snapshot = None
        self.error = None

    def follow(self, stop):
        subscription = {"subscribe": {"requestId": "itf-score-paper", "marketSlugs": [self.market],
                                      "subscriptionType": "SUBSCRIPTION_TYPE_MARKET_DATA"}}
        while not stop.is_set():
            try:
                with connect(MARKETS_WS, additional_headers=auth_headers(self.credentials),
                             open_timeout=8, close_timeout=1) as socket:
                    socket.send(json.dumps(subscription))
                    while not stop.is_set():
                        try:
                            frame = socket.recv(timeout=1)
                        except TimeoutError:
                            continue
                        if frame == "ping":
                            socket.send("pong")
                            continue
                        message = json.loads(frame)
                        if not isinstance(message, dict):
                            continue
                        if message.get("error"):
                            self.error = "Polymarket US rejected the book subscription."
                            return
                        book = message.get("marketData")
                        if not isinstance(book, dict) or book.get("marketSlug") != self.market:
                            continue
                        full = all(isinstance(book.get(side), list) for side in ("bids", "offers"))
                        previous = self.snapshot
                        observed = utc(book.get("transactTime"))
                        if not observed or not full or book.get("state") != "MARKET_STATE_OPEN":
                            self.snapshot = None
                        elif previous and observed < utc(previous[0]["transactTime"]):
                            continue
                        else:
                            self.snapshot = (book, now())
                            if not previous:
                                print(f"Receiving full US book for {self.market}", flush=True)
                        self.wake.set()
            except InvalidStatus as exc:
                if exc.response.status_code in (401, 403):
                    self.error = "Polymarket US rejected the book WebSocket credentials."
                    return
            except (OSError, ValueError, WebSocketException, TimeoutError):
                pass
            finally:
                self.snapshot = None
                self.wake.set()
            stop.wait(2)

    def usable_after(self, score):
        snapshot = self.snapshot
        received = utc(score.get("score_utc"))
        if snapshot and received and snapshot[1] >= received:
            observed = utc(snapshot[0].get("transactTime"))
            if observed and 0 <= (now() - observed).total_seconds() <= STREAM_BOOK_MAX_AGE:
                return snapshot
        return None


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


def match_selection(event, score, *, score_based=False):
    slug = event.get("slug") or ""
    first, separator, second = str(event.get("title") or "").partition(" vs. ")
    if (not separator or not SLUG.fullmatch(slug) or (score.get("event_slug") and slug != score["event_slug"])
            or any(" / " in side or " - " in side for side in (first, second))
            or event.get("live") is not True or event.get("ended") is True
            or event.get("active") is not True or event.get("closed") is not False
            or not fresh(score.get("score_utc"), SCORE_MAX_AGE)):
        return None
    contracts = [market for market in event.get("markets", [])
                 if market.get("slug") == f"aec-{slug}" and market.get("marketType") == "moneyline"
                 and market.get("sportsMarketType") == "tennis_match_winner"
                 and market.get("active") is True and market.get("closed") is False]
    if len(contracts) != 1:
        return None
    contract = contracts[0]
    sides = contract.get("marketSides") or []
    if len(sides) != 2 or {side.get("long") for side in sides} != {True, False}:
        return None
    ordered = []
    for column in score["score_columns"]:
        matches = [side for side in sides if name(column) in team_names(side)]
        if len(matches) != 1:
            return None
        ordered.append(matches[0])
    if len(ordered) != 2 or ordered[0] is ordered[1]:
        return None
    names = []
    for side in ordered:
        labels = [label for label in (first, second) if name(label) in team_names(side)]
        if len(labels) != 1:
            return None
        names.append(labels[0])
    if len(set(names)) != 2:
        return None
    match = {"event": slug, "title": event["title"]}
    ordered_score = {**score, "score_order": " / ".join(names)}
    projection = tennis_projection(match, ordered_score, now())
    if not projection or not projection["elo_prior"] or (not score_based and not projection["checkpoint"]):
        return None
    first_probability = projection["first_probability"]
    selected = int(first_probability < .5)
    probability = (first_probability, 1 - first_probability)[selected]
    displayed_probability = probability
    if score_based:
        successors = next_point_scores(score["score_text"], score["period_text"])
        if not successors:
            return None
        for next_score, next_period in successors:
            alternative = tennis_projection(match, {**ordered_score, "score_text": next_score,
                                                     "period_text": next_period})
            if not alternative:
                return None
            probability = min(probability, alternative["first_probability" if selected == 0 else "second_probability"])
    if probability < MIN_PROBABILITY or ordered[selected].get("tradable") is not True:
        return None
    pick = {"side": names[selected], "probability": probability, "checkpoint": projection["checkpoint"]}
    if score_based:
        pick.update(displayed_probability=displayed_probability, assumed_unseen_points=1)
    return contract, ordered[selected]["long"], pick


def order_for_stake(pick, contract, long, book, stake):
    # Find the largest whole-contract quantity that fits the worst-case limit and fee.
    minimum = max(1, int(decimal(contract["minimumTradeQty"]).to_integral_value(rounding=ROUND_CEILING)))
    low, high = minimum, min(int(stake / CENT), 10_000)
    best = None
    while low <= high:
        quantity = (low + high) // 2
        try:
            payload, summary = quote_order(pick, contract, long, quantity, book, now())
        except WagerError:
            high = quantity - 1
            continue
        if decimal(summary["total_estimate"]) <= stake:
            best = payload, summary
            low = quantity + 1
        else:
            high = quantity - 1
    while best:
        payload, summary = best
        if summary["screened_ev"] >= MIN_SCREENED_EV:
            payload["manualOrderIndicator"] = "MANUAL_ORDER_INDICATOR_AUTOMATIC"
            return best
        quantity = payload["quantity"] - 1
        if quantity < minimum:
            break
        best = quote_order(pick, contract, long, quantity, book, now())
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


def price_candidate(candidate, stake, stream=None):
    game_id, state, event = candidate
    slug = event["slug"]
    try:
        if stream:
            snapshot = stream.usable_after(state)
            if not snapshot:
                return None
        else:
            event = public("/v1/events/slug/" + slug)["event"]
        selection = match_selection(event, state, score_based=stream is not None)
        if selection is None:
            return None
        contract, long, pick = selection
        book = snapshot[0] if stream else public(
            "/v1/markets/" + contract["slug"] + "/book", fresh_book=True
        )["marketData"]
        ticket = order_for_stake(pick, contract, long, book, stake)
        if ticket:
            payload, summary = ticket
            detail = {**summary, **pick, "score": state["score_text"]}
            if stream and stream.usable_after(state) is not snapshot:
                return None
            if stream:
                detail.update(score_received_at=state["score_utc"], book_received_at=snapshot[1].isoformat())
            return game_id, slug, state, payload, detail
    except DATA_ERRORS as exc:
        print(f"Skipped {slug}: {type(exc).__name__}", flush=True)
    return None


def run(args, exchange, stream_credentials=None):
    scores, stop, wake = Scores(), Event(), Event()
    stream = BookStream("aec-" + args.stream_event, stream_credentials, wake) if args.stream_event else None
    mode = "live" if exchange else "paper"
    with closing(ledger(args.db)) as store, ThreadPoolExecutor(max_workers=4) as pool:
        if exchange and store.execute("SELECT 1 FROM tennis_orders WHERE mode='live' AND state='pending'").fetchone():
            raise RuntimeError("The ledger has an unresolved live order. Reconcile it with Polymarket US before trading.")
        def publish(game_id, state):
            scores.publish(game_id, state)
            if stream:
                wake.set()

        workers = [Thread(target=follow_scores, args=(stop, publish), daemon=True)]
        if stream:
            workers.append(Thread(target=stream.follow, args=(stop,), daemon=True))
        for worker in workers:
            worker.start()
        heartbeat = refresh_catalog = 0
        events = {}
        try:
            while not stop.is_set():
                wake.clear()
                if stream and stream.error:
                    raise RuntimeError(stream.error)
                if monotonic() >= refresh_catalog:
                    try:
                        if stream:
                            event = public("/v1/events/slug/" + args.stream_event)["event"]
                            events = {args.stream_event: event}
                        else:
                            events = live_catalog()
                    except DATA_ERRORS as exc:
                        events = {}
                        print(f"ITF market catalog unavailable: {type(exc).__name__}", flush=True)
                    refresh_catalog = monotonic() + 15
                active = scores.active_games()
                candidates = [(game_id, state, event) for game_id, state in active
                              if (event := catalog_event(events, state)) is not None
                              and not store.execute("SELECT 1 FROM tennis_orders WHERE mode=? AND event=?",
                                                    (mode, event["slug"])).fetchone()]
                if monotonic() >= heartbeat:
                    print(f"{mode.upper()} · {len(events)} ITF listings · {len(active)} fresh score streams "
                          f"· {len(candidates)} matched and eligible for pricing", flush=True)
                    heartbeat = monotonic() + 30
                results = pool.map(lambda candidate: price_candidate(candidate, args.stake, stream), candidates)
                for result in results:
                    if not result:
                        continue
                    game_id, slug, state, payload, summary = result
                    newest = scores.get(game_id)
                    if (not fresh(state["score_utc"], SCORE_MAX_AGE)
                            or not fresh(summary["quote_utc"], BOOK_MAX_AGE)
                            or (not stream and first_set_checkpoint(newest, now()) != summary["checkpoint"])
                            or any(newest.get(key) != state.get(key)
                                   for key in ("phase", "score_text", "period_text", "score_columns"))):
                        continue
                    if stream:
                        current = stream.usable_after(newest)
                        if not current or current[1].isoformat() != summary["book_received_at"]:
                            continue
                    if not reserve(store, mode, slug, summary, args.daily_cap):
                        continue
                    print(json.dumps({"mode": mode, "event": slug, **summary,
                                      **(submit(exchange, store, slug, payload) if exchange else {})}), flush=True)
                wake.wait(args.interval)
        finally:
            stop.set()
            for worker in workers:
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
    parser.add_argument("--stream-event", help="Paper-only ITF event slug for streamed score-based entries")
    parser.add_argument("--env-file", type=Path, help="Read book WebSocket credentials for --stream-event")
    args = parser.parse_args()
    if args.interval < 3 or (args.daily_cap is not None and args.daily_cap < args.stake):
        parser.error("Use at least a 3-second scan interval and a daily cap at least as large as the stake.")
    if not all(load_elo_break_models()):
        parser.error("The ITF Elo ratings and set-break model artifacts are required.")
    if args.stream_event and (args.live or not SLUG.fullmatch(args.stream_event)
                              or args.stream_event.split("-", 1)[0].upper() not in ITF_LEAGUES):
        parser.error("--stream-event requires an ITF event slug in paper mode")
    if args.env_file and not args.stream_event:
        parser.error("--env-file requires --stream-event")
    context = nullcontext()
    if args.live:
        key = os.getenv("POLYMARKET_KEY_ID")
        secret = os.getenv("POLYMARKET_SECRET_KEY")
        if not key or not secret or args.daily_cap is None:
            parser.error("--live requires both Polymarket US credential variables and --daily-cap.")
        from polymarket_us import PolymarketUS

        context = PolymarketUS(key_id=key, secret_key=secret, timeout=6, max_retries=0)
    try:
        stream_credentials = get_credentials(False, args.env_file) if args.stream_event else None
        with context as exchange:
            run(args, exchange, stream_credentials)
    except KeyboardInterrupt:
        print("Stopped.", flush=True)
    except (RuntimeError, ValueError, OSError, ImportError, sqlite3.Error) as exc:
        parser.exit(1, f"Stopped: {exc}\n")


if __name__ == "__main__":
    main()
