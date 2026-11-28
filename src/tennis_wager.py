"""Local live runner for ITF singles match-winner orders on Polymarket US."""

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing, nullcontext
from datetime import datetime, timedelta, timezone
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
PREMATCH_LEAD = timedelta(minutes=15)
DELAYED_START_GRACE = timedelta(minutes=30)
DATA_ERRORS = (requests.RequestException, ValueError, KeyError, TypeError)


def now():
    return datetime.now(timezone.utc)


def fresh(timestamp, max_age):
    value = utc(timestamp)
    return value is not None and -5 <= (now() - value).total_seconds() <= max_age


def rejected(diagnostic, reason, **details):
    if diagnostic is not None:
        diagnostic.clear()
        diagnostic.update(reason=reason, **details)
    return None


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
            if (update.get("score_source") == "polymarket_us" and previous
                    and utc(update["score_utc"]) <= utc(previous["score_utc"])):
                return
            self.games[game_id] = {**update, "first_set_break_utc": observe_first_set_break(previous, update, utc)}

    def discard(self, game_id):
        with self.lock:
            self.games.pop(game_id, None)

    def get(self, game_id):
        with self.lock:
            return self.games.get(game_id, {})

    def active_games(self):
        with self.lock:
            return [(game_id, state) for game_id, state in self.games.items() if state["phase"] == "live"
                    and fresh(state["score_utc"], SCORE_MAX_AGE)]


class BookStream:
    def __init__(self, credentials, wake, *, request_id="itf-score-paper"):
        self.credentials = credentials
        self.wake = wake
        self.request_id = request_id
        self.lock = Lock()
        self.markets = frozenset()
        self.snapshots = {}
        self.refreshed_scores = {}
        self.error = None

    def set_markets(self, markets):
        markets = frozenset(markets)
        with self.lock:
            if markets != self.markets:
                self.markets = markets
                self.snapshots.clear()
                self.refreshed_scores = {market: timestamp for market, timestamp in self.refreshed_scores.items()
                                         if market in markets}
                self.wake.set()

    def publish(self, markets, book):
        if not isinstance(book, dict) or book.get("marketSlug") not in markets:
            return
        slug = book["marketSlug"]
        observed = utc(book.get("transactTime"))
        full = all(isinstance(book.get(side), list) for side in ("bids", "offers"))
        with self.lock:
            if markets != self.markets:
                return
            previous = self.snapshots.get(slug)
            if not observed or not full or book.get("state") != "MARKET_STATE_OPEN":
                self.snapshots.pop(slug, None)
            elif previous and observed < utc(previous[0]["transactTime"]):
                return
            else:
                self.snapshots[slug] = book, now()
                if not previous and "local_book_source" not in book:
                    print(f"Receiving full US book for {slug}", flush=True)
            self.wake.set()

    def follow(self, stop):
        while not stop.is_set():
            markets = self.markets
            if not markets:
                stop.wait(0.25)
                continue
            subscription = {"subscribe": {"requestId": self.request_id, "marketSlugs": sorted(markets),
                                          "subscriptionType": "SUBSCRIPTION_TYPE_MARKET_DATA"}}
            try:
                with connect(MARKETS_WS, additional_headers=auth_headers(self.credentials),
                             open_timeout=8, close_timeout=1) as socket:
                    socket.send(json.dumps(subscription))
                    while not stop.is_set() and markets == self.markets:
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
                        self.publish(markets, message.get("marketData"))
            except InvalidStatus as exc:
                if exc.response.status_code in (401, 403):
                    self.error = "Polymarket US rejected the book WebSocket credentials."
                    return
            except (OSError, ValueError, WebSocketException, TimeoutError):
                pass
            finally:
                with self.lock:
                    self.snapshots.clear()
                self.wake.set()
            if markets == self.markets:
                stop.wait(2)

    def latest(self, market, *, clock, max_age, after=None):
        snapshot = self.latest_snapshot(market, clock=clock, max_age=max_age, after=after)
        return snapshot[0] if snapshot else None

    def latest_snapshot(self, market, *, clock, max_age, after=None, received_at_or_after=None):
        with self.lock:
            snapshot = self.snapshots.get(market)
        if snapshot is None:
            return None
        book, received = snapshot
        observed = utc(book.get("transactTime"))
        if (observed is not None and 0 <= (clock() - observed).total_seconds() <= max_age
                and (after is None or observed > after)
                and (received_at_or_after is None or received >= received_at_or_after)):
            return snapshot
        return None

    def usable_after(self, market, score, *, diagnostic=None):
        with self.lock:
            snapshot = self.snapshots.get(market)
        if not snapshot:
            return rejected(diagnostic, "no_open_book_snapshot")
        received = utc(score.get("score_received_at") or score.get("score_utc"))
        if not received:
            return rejected(diagnostic, "missing_score_receipt_time")
        if snapshot[1] < received:
            return rejected(diagnostic, "book_received_before_score")
        observed = utc(snapshot[0].get("transactTime"))
        age = (now() - observed).total_seconds() if observed else None
        if age is None or not 0 <= age <= STREAM_BOOK_MAX_AGE:
            return rejected(diagnostic, "stale_or_invalid_book_timestamp", book_age_seconds=age)
        return snapshot

    def refresh_after(self, market, score, *, diagnostic=None):
        timestamp = score["score_utc"]
        if not fresh(timestamp, SCORE_MAX_AGE):
            return rejected(diagnostic, "stale_score")
        with self.lock:
            markets = self.markets
            if market not in markets or self.refreshed_scores.get(market) == timestamp:
                return None
            self.refreshed_scores[market] = timestamp
        try:
            book = public(f"/v1/markets/{market}/book", fresh_book=True)["marketData"]
        except DATA_ERRORS as exc:
            return rejected(diagnostic, "rest_book_refresh_failed", error_type=type(exc).__name__)
        streamed = self.usable_after(market, score)
        if streamed:
            return streamed
        if (not isinstance(book, dict) or book.get("marketSlug") != market
                or book.get("state") != "MARKET_STATE_OPEN"
                or not all(isinstance(book.get(side), list) for side in ("bids", "offers"))):
            return rejected(diagnostic, "rest_book_unavailable")
        self.publish(markets, {**book, "local_book_source": "rest_refresh"})
        return self.usable_after(market, score, diagnostic=diagnostic)


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


def stream_catalog():
    reference = now()
    events = {}
    for league in sorted(ITF_LEAGUES):
        for offset in range(0, 1000, 50):
            page = public(f"/v2/leagues/{league.lower()}/events?limit=50&offset={offset}",
                          fresh_score=True)["events"]
            received_at = now()
            for event in page:
                slug = event.get("slug") or ""
                start = utc(event.get("startTime"))
                near_start = start and -DELAYED_START_GRACE <= start - reference <= PREMATCH_LEAD
                if slug.startswith(league.lower() + "-") and (event.get("live") is True or near_start):
                    if tennis_contract(event):
                        events[slug] = {**event, "score_received_at": received_at}
            if len(page) < 50:
                break
        else:
            raise WagerError(f"{league} event catalog exceeds its request limit.")
    return events


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


def tennis_contract(event):
    slug = event.get("slug") or ""
    first, separator, second = str(event.get("title") or "").partition(" vs. ")
    if (not separator or not SLUG.fullmatch(slug)
            or any(" / " in side or " - " in side for side in (first, second))
            or event.get("ended") is True
            or event.get("active") is not True or event.get("closed") is not False):
        return None
    contracts = [market for market in event.get("markets", [])
                 if market.get("slug") == f"aec-{slug}" and market.get("marketType") == "moneyline"
                 and market.get("sportsMarketType") == "tennis_match_winner"
                 and market.get("active") is True and market.get("closed") is False]
    if len(contracts) != 1:
        return None
    return contracts[0]


def us_score(event, received_at):
    contract = tennis_contract(event)
    state = event.get("eventState") or {}
    if (not contract or event.get("live") is not True or state.get("type") != "tennis"
            or state.get("live") is not True or state.get("ended") is True):
        return None
    timestamp = utc(state.get("updatedAt"))
    score, period = state.get("score"), state.get("period")
    if not (timestamp and isinstance(score, str) and score and isinstance(period, str) and period):
        return None
    if timestamp > received_at:
        return None
    sides = contract.get("marketSides") or []
    if len(sides) != 2:
        return None
    players = {team.get("ordering"): team.get("abbreviation") or team.get("name")
               for side in sides if (team := side.get("team"))}
    if set(players) != {"home", "away"} or not all(players.values()) or players["home"] == players["away"]:
        return None
    slug = event["slug"]
    columns = [players["home"], players["away"]]
    return {
        "event_slug": slug, "score_league": slug.split("-", 1)[0].upper(),
        "score_columns": columns, "score_order": " / ".join(columns), "phase": "live",
        "score_text": score, "period_text": period, "score_source": "polymarket_us",
        "score_utc": timestamp.isoformat(), "score_received_at": received_at.isoformat(),
    }


def match_selection(event, score, *, score_based=False, diagnostic=None):
    slug = event.get("slug") or ""
    if not fresh(score.get("score_utc"), SCORE_MAX_AGE):
        return rejected(diagnostic, "stale_score")
    if (score.get("event_slug") and slug != score["event_slug"]) or event.get("live") is not True:
        return rejected(diagnostic, "score_and_event_mismatch")
    contract = tennis_contract(event)
    if contract is None:
        return rejected(diagnostic, "missing_match_winner_contract")
    first, _, second = event["title"].partition(" vs. ")
    sides = contract.get("marketSides") or []
    if len(sides) != 2 or {side.get("long") for side in sides} != {True, False}:
        return rejected(diagnostic, "ambiguous_market_sides")
    ordered = []
    for column in score["score_columns"]:
        matches = [side for side in sides if name(column) in team_names(side)]
        if len(matches) != 1:
            return rejected(diagnostic, "ambiguous_score_columns")
        ordered.append(matches[0])
    if len(ordered) != 2 or ordered[0] is ordered[1]:
        return rejected(diagnostic, "ambiguous_score_columns")
    names = []
    for side in ordered:
        labels = [label for label in (first, second) if name(label) in team_names(side)]
        if len(labels) != 1:
            return rejected(diagnostic, "ambiguous_player_names")
        names.append(labels[0])
    if len(set(names)) != 2:
        return rejected(diagnostic, "ambiguous_player_names")
    match = {"event": slug, "title": event["title"]}
    ordered_score = {**score, "score_order": " / ".join(names)}
    projection = tennis_projection(match, ordered_score, now())
    if not projection:
        return rejected(diagnostic, "unsupported_point_score")
    if not projection["elo_prior"]:
        return rejected(diagnostic, "missing_elo_rating")
    if not score_based and not projection["checkpoint"]:
        return rejected(diagnostic, "outside_set_break_window")
    first_probability = projection["first_probability"]
    selected = int(first_probability < .5)
    probability = (first_probability, 1 - first_probability)[selected]
    displayed_probability = probability
    if score_based:
        successors = next_point_scores(score["score_text"], score["period_text"])
        if not successors:
            return rejected(diagnostic, "cannot_stress_test_next_point")
        for next_score, next_period in successors:
            alternative = tennis_projection(match, {**ordered_score, "score_text": next_score,
                                                     "period_text": next_period})
            if not alternative:
                return rejected(diagnostic, "cannot_stress_test_next_point")
            probability = min(probability, alternative["first_probability" if selected == 0 else "second_probability"])
    if probability < MIN_PROBABILITY:
        return rejected(diagnostic, "probability_below_floor", side=names[selected],
                        displayed_probability=displayed_probability, conservative_probability=probability,
                        minimum_probability=MIN_PROBABILITY)
    if ordered[selected].get("tradable") is not True:
        return rejected(diagnostic, "side_unavailable")
    pick = {"side": names[selected], "probability": probability, "checkpoint": projection["checkpoint"]}
    if score_based:
        pick.update(displayed_probability=displayed_probability, assumed_unseen_points=1)
    return contract, ordered[selected]["long"], pick


def order_for_stake(pick, contract, long, book, stake, *, diagnostic=None):
    # Find the largest whole-contract quantity that fits the worst-case limit and fee.
    minimum = max(1, int(decimal(contract["minimumTradeQty"]).to_integral_value(rounding=ROUND_CEILING)))
    max_quantity = min(int(stake / CENT), 10_000)
    low, high = minimum, max_quantity
    best = None
    over_budget = None
    quote_error = None
    while low <= high:
        quantity = (low + high) // 2
        try:
            payload, summary = quote_order(pick, contract, long, quantity, book, now())
        except WagerError as exc:
            quote_error = str(exc)
            high = quantity - 1
            continue
        if decimal(summary["total_estimate"]) <= stake:
            best = payload, summary
            low = quantity + 1
        else:
            over_budget = summary["total_estimate"]
            high = quantity - 1
    if not best:
        if over_budget is not None or minimum > max_quantity:
            return rejected(diagnostic, "stake_too_small", stake=str(stake), quoted_cost=over_budget)
        return rejected(diagnostic, "no_executable_quote", detail=quote_error)
    best_failed_quote = best[1]
    while best:
        payload, summary = best
        if summary["screened_ev"] >= MIN_SCREENED_EV:
            payload["manualOrderIndicator"] = "MANUAL_ORDER_INDICATOR_AUTOMATIC"
            return best
        if summary["screened_ev"] > best_failed_quote["screened_ev"]:
            best_failed_quote = summary
        quantity = payload["quantity"] - 1
        if quantity < minimum:
            break
        best = quote_order(pick, contract, long, quantity, book, now())
    return rejected(diagnostic, "roi_below_floor", limit=best_failed_quote["limit"],
                    quantity=best_failed_quote["quantity"], cost=best_failed_quote["total_estimate"],
                    screened_roi=best_failed_quote["screened_ev"], minimum_roi=MIN_SCREENED_EV)


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


def price_candidate(candidate, stake, stream=None, *, refresh_book=False, diagnostic=None):
    game_id, state, event = candidate
    slug = event["slug"]
    try:
        if stream:
            snapshot = stream.usable_after(f"aec-{slug}", state, diagnostic=diagnostic)
            if not snapshot and not refresh_book:
                return None
        else:
            event = public("/v1/events/slug/" + slug)["event"]
        selection = match_selection(event, state, score_based=stream is not None, diagnostic=diagnostic)
        if selection is None:
            return None
        contract, long, pick = selection
        if stream and not snapshot:
            snapshot = stream.refresh_after(contract["slug"], state, diagnostic=diagnostic)
            if not snapshot:
                return None
        book = snapshot[0] if stream else public(
            "/v1/markets/" + contract["slug"] + "/book", fresh_book=True
        )["marketData"]
        ticket = order_for_stake(pick, contract, long, book, stake, diagnostic=diagnostic)
        if ticket:
            payload, summary = ticket
            detail = {**summary, **pick, "score": state["score_text"]}
            if stream and stream.usable_after(contract["slug"], state) is not snapshot:
                return rejected(diagnostic, "book_changed_during_pricing")
            if stream:
                detail.update(score_received_at=state.get("score_received_at", state["score_utc"]),
                              score_source_updated_at=state["score_utc"], book_received_at=snapshot[1].isoformat(),
                              book_source=book.get("local_book_source", "websocket"))
            return game_id, slug, state, payload, detail
    except DATA_ERRORS as exc:
        print(f"Skipped {slug}: {type(exc).__name__}", flush=True)
        return rejected(diagnostic, "pricing_error", error_type=type(exc).__name__)
    return None


def report_diagnostics(events, scores, active, candidates, latest, rejected_counts):
    active_slugs = {slug for slug, _ in active}
    candidate_slugs = {event["slug"] for _, _, event in candidates}
    matches = {}
    reference = now()
    for slug, event in events.items():
        if event.get("live") is not True:
            continue
        state = scores.get(slug)
        if slug in candidate_slugs:
            matches[slug] = latest.get(slug, {"reason": "awaiting_pricing"})
        elif slug in active_slugs:
            matches[slug] = {"reason": "already_logged"}
        else:
            observed = utc(state.get("score_utc"))
            matches[slug] = (
                {"reason": "stale_score", "score_age_seconds": round((reference - observed).total_seconds(), 1)}
                if observed else {"reason": "no_usable_us_score"}
            )
    print(json.dumps({"diagnostic": "tennis_pricing", "matches": matches,
                      "pricing_skips_since_last_status": dict(rejected_counts)}), flush=True)


def run(args, exchange, stream_credentials=None):
    scores, stop, wake = Scores(), Event(), Event()
    stream = BookStream(stream_credentials, wake) if stream_credentials else None
    if args.stream_event and stream:
        stream.set_markets([f"aec-{args.stream_event}"])
    mode = "live" if exchange else "paper"
    with closing(ledger(args.db)) as store, ThreadPoolExecutor(max_workers=4) as pool:
        if exchange and store.execute("SELECT 1 FROM tennis_orders WHERE mode='live' AND state='pending'").fetchone():
            raise RuntimeError("The ledger has an unresolved live order. Reconcile it with Polymarket US before trading.")
        worker = (Thread(target=stream.follow, args=(stop,), daemon=True) if stream
                  else Thread(target=follow_scores, args=(stop, scores.publish), daemon=True))
        worker.start()
        heartbeat = refresh_catalog = 0
        events = {}
        latest = {}
        rejected_counts = Counter()

        def check(candidate):
            diagnostic = {} if args.explain else None
            result = price_candidate(candidate, args.stake, stream, refresh_book=args.refresh_book,
                                     diagnostic=diagnostic)
            return candidate[2]["slug"], result, diagnostic

        def record_skip(slug, diagnostic):
            if args.explain:
                latest[slug] = diagnostic
                rejected_counts[diagnostic["reason"]] += 1

        try:
            while not stop.is_set():
                wake.clear()
                if stream and stream.error:
                    raise RuntimeError(stream.error)
                if monotonic() >= refresh_catalog:
                    try:
                        if args.stream_event:
                            event = public("/v1/events/slug/" + args.stream_event, fresh_score=True)["event"]
                            events = {args.stream_event: {**event, "score_received_at": now()}}
                        elif args.stream_all:
                            events = stream_catalog()
                        else:
                            events = live_catalog()
                    except DATA_ERRORS as exc:
                        events = {}
                        print(f"ITF market catalog unavailable: {type(exc).__name__}", flush=True)
                    if args.stream_all:
                        stream.set_markets(f"aec-{slug}" for slug in events)
                    if stream:
                        for slug, event in events.items():
                            if update := us_score(event, event["score_received_at"]):
                                scores.publish(slug, update)
                            else:
                                scores.discard(slug)
                    refresh_catalog = monotonic() + (args.interval if stream else 15)
                active = scores.active_games()
                if stream:
                    active = [(slug, state) for slug, state in active if slug in events]
                candidates = [(game_id, state, event) for game_id, state in active
                              if (event := catalog_event(events, state)) is not None
                              and not store.execute("SELECT 1 FROM tennis_orders WHERE mode=? AND event=?",
                                                    (mode, event["slug"])).fetchone()]
                report_due = monotonic() >= heartbeat
                if report_due:
                    reported = sum(event.get("live") is True for event in events.values())
                    print(f"{mode.upper()} · {len(events)} ITF listings ({reported} live) · {len(active)} fresh scores "
                          f"· {len(candidates)} matched and eligible for pricing", flush=True)
                    heartbeat = monotonic() + 30
                for slug, result, diagnostic in pool.map(check, candidates):
                    if not result:
                        if diagnostic:
                            record_skip(slug, diagnostic)
                        continue
                    game_id, slug, state, payload, summary = result
                    newest = scores.get(game_id)
                    if (not fresh(state["score_utc"], SCORE_MAX_AGE)
                            or not fresh(summary["quote_utc"], BOOK_MAX_AGE)
                            or (not stream and first_set_checkpoint(newest, now()) != summary["checkpoint"])
                            or any(newest.get(key) != state.get(key)
                                   for key in ("phase", "score_text", "period_text", "score_columns"))):
                        record_skip(slug, {"reason": "score_or_book_changed_before_entry"})
                        continue
                    if stream:
                        current = stream.usable_after(f"aec-{slug}", newest)
                        if not current or current[1].isoformat() != summary["book_received_at"]:
                            record_skip(slug, {"reason": "book_changed_before_entry"})
                            continue
                    if not reserve(store, mode, slug, summary, args.daily_cap):
                        record_skip(slug, {"reason": "ledger_or_daily_cap"})
                        continue
                    if args.explain:
                        latest[slug] = {"reason": "paper_entry_logged"}
                    print(json.dumps({"mode": mode, "event": slug, **summary,
                                      **(submit(exchange, store, slug, payload) if exchange else {})}), flush=True)
                if args.explain and report_due:
                    report_diagnostics(events, scores, active, candidates, latest, rejected_counts)
                    latest.clear()
                    rejected_counts.clear()
                wake.wait(args.interval)
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
    streaming = parser.add_mutually_exclusive_group()
    streaming.add_argument("--stream-all", action="store_true", help="Discover and stream ITF singles in paper mode")
    streaming.add_argument("--stream-event", help="Stream one ITF singles event in paper mode")
    parser.add_argument("--explain", action="store_true", help="Report score and pricing rejections in streamed paper mode")
    parser.add_argument("--refresh-book", action="store_true",
                        help="Try a REST book refresh once per qualifying score when streaming has no usable book")
    parser.add_argument("--env-file", type=Path, help="Read book WebSocket credentials for streaming")
    args = parser.parse_args()
    if args.interval < 3 or (args.daily_cap is not None and args.daily_cap < args.stake):
        parser.error("Use at least a 3-second scan interval and a daily cap at least as large as the stake.")
    if not all(load_elo_break_models()):
        parser.error("The ITF Elo ratings and set-break model artifacts are required.")
    use_stream = args.stream_all or args.stream_event
    if args.live and use_stream:
        parser.error("Streaming score-based entries are paper-only")
    if args.stream_event and (not SLUG.fullmatch(args.stream_event)
                              or args.stream_event.split("-", 1)[0].upper() not in ITF_LEAGUES):
        parser.error("--stream-event requires an ITF event slug")
    if (args.env_file or args.explain or args.refresh_book) and not use_stream:
        parser.error("--env-file, --explain, and --refresh-book require --stream-all or --stream-event")
    context = nullcontext()
    if args.live:
        key = os.getenv("POLYMARKET_KEY_ID")
        secret = os.getenv("POLYMARKET_SECRET_KEY")
        if not key or not secret or args.daily_cap is None:
            parser.error("--live requires both Polymarket US credential variables and --daily-cap.")
        from polymarket_us import PolymarketUS

        context = PolymarketUS(key_id=key, secret_key=secret, timeout=6, max_retries=0)
    try:
        stream_credentials = get_credentials(False, args.env_file) if use_stream else None
        with context as exchange:
            run(args, exchange, stream_credentials)
    except KeyboardInterrupt:
        print("Stopped.", flush=True)
    except (RuntimeError, ValueError, OSError, ImportError, sqlite3.Error) as exc:
        parser.exit(1, f"Stopped: {exc}\n")


if __name__ == "__main__":
    main()
