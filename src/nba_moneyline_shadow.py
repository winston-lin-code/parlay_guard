"""Forward paper test of NBA moneyline timing and order size on Polymarket US."""

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
import csv
from datetime import timedelta
from decimal import Decimal, ROUND_CEILING
import json
from math import isfinite
from pathlib import Path
import sqlite3
from tempfile import NamedTemporaryFile
from threading import Event, Semaphore, Thread
from time import monotonic, sleep

import requests

from . import halftime_paper as halftime
from .env_config import env_value
from .inplay_scan import scoreboard, utc
from .pregame_capture import signing_credentials
from .pro_live_ev import best_paper_quote
from .run_research import live_credentials
from .tennis_wager import BookStream
from .us_wager import BOOK_MAX_AGE, ONE, SCANNER_BOOKS, BookRateLimited, buy_levels, decimal, public, quote_order
from .wcbb_anytime import (MIN_EV, MIN_PRICE, MIN_PROBABILITY, SLIPPAGE, STAKE, display_team,
                           forecast, game_state, now, settled_payout, settlement_market_batches)

SUMMARY_URL = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/summary"
DEFAULT_OUTPUT = Path("data/paper_trades/nba_moneyline_shadow")
WINDOWS = ("anytime", "halftime_window")
SIZES = ("full", "half_shares")
SCAN_INTERVAL = 12
REST_FALLBACKS_PER_SCAN = 2
DELAYED_STREAM_WAIT = 0.75
SIGNAL_COLUMNS = (
    "window", "size", "game_id", "market", "long", "matchup", "team", "checkpoint", "clock", "away_score",
    "home_score", "probability", "screened_ev", "limit", "initial_ask", "initial_depth_at_limit", "score_utc",
    "initial_book_utc", "initial_book_source", "signal_utc", "full_quantity", "half_shares_quantity",
    "requested_delay_seconds", "delayed_book_utc", "delayed_book_source", "observed_utc",
    "observed_delay_seconds", "delayed_depth_at_limit", "reason",
    "quantity", "status", "estimated_cost", "payout", "settled_at", "net_pnl",
)


class Ledger:
    def __init__(self, output):
        output.mkdir(parents=True, exist_ok=True)
        self.output = output
        self.path = output / "study.sqlite3"
        with closing(self.connect()) as db, db:
            db.execute("""CREATE TABLE IF NOT EXISTS signals (
                game_id TEXT NOT NULL, window TEXT NOT NULL, market TEXT NOT NULL, signal TEXT NOT NULL,
                execution TEXT, payout TEXT, settled_at TEXT,
                PRIMARY KEY (game_id, window), UNIQUE (market, window)
            )""")

    def connect(self):
        return sqlite3.connect(self.path, timeout=5)

    def seen(self):
        with closing(self.connect()) as db:
            return set(db.execute("SELECT game_id, window FROM signals"))

    def reserve(self, signal, windows):
        saved = []
        with closing(self.connect()) as db, db:
            for window in windows:
                inserted = db.execute("""INSERT INTO signals (game_id, window, market, signal) VALUES (?, ?, ?, ?)
                    ON CONFLICT DO NOTHING""", (signal["game_id"], window, signal["market"], json.dumps(signal)))
                if inserted.rowcount:
                    saved.append(window)
        return saved

    def record_execution(self, game_id, windows, execution):
        with closing(self.connect()) as db, db:
            db.executemany("UPDATE signals SET execution=? WHERE game_id=? AND window=? AND execution IS NULL",
                           [(json.dumps(execution), game_id, window) for window in windows])

    def entries(self):
        with closing(self.connect()) as db:
            return [{**json.loads(signal), "window": window, "execution": json.loads(execution) if execution else {},
                     "payout": payout, "settled_at": settled_at}
                    for window, signal, execution, payout, settled_at in db.execute(
                        "SELECT window, signal, execution, payout, settled_at FROM signals ORDER BY rowid")]

    def settle(self):
        pending = [row for row in self.entries() if row["payout"] is None]
        for batch in settlement_market_batches(row["market"] for row in pending):
            markets = {market["slug"]: market for market in batch}
            checked = now().isoformat()
            updates = []
            for row in pending:
                if row["market"] in markets and (payout := settled_payout(markets[row["market"]], row["long"])) is not None:
                    updates.append((str(payout), checked, row["game_id"], row["window"]))
            with closing(self.connect()) as db, db:
                db.executemany("UPDATE signals SET payout=?, settled_at=? WHERE game_id=? AND window=? AND payout IS NULL",
                               updates)


def fresh_book(slug):
    return SCANNER_BOOKS.request(f"/v1/markets/{slug}/book", public, fresh_book=True)


class BookUnavailable(ValueError):
    pass


class OrderBooks:
    def __init__(self, stream):
        self.stream = stream
        self.initial_fallbacks = Semaphore(REST_FALLBACKS_PER_SCAN)
        self.delayed_fallbacks = Semaphore(REST_FALLBACKS_PER_SCAN)

    def read(self, slug, *, execution_at=None):
        if self.stream is not None:
            deadline = monotonic() + DELAYED_STREAM_WAIT
            while True:
                snapshot = self.stream.latest_snapshot(slug, clock=now, max_age=BOOK_MAX_AGE,
                                                       received_at_or_after=execution_at)
                if snapshot is not None:
                    book, received = snapshot
                    return book, received if execution_at else now(), "websocket"
                remaining = deadline - monotonic()
                if execution_at is None or self.stream.error or remaining <= 0:
                    break
                sleep(min(.05, remaining))
            budget = self.delayed_fallbacks if execution_at else self.initial_fallbacks
            if not budget.acquire(blocking=False):
                raise BookUnavailable("Waiting for a fresh streamed book · REST fallback budget reached")
        book = fresh_book(slug)
        return book, now(), "rest"


def book_timestamp(book, slug, reference):
    observed = utc(book.get("transactTime"))
    if book.get("marketSlug") == slug and observed and 0 <= (reference - observed).total_seconds() <= BOOK_MAX_AGE:
        return observed
    return None


def feed_issue(source, exc):
    if isinstance(exc, BookRateLimited):
        return f"{source} rate-limited"
    response = exc.response if isinstance(exc, requests.HTTPError) else None
    if response is not None:
        return f"{source} returned HTTP {response.status_code}"
    if str(exc) == "Upstream returned a cached live feed":
        return f"{source} returned a stale response"
    return f"{source} unavailable ({type(exc).__name__})"


def retry_espn(fetch, *args, **kwargs):
    try:
        return fetch(*args, **kwargs)
    except requests.HTTPError as exc:
        response = exc.response
        if response is None or response.status_code not in (502, 503, 504):
            raise
    return fetch(*args, **kwargs)


def signal_for(game, contract, assignments, books):
    try:
        summary = retry_espn(halftime.read_json, "GET", SUMMARY_URL, max_age=15, params={"event": game["id"]},
                             headers={"Cache-Control": "no-cache"})
    except halftime.FEED_ERRORS as exc:
        return None, feed_issue("ESPN win-probability feed", exc)
    if not isinstance(summary, dict) or not summary.get("winprobability") or not summary.get("plays"):
        return None, "ESPN provided no play-by-play win probability"
    try:
        book, received, source = books.read(contract["slug"])
    except BookUnavailable as exc:
        return None, str(exc)
    except halftime.FEED_ERRORS as exc:
        return None, feed_issue("Polymarket US order book", exc)
    quoted = book_timestamp(book, contract["slug"], received)
    if quoted is None or quoted < game["tip"]:
        return None, "No fresh Polymarket US order book"
    if book.get("state") != "MARKET_STATE_OPEN":
        return None, "Polymarket US book paused or closed"
    estimate = forecast(summary, game, quoted)
    if estimate is None:
        return None, "No eligible ESPN win probability for the quote-time score"
    home, score_at = estimate
    selected = "home" if home >= Decimal(".5") else "away"
    probability = home if selected == "home" else ONE - home
    if probability <= MIN_PROBABILITY:
        return None, "Win probability at or below 60%"
    long = assignments[selected]
    side = next(side for side in contract["marketSides"] if side["long"] is long)
    if side.get("tradable") is not True:
        return None, "Selected moneyline side unavailable"
    levels = buy_levels(book, long)
    ask = min((price for price, _ in levels), default=None)
    if ask is None:
        return None, "No executable ask for the selected team"
    if ask < MIN_PRICE:
        return None, "Best ask below 20 cents"
    offer = best_paper_quote(contract, side, float(probability), book, STAKE, SLIPPAGE, received)
    if not offer:
        return None, "Insufficient executable depth within the $5 cap"
    if offer["screened_ev"] < float(MIN_EV):
        return None, "Screened EV below 5%"
    full = offer["quantity"]
    small = full // 2
    minimum = int(decimal(contract["minimumTradeQty"]).to_integral_value(rounding=ROUND_CEILING))
    limit = decimal(offer["limit"])
    signal = {"game_id": game["id"], "market": contract["slug"], "long": long,
            "matchup": " vs ".join(display_team(game["teams"][key]) for key in ("away", "home")),
            "team": display_team(game["teams"][selected]), "checkpoint": game["checkpoint"],
            "clock": game["clock_text"], "away_score": game["scores"]["away"], "home_score": game["scores"]["home"],
            "probability": float(probability), "screened_ev": offer["screened_ev"], "limit": str(limit),
            "initial_ask": str(ask), "initial_depth_at_limit": str(sum(size for price, size in levels if price <= limit)),
            "initial_book_utc": quoted.isoformat(), "initial_book_source": source,
            "score_utc": score_at.isoformat(), "signal_utc": received.isoformat(),
            "full_quantity": full, "half_shares_quantity": small if small >= max(1, minimum) else 0}
    return signal, None


def delayed_execution(signal, contract, books):
    signalled_at = utc(signal["signal_utc"])
    execution_at = signalled_at + max(timedelta(microseconds=1), timedelta(seconds=signal["requested_delay_seconds"]))
    try:
        book, received, source = books.read(signal["market"], execution_at=execution_at)
        if received < execution_at:
            return {"reason": "book_received_before_execution_target"}
        quoted = book_timestamp(book, signal["market"], received)
        if quoted is None or quoted < utc(signal["initial_book_utc"]):
            return {"reason": "stale_or_wrong_book"}
        state = book.get("state")
        if state not in ("MARKET_STATE_OPEN", "MARKET_STATE_PAUSED", "MARKET_STATE_CLOSED"):
            return {"reason": "unknown_book_state"}
        limit = decimal(signal["limit"])
        long = signal["long"]
        depth = sum(size for price, size in buy_levels(book, long) if price <= limit) if state == "MARKET_STATE_OPEN" else 0
        result = {"delayed_book_utc": quoted.isoformat(), "delayed_book_source": source,
                  "observed_utc": received.isoformat(),
                  "observed_delay_seconds": round((received - signalled_at).total_seconds(), 3),
                  "delayed_depth_at_limit": str(depth), "reason": state}
        for size in SIZES:
            quantity = signal[f"{size}_quantity"]
            if quantity == 0:
                result[size] = {"status": "not_applicable"}
            elif depth < quantity:
                result[size] = {"status": "no_fill"}
            else:
                _, quote = quote_order({"side": signal["team"], "probability": signal["probability"]}, contract,
                                       long, quantity, book, received, slippage=SLIPPAGE)
                cost = decimal(quote["total_estimate"])
                if decimal(quote["limit"]) <= limit and cost <= STAKE:
                    result[size] = {"status": "simulated_fill", "cost": str(cost)}
                else:
                    result[size] = {"status": "no_fill"}
        return result
    except BookUnavailable as exc:
        return {"reason": str(exc)}
    except halftime.FEED_ERRORS as exc:
        return {"reason": feed_issue("Polymarket US delayed book", exc)}


def scan(ledger, latency, *, explain=False, stream=None):
    reference = now()
    started = reference.astimezone(halftime.ET)
    day = started.date()
    days = (day - timedelta(days=1), day) if started.hour < 6 else (day,)
    tips = (reference - timedelta(hours=8), reference + timedelta(hours=2))
    seen = ledger.seen()
    games, warnings = {}, []
    books = OrderBooks(stream)

    def check(game, match):
        windows = [window for window in WINDOWS if (game["id"], window) not in seen
                   and (window == "anytime" or game["inside_window"])]
        if not windows:
            return 0, None
        if match is None:
            return 0, "No uniquely matched Polymarket US moneyline"
        try:
            _, contract, assignments = match
            signal, reason = signal_for(game, contract, assignments, books)
        except halftime.FEED_ERRORS as exc:
            return 0, feed_issue("NBA pricing data", exc)
        if signal is None:
            return 0, reason
        signal["requested_delay_seconds"] = latency
        windows = ledger.reserve(signal, windows)
        if not windows:
            return 0, None
        sleep(latency)
        ledger.record_execution(game["id"], windows, delayed_execution(signal, contract, books))
        return len(windows), None

    with ThreadPoolExecutor(max_workers=6) as pool:
        inventory = pool.submit(halftime.live_inventory, "NBA", start_window=tips)
        for date in days:
            try:
                events = retry_espn(scoreboard, date)
            except halftime.FEED_ERRORS as exc:
                warnings.append(feed_issue("ESPN NBA scoreboard", exc))
                continue
            for event in events:
                if (event.get("season") or {}).get("type") not in (2, "2"):
                    continue
                if game := game_state(event, league="NBA"):
                    window = halftime.game_state("NBA", event)
                    game["inside_window"] = window is not None
                    if window:
                        game["checkpoint"] = window["checkpoint"]
                    games[game["id"]] = game
        try:
            listings = inventory.result()[0].get("NBA", [])
        except halftime.FEED_ERRORS as exc:
            return len(games), 0, [*warnings, feed_issue("Polymarket US NBA listings", exc)]
        matches = []
        for game in games.values():
            try:
                found = [match for slug, market in listings if (match := halftime.verified_contract(game, slug, market))]
            except halftime.FEED_ERRORS:
                found = []
            matches.append(found[0] if len(found) == 1 else None)
        if stream is not None:
            stream.set_markets(match[1]["slug"] for match in matches if match)
            if stream.error:
                warnings.append(stream.error)
        outcomes = list(pool.map(check, games.values(), matches))
    rejected = Counter(reason for _, reason in outcomes if reason)
    if rejected:
        warnings.append("Skipped · " + " · ".join(f"{count} {reason}" for reason, count in rejected.most_common()))
    if explain:
        for game, (_, reason) in zip(games.values(), outcomes):
            if reason:
                matchup = " vs ".join(display_team(game["teams"][side]) for side in ("away", "home"))
                warnings.append(f"{matchup} · {game['checkpoint']} {game['clock_text']} · {reason}")
    signals = sum(count for count, _ in outcomes)
    return len(games), signals, warnings


def trade_rows(entries):
    for entry in entries:
        execution = entry["execution"]
        for size in SIZES:
            attempt = execution.get(size, {})
            quantity = entry[f"{size}_quantity"]
            state = "not_applicable" if quantity == 0 else attempt.get("status", "unobserved")
            cost = decimal(attempt["cost"]) if state == "simulated_fill" else Decimal(0)
            pnl = Decimal(0) if state == "no_fill" else None
            if state == "simulated_fill" and entry["payout"] is not None:
                pnl = decimal(entry["payout"]) * quantity - cost
            yield {**{key: value for key, value in entry.items() if key != "execution"},
                   **{key: value for key, value in execution.items() if key not in SIZES},
                   "size": size, "quantity": quantity, "status": state, "estimated_cost": cost, "net_pnl": pnl}


def performance(trades, window, size):
    signals = [row for row in trades if row["window"] == window and row["size"] == size]
    fills = [row for row in signals if row["status"] == "simulated_fill"]
    settled = [row for row in fills if row["net_pnl"] is not None]
    no_fills = sum(row["status"] == "no_fill" for row in signals)
    observed = len(fills) + no_fills
    spend = sum(row["estimated_cost"] for row in settled)
    pnl = sum(row["net_pnl"] for row in settled)
    evaluated = len(settled) + no_fills
    return {"window": window, "size": size, "signals": len(signals), "simulated_fills": len(fills),
            "no_fills": no_fills, "unobserved": sum(row["status"] == "unobserved" for row in signals),
            "not_applicable": sum(row["status"] == "not_applicable" for row in signals),
            "fill_rate_observed": len(fills) / observed if observed else None,
            "settled_fills": len(settled), "wins": sum(decimal(row["payout"]) == ONE for row in settled),
            "losses": sum(decimal(row["payout"]) == 0 for row in settled), "settled_cost": spend, "net_pnl": pnl,
            "roi_on_settled_cost": pnl / spend if spend else None,
            "pnl_per_evaluated_signal": pnl / Decimal(evaluated) if evaluated else None}


def write_csv(path, rows, columns):
    temporary = None
    try:
        with NamedTemporaryFile(mode="w", encoding="utf-8", newline="", dir=path.parent, delete=False) as handle:
            temporary = Path(handle.name)
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def report(ledger):
    trades = list(trade_rows(ledger.entries()))
    stats = [performance(trades, window, size) for window in WINDOWS for size in SIZES]
    for name, rows, columns in (("signals.csv", trades, SIGNAL_COLUMNS), ("summary.csv", stats, list(stats[0]))):
        try:
            write_csv(ledger.output / name, rows, columns)
        except OSError as exc:
            print(f"Could not refresh {name} ({type(exc).__name__})", flush=True)
    for row in stats:
        roi = row["roi_on_settled_cost"]
        label = "n/a" if roi is None else f"{roi:+.2%}"
        rate = row["fill_rate_observed"]
        fill_rate = "n/a" if rate is None else f"{rate:.1%}"
        print(f"{row['window']:16} {row['size']:12} {row['signals']:4} signals | {row['simulated_fills']:4} filled | "
              f"{row['no_fills']:4} no fill | {row['unobserved']:4} unobserved | fill rate {fill_rate} | "
              f"net P&L {row['net_pnl']:+.2f} | settled ROI {label}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--latency-seconds", type=float, default=env_value("NBA_SHADOW_LATENCY_SECONDS") or "2",
                        help="Assumed order delay before fetching the second book (default 2 seconds)")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Directory for the SQLite ledger and CSVs")
    parser.add_argument("--explain", action="store_true", help="Show the reason each game was skipped")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--once", action="store_true", help="Run one scan, update settlements, and export reports")
    mode.add_argument("--report", action="store_true", help="Update settlements and export the existing ledger")
    args = parser.parse_args()
    if not isfinite(args.latency_seconds) or args.latency_seconds < 0:
        parser.error("--latency-seconds must be a finite, nonnegative number")
    ledger = Ledger(args.output)
    print(f"PAPER · NBA moneyline timing and size · {ledger.output.resolve()}", flush=True)
    stream = worker = None
    stop = Event()
    if not args.report:
        try:
            credentials = signing_credentials(*live_credentials(None))
        except (ValueError, ImportError):
            print("Books · paced Polymarket US REST · WebSocket credentials unavailable", flush=True)
        else:
            stream = BookStream(credentials, Event(), request_id="nba-moneyline-shadow")
            worker = Thread(target=stream.follow, args=(stop,), daemon=True)
            worker.start()
            print("Books · Polymarket US WebSocket with limited REST fallback", flush=True)
    last_settlement = float("-inf")
    try:
        while True:
            started = monotonic()
            if not args.report:
                games, signals, warnings = scan(ledger, args.latency_seconds, explain=args.explain, stream=stream)
                print(f"{games} live NBA games · {signals} new signals", flush=True)
                for warning in warnings:
                    print(warning, flush=True)
            if started - last_settlement >= 60:
                try:
                    ledger.settle()
                except halftime.FEED_ERRORS as exc:
                    print(f"Exchange settlements unavailable ({type(exc).__name__})", flush=True)
                last_settlement = started
            report(ledger)
            if args.once or args.report:
                return
            sleep(max(1, SCAN_INTERVAL - (monotonic() - started)))
    except KeyboardInterrupt:
        return
    finally:
        stop.set()
        if worker is not None:
            worker.join(timeout=2)


if __name__ == "__main__":
    main()
