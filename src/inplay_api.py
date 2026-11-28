"""Independent live scanner so slow pregame model refreshes don't block quotes."""

import asyncio
import json
import os
import re
import sqlite3
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse

from .clob_book_stream import ClobBookStream
from .inplay_scan import MAX_AGE, SCORING_MARKETS, _books, as_num, priced_side, scan_catalog, scan_opportunities, utc

from .sports_inventory import scan_sports_events
from .college_ev_scan import scan_college
from .nfl_inplay_scan import scan_nfl
from .college_inplay_scan import scan_college_live
from .game_board_feed import FINAL_SCORE_AGE, MAX_SCORE_AGE, best_ask, follow_scores, moneyline_fixtures, scan_moneylines
from .halftime_paper import LEAGUES as HALFTIME_LEAGUES, screen_halftime
from .tennis_live import ITF_LEAGUES, observe_first_set_break, tennis_projection
from .tennis_wager import SCORE_MAX_AGE as ITF_SCORE_MAX_AGE, fresh as fresh_itf_score
from .us_wager import BOOK_MAX_AGE
from . import cs2_live_ev, pro_live_ev, wcbb_anytime
from .pro_halftime_orders import LIVE_STAKE

router = APIRouter()
ROOT = Path(__file__).resolve().parents[1]
PAGE = ROOT / "static" / "inplay.html"
RECENT_PICK_SECONDS = 120
MODEL_REFRESH_SECONDS = 8
SCORE_SOURCES = {"nba": "nba", "nfl": "nfl", "cbb": "mbb", "cfb": "cfb"}


def pro_snapshot_for_display(snapshot, now):
    scans = []
    for scan in snapshot["scans"]:
        games = []
        for game in scan.get("games", []):
            quotes = {side: quote for side, quote in game.get("quotes", {}).items()
                      if quote and (quoted := utc(quote.get("quote_utc")))
                      and 0 <= (now - quoted).total_seconds() <= pro_live_ev.MAX_BOOK_AGE}
            status = game["pricing_status"]
            if status == "priced" and not quotes:
                status = "quote_expired"
            games.append({**game, "quotes": quotes, "pricing_status": status})
        scans.append({**scan, "games": games} if "games" in scan else scan)
    return {**snapshot, "scans": scans}


def rank_opportunities(rows, games, now):
    """Return fresh candidates and identify the top positive screened EV per fixture."""
    fixtures = {game["event"]: game for game in games}
    ranked = []
    for row in rows:
        game = fixtures.get(row["event"])
        quote = utc(row.get("quote_utc"))
        if (game is None or game.get("phase") != row.get("phase")
                or quote is None or not -5 <= (now - quote).total_seconds() <= MAX_AGE
                or row["screened_ev"] <= 0):
            continue
        if row["phase"] == "pregame":
            tip = utc(game.get("start_utc"))
            if tip is None or tip <= now:
                continue
        ranked.append(row)

    ranked.sort(key=lambda row: (-row["screened_ev"], -row["probability"], row["token_id"]))
    picks = {}
    for row in ranked:
        picks.setdefault(row["event"], row["token_id"])
    return ranked, picks


def agrees_with_board(row, score, model_at):
    if score.get("phase") == "final":
        return False
    if score.get("phase") != "live":
        return True
    if row["phase"] == "pregame":
        return False
    latest = utc(score.get("score_utc"))
    if latest is None or model_at is None or latest < model_at:
        return True
    # Only compare period labels when both feeds use the same explicit format.
    model_period, score_period = (str(value or "").upper().replace(" ", "")
                                  for value in (row.get("period_text"), score.get("period_text")))
    if (re.fullmatch(r"[QH]\d+", model_period) and re.fullmatch(r"[QH]\d+", score_period)
            and model_period[0] == score_period[0] and model_period != score_period):
        return False
    home_away = re.fullmatch(r"(\d+)\s*[-–]\s*(\d+)", score.get("score_text", ""))
    return home_away is None or row.get("score") == [int(home_away[2]), int(home_away[1])]


class ScanState:
    def __init__(self):
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.workers = []
        self.market_books = ClobBookStream()
        self.revision = 0
        self.score_wake = {source: threading.Event() for source in SCORE_SOURCES.values()}
        self.recent_picks = {}
        self.snapshot = {"updated_utc": None, "coverage": {}, "opportunities": [], "warnings": []}
        self.catalog = {"cataloged": 0, "open_order_books": 0, "status": "waiting"}
        self.sports = {"status": "waiting", "games": []}
        self.has_complete_sports_slate = False
        self.moneyline_quotes = {}
        self.scores = {}
        self.live_sports = {}
        self.halftime = {"updated_utc": None, "leagues": list(HALFTIME_LEAGUES), "picks": [], "warnings": []}
        self.pro_halftime = {"updated_utc": None, "leagues": list(pro_live_ev.LEAGUES),
                             "scans": [], "entries": [], "paper_logging": False, "live_ordering": False}
        self.pro_halftime_orders = None
        self.wcbb = {"status": "waiting", "updated_utc": None, "games": [], "entries": [],
                     "performance": wcbb_anytime.performance([]), "orders": [], "warnings": []}
        self.wcbb_orders = None
        self.wcbb_books = None
        self.wcbb_model = None
        self.wcbb_order_read_error = None
        self.wcbb_settlement_error = None
        self.cs2 = {"updated_utc": None, "games": [], "warnings": [], "rating_sources": [], "round_feed": False}
        self.college = {"updated_utc": None, "games": [], "opportunities": [], "warnings": []}

    def start(self):
        self.stop.clear()
        tasks = ((self.live_loop,), (self.catalog_loop,), (self.sports_loop,), (self.college_loop,),
                 (self.board_loop,), (self.score_loop,), (self.halftime_loop,), (self.pro_halftime_loop,),
                 (self.wcbb_loop,), (self.wcbb_settlement_loop,), (self.cs2_loop,),
                 (self.market_books.follow, self.stop, self.wanted_tokens),
                 (self.league_live_loop, "nfl", scan_nfl),
                 (self.league_live_loop, "mbb", scan_college_live, "mbb"),
                 (self.league_live_loop, "cfb", scan_college_live, "cfb"))
        if self.wcbb_orders is not None:
            tasks += ((self.wcbb_order_loop,),)
        if self.wcbb_books is not None:
            tasks += ((self.wcbb_books.follow, self.stop),)
        for target, *args in tasks:
            worker = threading.Thread(target=target, args=args, daemon=True)
            worker.start()
            self.workers.append(worker)

    def shutdown(self):
        self.stop.set()
        for wake in self.score_wake.values():
            wake.set()
        for worker in self.workers:
            worker.join(timeout=1)
        self.workers.clear()

    def live_loop(self):
        while not self.stop.is_set():
            start = time.monotonic()
            self.score_wake["nba"].clear()
            try:
                snapshot = scan_opportunities(books_reader=self.rest_books)
            except Exception as exc:
                snapshot = {"updated_utc": None, "coverage": {}, "opportunities": [],
                            "warnings": [f"Live scan unavailable: {type(exc).__name__}"]}
            with self.lock:
                self.snapshot = snapshot
                self.revision += 1
            self.wait_for_scores("nba", start)

    def wait_for_scores(self, league, started):
        self.score_wake[league].wait(max(1, MODEL_REFRESH_SECONDS - (time.monotonic() - started)))
        self.stop.wait(max(0, 3 - (time.monotonic() - started)))

    def wanted_tokens(self):
        with self.lock:
            sources = self.snapshot, self.college, *self.live_sports.values()
            fixtures = self.sports.get("games", [])
        now = datetime.now(timezone.utc)
        return ({row["token_id"] for source in sources for row, _ in source.get("quote_candidates", [])}
                | {token for fixture in moneyline_fixtures(fixtures, now) for _, token in fixture["moneyline"]})

    def rest_books(self, tokens):
        streamed = self.market_books.snapshot(tokens)
        now = datetime.now(timezone.utc).timestamp()
        missing = []
        for token in tokens:
            live = streamed.get(token, {})
            stamp = as_num(live.get("timestamp"))
            if not live.get("closed") and (stamp is None or not -5 <= now - stamp / 1000 <= MAX_AGE
                                          or live.get("min_order_size") is None):
                missing.append(token)
        books = _books(missing) if missing else {}
        self.market_books.observe_market_terms(books)
        return books

    def halftime_loop(self):
        while not self.stop.is_set():
            started = time.monotonic()
            try:
                value = screen_halftime(book_stream=self.wcbb_books)
            except Exception as exc:
                value = {"updated_utc": None, "leagues": list(HALFTIME_LEAGUES), "picks": [],
                         "warnings": [f"Halftime screen unavailable: {type(exc).__name__}"]}
            with self.lock:
                self.halftime = value
            self.stop.wait(max(4, 12 - (time.monotonic() - started)))

    def pro_halftime_loop(self):
        live_orders = self.pro_halftime_orders
        logging = live_orders is None and os.getenv("PG_ENABLE_PRO_HALFTIME_PAPER") == "1"
        ledger = ROOT / pro_live_ev.DEFAULT_PAPER_DB
        stake = LIVE_STAKE if live_orders is not None else Decimal("10.00")

        def scan(league):
            try:
                return pro_live_ev.scan(league, stake, Decimal(".01"), paper_db=ledger if logging else None,
                                        halftime_only=True, live_orders=live_orders)
            except (sqlite3.Error, ValueError, OSError) as exc:
                return {"league": league, "status": "halftime_scan_unavailable", "error": type(exc).__name__}

        with ThreadPoolExecutor(max_workers=len(pro_live_ev.LEAGUES)) as pool:
            while not self.stop.is_set():
                started = time.monotonic()
                scans = list(pool.map(scan, pro_live_ev.LEAGUES))
                try:
                    entries = live_orders.recent() if live_orders is not None else pro_live_ev.recent_paper_entries(ledger)
                    warning = None
                    if any(entry.get("state") in ("pending", "unknown") for entry in entries):
                        warning = "An order state is unresolved. Check your Polymarket US activity."
                except (sqlite3.Error, ValueError, OSError):
                    entries, warning = [], "Halftime ledger unavailable."
                value = {"updated_utc": datetime.now(timezone.utc).isoformat(),
                         "leagues": list(pro_live_ev.LEAGUES), "scans": scans, "entries": entries,
                         "stake_cap": str(stake), "paper_logging": logging,
                         "live_ordering": live_orders is not None, "warning": warning}
                with self.lock:
                    self.pro_halftime = value
                self.stop.wait(max(1, 5 - (time.monotonic() - started)))

    def wcbb_loop(self):
        ledger = ROOT / (wcbb_anytime.MODEL_DB_PATH if self.wcbb_model else wcbb_anytime.DB_PATH)
        start_index = 0
        while not self.stop.is_set():
            started = time.monotonic()
            try:
                value = wcbb_anytime.scan(ledger, start_index=start_index, live_orders=self.wcbb_orders,
                                          book_stream=self.wcbb_books, model=self.wcbb_model)
                start_index += 1
            except Exception as exc:
                with self.lock:
                    value = {**self.wcbb, "status": "feed_unavailable", "updated_utc": None, "games": [],
                             "warnings": [f"WCBB scan unavailable ({type(exc).__name__})."]}
            with self.lock:
                self.wcbb = value
            self.stop.wait(max(3, 12 - (time.monotonic() - started)))

    def wcbb_settlement_loop(self):
        ledger = ROOT / (wcbb_anytime.MODEL_DB_PATH if self.wcbb_model else wcbb_anytime.DB_PATH)
        while not self.stop.is_set():
            try:
                wcbb_anytime.settle(ledger)
                error = None
            except (sqlite3.Error, ValueError, OSError) as exc:
                error = f"WCBB settlements unavailable ({type(exc).__name__})."
            with self.lock:
                self.wcbb_settlement_error = error
            self.stop.wait(60)

    def wcbb_order_loop(self):
        while not self.stop.is_set():
            try:
                self.wcbb_orders.refresh()
                error = None
            except Exception as exc:
                error = f"WCBB order status could not be refreshed ({type(exc).__name__})."
            with self.lock:
                self.wcbb_order_read_error = error
            self.stop.wait(10)

    def cs2_loop(self):
        reader = cs2_live_ev.Reader()
        with ThreadPoolExecutor(max_workers=8) as pool:
            while not self.stop.is_set():
                started = time.monotonic()
                try:
                    value = reader.snapshot(pool)
                except Exception as exc:
                    value = {"updated_utc": None, "games": [], "rating_sources": [], "round_feed": False,
                             "warnings": [f"CS2 reader unavailable ({type(exc).__name__})."]}
                with self.lock:
                    self.cs2 = value
                self.stop.wait(max(1, cs2_live_ev.PRICE_INTERVAL - (time.monotonic() - started)))

    def update_catalog(self, value):
        with self.lock:
            self.catalog = {**value, "observed_utc": datetime.now(timezone.utc).isoformat()}

    def catalog_loop(self):
        while not self.stop.is_set():
            try:
                scan_catalog(self.stop, self.update_catalog)
            except Exception:
                with self.lock:
                    self.catalog = {**self.catalog, "status": "partial"}
            self.stop.wait(15 * 60)

    def update_sports_progress(self, value):
        with self.lock:
            # Keep the last full slate visible while the new pass is incomplete.
            if not self.has_complete_sports_slate:
                self.sports = value

    def sports_loop(self):
        while not self.stop.is_set():
            try:
                value = scan_sports_events(self.stop, self.update_sports_progress)
                with self.lock:
                    if value["status"] == "complete":
                        self.has_complete_sports_slate = True
                        self.sports = value
                    elif value["games"] and not self.has_complete_sports_slate:
                        self.sports = value
                    else:
                        self.sports = {**self.sports, "status": "partial"}
            except Exception:
                with self.lock:
                    self.sports = {**self.sports, "status": "partial"}
            self.stop.wait(90)

    def board_loop(self):
        while not self.stop.is_set():
            started = time.monotonic()
            with self.lock:
                games = self.sports.get("games", [])
            result = scan_moneylines(games, books_reader=self.rest_books)
            with self.lock:
                self.moneyline_quotes = result
                self.revision += 1
                now = datetime.now(timezone.utc)
                self.scores = {key: row for key, row in self.scores.items()
                               if now - utc(row["score_utc"]) <= FINAL_SCORE_AGE}
            self.stop.wait(max(3, 10 - (time.monotonic() - started)))

    def update_score(self, game_id, update):
        with self.lock:
            previous = self.scores.get(game_id, {})
            if previous.get("phase") != "final":
                first_set_break = observe_first_set_break(previous, update, utc)
                self.scores[game_id] = {**update, "first_set_break_utc": first_set_break} if first_set_break else update
                self.revision += 1
                if any(previous.get(key) != update.get(key) for key in ("phase", "score_text", "period_text")):
                    league = update.get("event_slug", "").split("-")[0] or update.get("score_league", "").lower()
                    if league in SCORE_SOURCES:
                        self.score_wake[SCORE_SOURCES[league]].set()

    def score_loop(self):
        follow_scores(self.stop, self.update_score)

    def remember_picks(self, ranked, picks, now):
        current = {(row["event"], row["token_id"]): row for row in ranked}
        with self.lock:
            self.recent_picks = {key: row for key, row in self.recent_picks.items()
                                 if (now - utc(row["last_seen_utc"])).total_seconds() <= RECENT_PICK_SECONDS}
            for event, token in picks.items():
                key = event, token
                self.recent_picks[key] = {**current[key], "last_seen_utc": now.isoformat()}
            recent = {}
            for key, row in sorted(self.recent_picks.items(), key=lambda item: item[1]["last_seen_utc"], reverse=True):
                rows = recent.setdefault(key[0], [])
                if key not in current and len(rows) < 3:
                    rows.append(row)
            return {event: rows for event, rows in recent.items() if rows}

    def read(self):
        with self.lock:
            snapshot, catalog, sports = dict(self.snapshot), dict(self.catalog), dict(self.sports)
            college, live_sports = dict(self.college), list(self.live_sports.values())
            quotes, scores = dict(self.moneyline_quotes), dict(self.scores)
            halftime = dict(self.halftime)
            pro_halftime = dict(self.pro_halftime)
            wcbb = {**self.wcbb, "live_ordering": self.wcbb_orders is not None}
            wcbb_errors = (self.wcbb_settlement_error, self.wcbb_order_read_error)
            cs2 = self.cs2
        now = datetime.now(timezone.utc)
        fresh, games, observations = [], [], []
        sources = (snapshot, college, *live_sports)
        fixtures = moneyline_fixtures(sports.get("games", []), now)
        moneylines = {game["event"]: game["moneyline"] for game in fixtures}
        streamed = self.market_books.snapshot(
            {row["token_id"] for source in sources for row, _ in source.get("quote_candidates", [])}
            | {token for sides in moneylines.values() for _, token in sides})
        for source in sources:
            observed = utc(source.get("updated_utc"))
            if observed is None or not -5 <= (now - observed).total_seconds() <= MAX_AGE:
                continue
            observations.append(observed)
            views = {game["event"]: {**game, "state_observed_utc": observed.isoformat()}
                     for game in source.get("games", [])}
            games.extend(views.values())
            fresh.extend(source["opportunities"])
            for row, fees in source.get("quote_candidates", []):
                token = row["token_id"]
                live = streamed.get(token, {})
                rest = source["rest_books"].get(token, {})
                if live.get("closed"):
                    continue
                if (live.get("min_order_size") is not None
                        and (as_num(live.get("timestamp")) or 0) >= (as_num(rest.get("timestamp")) or 0)):
                    book, book_source = live, "websocket"
                else:
                    book, book_source = rest, "rest"
                priced = priced_side(book, row["probability"], fees, now)
                if priced is not None:
                    views[row["event"]]["priced_sides"] += 1
                    if priced["screened_ev"] > 0:
                        fresh.append({**row, **priced, "book_source": book_source})
        board = []
        score_states = {}
        catalog_at = utc(sports.get("observed_utc"))
        catalog_is_fresh = catalog_at is not None and -5 <= (now - catalog_at).total_seconds() <= MAX_SCORE_AGE.total_seconds()
        tennis_states = {}
        for game in sports.get("games", []):
            event = game["event"]
            state = {}
            score = scores.get(game.get("sports_game_id"))
            if score:
                max_age = FINAL_SCORE_AGE if score["phase"] == "final" else MAX_SCORE_AGE
                if now - utc(score["score_utc"]) <= max_age:
                    state = score
                    score_states[event] = state
            prices = {}
            if state.get("phase") != "final":
                prices = {row["token_id"]: row for row in quotes.get(event, [])
                          if -5 <= (now - utc(row["quote_utc"])).total_seconds() <= MAX_AGE}
                for side, token in moneylines.get(event, []):
                    if token not in streamed:
                        continue
                    book = streamed[token]
                    old = prices.get(token)
                    stamp = as_num(book.get("timestamp"))
                    if not book.get("closed") and old and stamp is not None and stamp / 1000 < utc(old["quote_utc"]).timestamp():
                        continue
                    prices.pop(token, None)
                    if quote := best_ask(book, now):
                        prices[token] = {"side": side, "token_id": token, **quote}
            current_score = state or (game if catalog_is_fresh else {})
            tennis = tennis_projection(game, current_score, now)
            if tennis:
                tennis_states[game.get("slug") or event] = (current_score, tennis)
            if state or prices or tennis:
                board.append({"event": event, "moneyline_quotes": list(prices.values()), "tennis_projection": tennis, **state})
        if isinstance(halftime.get("live_games"), list):
            live_games = []
            for game in halftime["live_games"]:
                current_score, tennis = tennis_states.get(game["event"], ({}, None))
                if not tennis or any(current_score.get(key) != game.get(key) for key in ("score_text", "period_text")):
                    # The venue can continue showing a match after its last score update.
                    if game.get("league") not in ITF_LEAGUES or fresh_itf_score(game.get("score_utc"), ITF_SCORE_MAX_AGE):
                        tennis = tennis_projection(game, game, now)
                    else:
                        tennis = None
                live_games.append({**game, "tennis_projection": tennis} if tennis else game)
            halftime["live_games"] = live_games
        # A confirmed live/final transition invalidates stale pregame EV immediately.
        times = {game["event"]: utc(game["state_observed_utc"]) for game in games}
        fresh = [row for row in fresh if agrees_with_board(
            row, score_states.get(row["event"], {}), times.get(row["event"]))]
        ranked, picks = rank_opportunities(fresh, games, now)
        recent = self.remember_picks(ranked, picks, now)
        sports["games"] = [{k: v for k, v in game.items() if k not in ("college_markets", "moneyline", "sports_game_id")}
                           for game in sports.get("games", [])]
        coverage = dict(snapshot.get("coverage", {}))
        for key in ("modeled_markets", "priced_sides"):
            coverage[key] = sum(game.get(key, 0) for game in games)
        halftime["picks"] = [pick for pick in halftime["picks"]
                             if -5 <= (now - utc(pick["quote_utc"])).total_seconds() <= BOOK_MAX_AGE]
        if any(wcbb_errors):
            wcbb["warnings"] = [*wcbb["warnings"], *(error for error in wcbb_errors if error)]
        public = {key: value for key, value in snapshot.items() if key not in ("quote_candidates", "rest_books")}
        return {**public, "catalog": catalog, "sports": sports, "stale": not observations,
                "updated_utc": max(observations).isoformat() if observations else snapshot.get("updated_utc"),
                "coverage": coverage, "games": games, "opportunities": ranked, "top_picks": picks, "recent_picks": recent,
                "board": board, "halftime_paper": halftime, "pro_halftime": pro_snapshot_for_display(pro_halftime, now),
                "wcbb_anytime": wcbb, "cs2": cs2_live_ev.read_for_display(cs2, now),
                "warnings": sorted({warning for source in sources for warning in source.get("warnings", [])}),
                "venue": "Polymarket CLOB (international)", "supported_markets": sorted(SCORING_MARKETS)}

    def league_live_loop(self, league, scanner, *args):
        while not self.stop.is_set():
            started = time.monotonic()
            self.score_wake[league].clear()
            with self.lock:
                fixtures = self.sports
            try:
                snapshot = scanner(fixtures, *args, books_reader=self.rest_books)
            except Exception as exc:
                snapshot = {"updated_utc": None, "games": [], "opportunities": [],
                            "warnings": [f"{league.upper()} live scan unavailable: {type(exc).__name__}"]}
            with self.lock:
                self.live_sports[league] = snapshot
                self.revision += 1
            self.wait_for_scores(league, started)

    def college_loop(self):
        while not self.stop.is_set():
            started = time.monotonic()
            with self.lock:
                fixtures = self.sports
            try:
                snapshot = scan_college(fixtures, books_reader=self.rest_books)
            except Exception as exc:
                snapshot = {"updated_utc": None, "games": [], "opportunities": [],
                            "warnings": [f"Pregame scan unavailable: {type(exc).__name__}"]}
            with self.lock:
                self.college = snapshot
                self.revision += 1
            self.stop.wait(max(4, 20 - (time.monotonic() - started)))


scanner = ScanState()


@router.get("/inplay", include_in_schema=False)
def scanner_page():
    return FileResponse(PAGE, headers={"Cache-Control": "no-store"})


@router.get("/api/inplay/opportunities")
def scanner_data():
    return JSONResponse(scanner.read(), headers={"Cache-Control": "no-store"})


@router.websocket("/ws/inplay")
async def scanner_socket(socket: WebSocket):
    origin, host = socket.headers.get("origin"), socket.headers.get("host")
    if origin and origin not in (f"http://{host}", f"https://{host}"):
        await socket.close(code=1008)
        return
    await socket.accept()
    revision, payload = None, None
    checked = sent = float("-inf")
    try:
        while not scanner.stop.is_set():
            now = time.monotonic()
            current = scanner.revision, scanner.market_books.revision
            if current != revision or now - checked >= 1:
                frame = json.dumps(await asyncio.to_thread(scanner.read), separators=(",", ":"), allow_nan=False)
                revision, checked = current, now
                if frame != payload or now - sent >= 5:
                    await socket.send_text(frame)
                    payload, sent = frame, now
            try:
                async with asyncio.timeout(0.25):
                    if (await socket.receive())["type"] == "websocket.disconnect":
                        return
            except TimeoutError:
                continue
    except (WebSocketDisconnect, OSError):
        return
