"""Independent live scanner so slow pregame model refreshes don't block quotes."""

import re
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import FileResponse, JSONResponse

from .inplay_scan import MAX_AGE, SCORING_MARKETS, scan_catalog, scan_opportunities, utc

from .sports_inventory import scan_sports_events
from .college_ev_scan import scan_college
from .game_board_feed import FINAL_SCORE_AGE, MAX_SCORE_AGE, follow_scores, scan_moneylines

router = APIRouter()
PAGE = Path(__file__).resolve().parents[1] / "static" / "inplay.html"


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
    home_away = re.fullmatch(r"(\d+)\s*[-–]\s*(\d+)", score.get("score_text", ""))
    return home_away is None or row.get("score") == [int(home_away[2]), int(home_away[1])]


class ScanState:
    def __init__(self):
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.workers = []
        self.snapshot = {"updated_utc": None, "coverage": {}, "opportunities": [], "warnings": []}
        self.catalog = {"cataloged": 0, "open_order_books": 0, "status": "waiting"}
        self.sports = {"status": "waiting", "games": []}
        self.has_complete_sports_slate = False
        self.moneyline_quotes = {}
        self.scores = {}
        self.college = {"updated_utc": None, "games": [], "opportunities": [], "warnings": []}

    def start(self):
        self.stop.clear()
        for target in (self.live_loop, self.catalog_loop, self.sports_loop, self.college_loop, self.board_loop, self.score_loop):
            worker = threading.Thread(target=target, daemon=True)
            worker.start()
            self.workers.append(worker)

    def shutdown(self):
        self.stop.set()
        for worker in self.workers:
            worker.join(timeout=1)
        self.workers.clear()

    def live_loop(self):
        while not self.stop.is_set():
            start = time.monotonic()
            try:
                snapshot = scan_opportunities()
            except Exception as exc:
                snapshot = {"updated_utc": None, "coverage": {}, "opportunities": [],
                            "warnings": [f"Live scan unavailable: {type(exc).__name__}"]}
            with self.lock:
                self.snapshot = snapshot
            self.stop.wait(max(4, 20 - (time.monotonic() - start)))

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
            result = scan_moneylines(games)
            with self.lock:
                self.moneyline_quotes = result
                now = datetime.now(timezone.utc)
                self.scores = {key: row for key, row in self.scores.items()
                               if now - utc(row["score_utc"]) <= FINAL_SCORE_AGE}
            self.stop.wait(max(3, 10 - (time.monotonic() - started)))

    def update_score(self, game_id, update):
        with self.lock:
            if self.scores.get(game_id, {}).get("phase") != "final":
                self.scores[game_id] = update

    def score_loop(self):
        follow_scores(self.stop, self.update_score)

    def read(self):
        with self.lock:
            snapshot, catalog, sports = dict(self.snapshot), dict(self.catalog), dict(self.sports)
            college, quotes, scores = dict(self.college), dict(self.moneyline_quotes), dict(self.scores)
        now = datetime.now(timezone.utc)
        fresh, games, observations = [], [], []
        for source in (snapshot, college):
            observed = utc(source.get("updated_utc"))
            if observed is None or not -5 <= (now - observed).total_seconds() <= MAX_AGE:
                continue
            observations.append(observed)
            games.extend({**game, "state_observed_utc": observed.isoformat()} for game in source.get("games", []))
            fresh.extend(source["opportunities"])
        board = []
        score_states = {}
        for game in sports.get("games", []):
            event = game["event"]
            state = {}
            score = scores.get(game.get("sports_game_id"))
            if score:
                max_age = FINAL_SCORE_AGE if score["phase"] == "final" else MAX_SCORE_AGE
                if now - utc(score["score_utc"]) <= max_age:
                    state = score
                    score_states[event] = state
            prices = [] if state.get("phase") == "final" else [
                row for row in quotes.get(event, []) if -5 <= (now - utc(row["quote_utc"])).total_seconds() <= MAX_AGE
            ]
            if state or prices:
                board.append({"event": event, "moneyline_quotes": prices, **state})
        # A confirmed live/final transition invalidates stale pregame EV immediately.
        model_at = utc(snapshot.get("updated_utc"))
        fresh = [row for row in fresh if agrees_with_board(row, score_states.get(row["event"], {}), model_at)]
        ranked, picks = rank_opportunities(fresh, games, now)
        sports["games"] = [{k: v for k, v in game.items() if k not in ("college_markets", "moneyline", "sports_game_id")}
                           for game in sports.get("games", [])]
        coverage = dict(snapshot.get("coverage", {}))
        for key in ("modeled_markets", "priced_sides"):
            coverage[key] = sum(game.get(key, 0) for game in games)
        return {**snapshot, "catalog": catalog, "sports": sports, "stale": not observations,
                "updated_utc": max(observations).isoformat() if observations else snapshot.get("updated_utc"),
                "coverage": coverage, "games": games, "opportunities": ranked, "top_picks": picks,
                "board": board,
                "warnings": sorted(set(snapshot.get("warnings", []) + college.get("warnings", []))),
                "venue": "Polymarket CLOB (international)", "supported_markets": sorted(SCORING_MARKETS)}

    def college_loop(self):
        while not self.stop.is_set():
            started = time.monotonic()
            with self.lock:
                fixtures = self.sports
            try:
                snapshot = scan_college(fixtures)
            except Exception as exc:
                snapshot = {"updated_utc": None, "games": [], "opportunities": [],
                            "warnings": [f"Pregame scan unavailable: {type(exc).__name__}"]}
            with self.lock:
                self.college = snapshot
            self.stop.wait(max(4, 20 - (time.monotonic() - started)))


scanner = ScanState()


@router.get("/inplay", include_in_schema=False)
def scanner_page():
    return FileResponse(PAGE, headers={"Cache-Control": "no-store"})


@router.get("/api/inplay/opportunities")
def scanner_data():
    return JSONResponse(scanner.read(), headers={"Cache-Control": "no-store"})
