"""Independent live scanner so slow pregame model refreshes don't block quotes."""

import threading
import time
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import FileResponse, JSONResponse

from .inplay_scan import MAX_AGE, SCORING_MARKETS, scan_catalog, scan_opportunities, utc

from .sports_inventory import scan_sports_events
from .college_ev_scan import scan_college

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


class ScanState:
    def __init__(self):
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.workers = []
        self.snapshot = {"updated_utc": None, "coverage": {}, "opportunities": [], "warnings": []}
        self.catalog = {"cataloged": 0, "open_order_books": 0, "status": "waiting"}
        self.sports = {"status": "waiting", "games": []}
        self.has_complete_sports_slate = False
        self.college = {"updated_utc": None, "games": [], "opportunities": [], "warnings": []}

    def start(self):
        self.stop.clear()
        for target in (self.live_loop, self.catalog_loop, self.sports_loop, self.college_loop):
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

    def read(self):
        with self.lock:
            snapshot, catalog, sports = dict(self.snapshot), dict(self.catalog), dict(self.sports)
            college = dict(self.college)
        now = datetime.now(timezone.utc)
        fresh, games, observations = [], [], []
        for source in (snapshot, college):
            observed = utc(source.get("updated_utc"))
            if observed is None or not -5 <= (now - observed).total_seconds() <= MAX_AGE:
                continue
            observations.append(observed)
            games.extend(source.get("games", []))
            fresh.extend(source["opportunities"])
        ranked, picks = rank_opportunities(fresh, games, now)
        sports["games"] = [{k: v for k, v in game.items() if k != "college_markets"}
                           for game in sports.get("games", [])]
        coverage = dict(snapshot.get("coverage", {}))
        for key in ("modeled_markets", "priced_sides"):
            coverage[key] = sum(game.get(key, 0) for game in games)
        return {**snapshot, "catalog": catalog, "sports": sports, "stale": not observations,
                "updated_utc": max(observations).isoformat() if observations else snapshot.get("updated_utc"),
                "coverage": coverage, "games": games, "opportunities": ranked, "top_picks": picks,
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
                            "warnings": [f"College scan unavailable: {type(exc).__name__}"]}
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
