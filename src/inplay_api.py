"""Independent live scanner so slow pregame model refreshes don't block quotes."""

import threading
import time
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import FileResponse, JSONResponse

from .inplay_scan import MAX_AGE, SCORING_MARKETS, scan_catalog, scan_opportunities, utc

router = APIRouter()
PAGE = Path(__file__).resolve().parents[1] / "static" / "inplay.html"


class ScanState:
    def __init__(self):
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.workers = []
        self.snapshot = {"updated_utc": None, "coverage": {}, "opportunities": [], "warnings": []}
        self.catalog = {"cataloged": 0, "open_order_books": 0, "status": "waiting"}

    def start(self):
        self.stop.clear()
        for target in (self.live_loop, self.catalog_loop):
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

    def read(self):
        with self.lock:
            snapshot, catalog = dict(self.snapshot), dict(self.catalog)
        now = datetime.now(timezone.utc)
        observed = utc(snapshot.get("updated_utc"))
        stale = observed is None or not -5 <= (now - observed).total_seconds() <= MAX_AGE
        fresh = []
        if not stale:
            for row in snapshot["opportunities"]:
                quote = utc(row.get("quote_utc"))
                if quote is not None and -5 <= (now - quote).total_seconds() <= MAX_AGE:
                    fresh.append(row)
        return {**snapshot, "catalog": catalog, "stale": stale, "opportunities": fresh,
                "venue": "Polymarket CLOB (international)", "supported_markets": sorted(SCORING_MARKETS)}


scanner = ScanState()


@router.get("/inplay", include_in_schema=False)
def scanner_page():
    return FileResponse(PAGE, headers={"Cache-Control": "no-store"})


@router.get("/api/inplay/opportunities")
def scanner_data():
    return JSONResponse(scanner.read(), headers={"Cache-Control": "no-store"})
