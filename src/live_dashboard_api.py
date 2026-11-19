"""Parlay Guard read-only dynamic dashboard.

Uses the project's existing immutable pre-tipoff forecast CSVs and
src.live_polymarket.get_live_quotes(). Never submits real or paper orders.

Install: add to src/api.py after app = FastAPI(...):
    from .live_dashboard_api import router as live_dashboard_router
    app.include_router(live_dashboard_router)
"""
from __future__ import annotations

import csv
import math
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from .live_polymarket import get_live_quotes

ROOT = Path(__file__).resolve().parents[1]
ET = ZoneInfo("America/New_York")
router = APIRouter(tags=["Live Dashboard"])

# Matches the conservative Elo strategy used in the user's backtest.
PRICE_PREMIUM = 0.02   # simulated extra 2 cents/share (not actual fees)
PROB_CUSHION = 0.05   # reduce Elo probability by five percentage points
MIN_ESTIMATED_ROI = 0.05
ROSTER_MAX_AGE_SECONDS = 4 * 3600
QUOTE_CACHE_SECONDS = 90

_quote_cache = {"key": None, "time": 0.0, "data": None}
_quote_lock = threading.Lock()


def _parse_utc(value):
    try:
        d = datetime.fromisoformat(str(value).strip().replace("Z", "+00:00"))
        if d.tzinfo is None:
            return None
        return d.astimezone(timezone.utc)
    except (ValueError, TypeError, AttributeError):
        return None


def _number(value):
    try:
        v = float(value)
        return v if math.isfinite(v) else None
    except (ValueError, TypeError):
        return None


def _latest_snapshot(date_str, v2=False):
    prefix = "live_predictions_v2" if v2 else "live_predictions"
    matches = sorted((ROOT / "models").glob(f"{prefix}_{date_str}_*.csv"),
                     key=lambda p: p.name, reverse=True)
    return matches[0] if matches else None


def _read_csv(path):
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def _load_roster(date_str, now):
    """Optional verified roster-check feed. No row is NOT a healthy lineup."""
    path = ROOT / "data" / "roster_checks.csv"
    checks = {}
    if not path.is_file():
        return checks
    for row in _read_csv(path):
        team = str(row.get("team", "")).strip().upper()
        if row.get("game_date") != date_str or not team:
            continue
        stamp = _parse_utc(row.get("checked_at_utc"))
        status = str(row.get("status", "")).strip().upper()
        if status not in {"CLEAR", "FLAG", "SKIP"} or stamp is None:
            continue
        if stamp > now or (now - stamp).total_seconds() > ROSTER_MAX_AGE_SECONDS:
            continue
        key = (date_str, team)
        if key not in checks or stamp > checks[key]["timestamp"]:
            checks[key] = {"status": status, "reason": str(row.get("reason", "")).strip(),
                           "timestamp": stamp}
    return checks


def _load_games():
    now = datetime.now(timezone.utc)
    date_str = now.astimezone(ET).date().isoformat()
    path = _latest_snapshot(date_str)
    if path is None:
        return {"date": date_str, "generated_at_utc": None, "source": None,
                "games": [], "notice": "No pre-tipoff forecast CSV for today. Run python -m src.predict_today first."}

    required = {"GAME_DATE", "GAME_ID", "AWAY_TEAM", "HOME_TEAM", "TIPOFF_UTC",
                "GENERATED_AT_UTC", "P_HOME_WIN_ELO"}
    records = _read_csv(path)
    if records and not required.issubset(records[0]):
        raise ValueError(f"Forecast {path.name} missing {sorted(required - set(records[0]))}")

    v2path = _latest_snapshot(date_str, v2=True)
    v2 = {}
    if v2path:
        for row in _read_csv(v2path):
            v2[(row.get("GAME_DATE"), row.get("AWAY_TEAM"), row.get("HOME_TEAM"))] = row
    roster = _load_roster(date_str, now)
    games = []
    seen = set()
    for row in records:
        if row.get("GAME_DATE") != date_str:
            continue
        tipoff = _parse_utc(row.get("TIPOFF_UTC"))
        generated = _parse_utc(row.get("GENERATED_AT_UTC"))
        p = _number(row.get("P_HOME_WIN_ELO"))
        if tipoff is None or generated is None or p is None or not (0 <= p <= 1):
            continue
        if not (generated <= now < tipoff):
            continue  # no retrospective or already-started games
        away, home = row["AWAY_TEAM"].strip().upper(), row["HOME_TEAM"].strip().upper()
        key = (date_str, away, home)
        if key in seen or away == home:
            continue
        seen.add(key)
        team_checks = []
        for team in (away, home):
            check = roster.get((date_str, team))
            team_checks.append({"team": team,
                                "status": check["status"] if check else "UNVERIFIED",
                                "reason": (check["reason"] or "Verified roster check") if check
                                          else "No recent verified roster check"})
        states = {c["status"] for c in team_checks}
        risk = ("SKIP" if "SKIP" in states else
                "UNVERIFIED" if "UNVERIFIED" in states else
                "FLAG" if "FLAG" in states else "CLEAR")
        v2row = v2.get(key, {})
        gb = _number(row.get("P_HOME_WIN_GB"))
        gb2 = _number(v2row.get("P_HOME_WIN_GB"))
        games.append({
            "game_id": str(row["GAME_ID"]).zfill(10), "date": date_str,
            "away": away, "home": home, "tipoff_utc": tipoff.isoformat(),
            "generated_at_utc": generated.isoformat(),
            "home_elo": _number(row.get("HOME_ELO")),
            "away_elo": _number(row.get("AWAY_ELO")),
            "p_home_elo": p, "p_home_gb": gb if gb is not None and 0 <= gb <= 1 else None,
            "p_home_gb_v2": gb2 if gb2 is not None and 0 <= gb2 <= 1 else None,
            "selected_team": home if p >= 0.5 else away,
            "selected_probability": max(p, 1-p),
            "roster_risk": risk, "roster_checks": team_checks,
        })
    games.sort(key=lambda g: g["tipoff_utc"])
    return {"date": date_str, "generated_at_utc": games[0]["generated_at_utc"] if games else None,
            "source": path.name, "games": games,
            "notice": "Only future-tipoff forecasts are shown. Roster data must be verified separately."}


def _market_result(game, quote):
    """Calculate the exact Elo 5pp cushion / >5% ROI screen, at +2c assumed price."""
    result = {"game_id": game["game_id"], "status": "NO_QUOTE", "reason": "No verified market quote",
              "ask": None, "effective_price": None, "estimated_roi": None,
              "adjusted_probability": max(0.0, game["selected_probability"] - PROB_CUSHION),
              "quoted_at_utc": None}
    tipoff = _parse_utc(game.get("tipoff_utc"))
    if tipoff is not None and datetime.now(timezone.utc) >= tipoff:
        result.update(status="SKIP", reason="Game has already started")
        return result
    if not isinstance(quote, dict) or quote.get("status") != "ok":
        if isinstance(quote, dict):
            result["reason"] = str(quote.get("reason") or result["reason"])
        return result
    side = quote.get("sides", {}).get(game["selected_team"], {})
    ask = _number(side.get("ask"))
    if ask is None or not (0 < ask < 1) or ask + PRICE_PREMIUM >= 1:
        return result
    effective = ask + PRICE_PREMIUM
    roi = result["adjusted_probability"] / effective - 1.0
    result.update(ask=ask, effective_price=effective, estimated_roi=roi)
    if game["roster_risk"] == "UNVERIFIED":
        result.update(status="SKIP", reason="Roster checks missing or stale")
    elif game["roster_risk"] == "SKIP":
        result.update(status="SKIP", reason="Significant roster risk")
    elif game["roster_risk"] == "FLAG":
        result.update(status="HOLD", reason="Roster uncertainty requires review")
    elif roi > MIN_ESTIMATED_ROI:
        result.update(status="REVIEW", reason="Estimated Elo edge; verify liquidity and model inputs")
    else:
        result.update(status="NO_EDGE", reason="Does not pass conservative EV screen")
    return result


@router.get("/live")
def live_dashboard():
    path = ROOT / "static" / "live.html"
    if not path.is_file():
        raise HTTPException(404, "static/live.html is missing")
    return FileResponse(path, media_type="text/html")


@router.get("/api/live/games")
def live_games():
    try:
        return _load_games()
    except (ValueError, OSError) as exc:
        raise HTTPException(503, f"Forecast data unavailable: {exc}") from exc


@router.get("/api/live/quotes")
def live_quotes():
    try:
        snapshot = _load_games()
    except (ValueError, OSError) as exc:
        raise HTTPException(503, f"Forecast data unavailable: {exc}") from exc
    games = snapshot["games"]
    if not games:
        return {"updated_at_utc": None, "results": [], "notice": snapshot["notice"]}

    key = tuple((g["away"], g["home"], g["date"]) for g in games)
    with _quote_lock:
        if (_quote_cache["key"] == key and _quote_cache["data"] is not None
                and time.monotonic() - _quote_cache["time"] < QUOTE_CACHE_SECONDS):
            quotes = _quote_cache["data"]
            fetched_at = _quote_cache["fetched_at"]
        else:
            quotes = None
            fetched_at = None
    if quotes is None:
        try:
            raw = get_live_quotes([{"away": g["away"], "home": g["home"], "date": g["date"]}
                                   for g in games])
        except Exception as exc:
            raise HTTPException(503, "Live Polymarket quotes unavailable; see server logs") from exc
        quotes = raw.get("games", {})
        fetched_at = datetime.now(timezone.utc).isoformat()
        with _quote_lock:
            _quote_cache.update(key=key, time=time.monotonic(), data=quotes, fetched_at=fetched_at)

    results = []
    for game in games:
        quote = quotes.get(f'{game["away"]}|{game["home"]}|{game["date"]}')
        results.append(_market_result(game, quote))
    return {"updated_at_utc": fetched_at, "results": results,
            "assumed_premium_cents": 2, "probability_cushion_pp": 5,
            "min_estimated_roi_pct": 5,
            "notice": "Read-only indicative best asks; depth, fees and execution not verified. No orders placed."}
