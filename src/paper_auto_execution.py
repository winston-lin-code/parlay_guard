"""Opt-in PAPER-ONLY NBA moneyline order-intent journal.

Consumes the existing src.auto_live.run_update() snapshot. Uses observed
international CLOB quotes for research and makes *assumed*, not real, fills.
No exchange credentials, HTTP requests, real orders, or settlement claims.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB = Path("data/paper_trades/auto_execution.sqlite3")
NBA_SLUG = re.compile(r"^nba-[a-z]{2,4}-[a-z]{2,4}-\d{4}-\d{2}-\d{2}$")

# First-pilot simulation parameters, not evidence of a trading edge.
STAKE_CENTS = 500
DAILY_CAP_CENTS = 2000
MIN_STRESSED_ROI = 0.10
PROBABILITY_HAIRCUT = 0.05
PRICE_PENALTY = 0.01
MAX_QUOTE_AGE = timedelta(seconds=120)
MAX_FORECAST_AGE = timedelta(hours=6)
MIN_LEAD_TIME = timedelta(minutes=10)
MAX_LEAD_TIME = timedelta(hours=3)


def _utc(value):
    try:
        if isinstance(value, datetime):
            dt = value
        else:
            dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        return dt.astimezone(timezone.utc) if dt.tzinfo else None
    except (TypeError, ValueError, OverflowError):
        return None


def _num(value):
    try:
        x = float(value)
        return x if math.isfinite(x) else None
    except (ValueError, TypeError):
        return None


def _candidate(game, now):
    """Return (candidate, skip_reason). Fail closed on missing data."""
    slug = str(game.get("slug") or "")
    if not NBA_SLUG.fullmatch(slug):
        return None, "invalid_event"
    tip = _utc(game.get("tipoff_utc"))
    if tip is None or not MIN_LEAD_TIME <= tip - now <= MAX_LEAD_TIME:
        return None, "outside_pregame_window"
    home = str(game.get("home") or "").upper()
    away = str(game.get("away") or "").upper()
    forecast = game.get("elo_winner_forecast", game.get("winner_forecast")) or {}
    chosen = str(forecast.get("predicted_winner") or "").upper()
    p_home = _num(forecast.get("home_probability"))
    generated = _utc(forecast.get("forecast_generated_at_utc"))
    if (home == away or chosen not in (home, away) or p_home is None
            or not 0 < p_home < 1 or generated is None
            or not timedelta(0) <= now - generated <= MAX_FORECAST_AGE):
        return None, "missing_or_stale_forecast"
    if chosen != (home if p_home >= 0.5 else away):
        return None, "inconsistent_winner"
    probability = p_home if chosen == home else 1 - p_home
    quote = (game.get("moneylines") or {}).get(chosen) or {}
    ask = _num(quote.get("ask"))
    shares = _num(quote.get("shares"))
    token = str(quote.get("token_id") or "")
    quoted = _utc(quote.get("observed_at_utc"))
    if (ask is None or not 0 < ask < 1 or shares is None or shares <= 0
            or not token.isdigit() or quoted is None
            or not timedelta(0) <= now - quoted <= MAX_QUOTE_AGE):
        return None, "missing_or_stale_quote"
    assumed_price = ask + PRICE_PENALTY
    if assumed_price >= 1 or ask * shares < STAKE_CENTS / 100:
        return None, "price_or_depth"
    stressed_probability = max(0.0, probability - PROBABILITY_HAIRCUT)
    stressed_roi = stressed_probability / assumed_price - 1
    if stressed_roi < MIN_STRESSED_ROI:
        return None, "insufficient_stressed_ev"
    return {
        "event_slug": slug, "team": chosen, "token_id": token,
        "model_probability": round(probability, 6),
        "stressed_probability": round(stressed_probability, 6),
        "observed_ask": ask, "assumed_fill_price": round(assumed_price, 6),
        "stressed_roi": round(stressed_roi, 6),
        "stake_cents": STAKE_CENTS,
        "assumed_shares": round((STAKE_CENTS / 100) / assumed_price, 6),
        "quote_observed_at_utc": quoted.isoformat(),
        "tipoff_utc": tip.isoformat(),
    }, None


def on_live_refresh(snapshot, root=ROOT, *, at=None):
    """Atomically record hypothetical fills once per event on opted-in refreshes."""
    now = _utc(at or datetime.now(timezone.utc))
    out = {"paper_only": True, "live_orders_submitted": 0, "assumed_fills": [],
           "skipped": {}, "notice": "Assumed fills, not exchange executions or verified P&L."}
    if now is None or (snapshot.get("refresh_status") or {}).get("stale"):
        out["reason"] = "invalid_or_stale_snapshot"
        return out
    updated = _utc(snapshot.get("updated_at_utc"))
    if updated is None or not timedelta(0) <= now - updated <= MAX_QUOTE_AGE:
        out["reason"] = "invalid_or_stale_snapshot"
        return out
    candidates = []
    for game in snapshot.get("games") or []:
        candidate, reason = _candidate(game, now)
        if candidate is not None:
            candidates.append(candidate)
        else:
            out["skipped"][str(game.get("slug") or "unknown")] = reason
    path = Path(root) / DEFAULT_DB
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path, timeout=15) as db:
        db.execute("""CREATE TABLE IF NOT EXISTS paper_entries (
            event_slug TEXT PRIMARY KEY, captured_at_utc TEXT NOT NULL,
            utc_day TEXT NOT NULL, team TEXT NOT NULL, token_id TEXT NOT NULL,
            stake_cents INTEGER NOT NULL, observed_ask REAL NOT NULL,
            assumed_fill_price REAL NOT NULL, assumed_shares REAL NOT NULL,
            model_probability REAL NOT NULL, stressed_roi REAL NOT NULL,
            quote_observed_at_utc TEXT NOT NULL, fill_status TEXT NOT NULL
        )""")
        db.execute("BEGIN IMMEDIATE")
        utc_day = now.date().isoformat()
        spent = db.execute("SELECT COALESCE(SUM(stake_cents),0) FROM paper_entries WHERE utc_day=?",
                           (utc_day,)).fetchone()[0]
        for row in sorted(candidates, key=lambda r: r["stressed_roi"], reverse=True):
            slug = row["event_slug"]
            if db.execute("SELECT 1 FROM paper_entries WHERE event_slug=?", (slug,)).fetchone():
                out["skipped"][slug] = "already_paper_traded"
                continue
            if spent + STAKE_CENTS > DAILY_CAP_CENTS:
                out["skipped"][slug] = "daily_paper_cap"
                continue
            db.execute("""INSERT INTO paper_entries (
                event_slug,captured_at_utc,utc_day,team,token_id,stake_cents,
                observed_ask,assumed_fill_price,assumed_shares,model_probability,
                stressed_roi,quote_observed_at_utc,fill_status
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""", (
                slug, now.isoformat(), utc_day, row["team"], row["token_id"],
                STAKE_CENTS, row["observed_ask"], row["assumed_fill_price"],
                row["assumed_shares"], row["model_probability"],
                row["stressed_roi"], row["quote_observed_at_utc"],
                "ASSUMED_PAPER_FILL_NOT_EXECUTABLE",
            ))
            spent += STAKE_CENTS
            out["assumed_fills"].append(row)
    out["daily_paper_exposure_dollars"] = spent / 100
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True,
                        help="Saved JSON from GET /api/live/data")
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args(argv)
    snapshot = json.loads(args.snapshot.read_text(encoding="utf-8-sig"))
    print(json.dumps(on_live_refresh(snapshot, args.root), indent=2))


if __name__ == "__main__":
    main()
