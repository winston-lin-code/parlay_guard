"""Read-only dashboard and NBA-result settlement for paper moneyline entries.

All fills and payouts are SIMULATED. A verified NBA box score is only a proxy
for exchange contract resolution; it is NOT a verified Polymarket settlement.
No credentials, network calls, positions, or orders are touched.
"""
from __future__ import annotations

import json
import re
import sqlite3
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

from .paper_auto_execution import DEFAULT_DB, ROOT

STARTING_BANKROLL_CENTS = 10_000  # $100 fixed when paper journal first initializes
SLUG = re.compile(r"^nba-([a-z]{2,4})-([a-z]{2,4})-(\d{4}-\d{2}-\d{2})$")


def _money(cents):
    return round(int(cents) / 100, 2)


def _payout_cents(stake_cents, price):
    """Fixed-dollar binary share stake pays $1/share on simulated win."""
    if not 0 < float(price) < 1:
        raise ValueError("Invalid paper share price")
    payout = Decimal(int(stake_cents)) / Decimal(str(price))
    return int(payout.quantize(Decimal("1"), rounding=ROUND_HALF_UP))


def _ensure_tables(db):
    """Call only while holding a transaction on the paper journal."""
    db.execute("""CREATE TABLE IF NOT EXISTS paper_config (
        id INTEGER PRIMARY KEY CHECK (id=1), starting_bankroll_cents INTEGER NOT NULL
    )""")
    db.execute("INSERT OR IGNORE INTO paper_config VALUES (1, ?)",
               (STARTING_BANKROLL_CENTS,))
    db.execute("""CREATE TABLE IF NOT EXISTS paper_settlements (
        event_slug TEXT PRIMARY KEY,
        nba_game_id_verified TEXT NOT NULL,
        home_pts REAL NOT NULL,
        away_pts REAL NOT NULL,
        result TEXT NOT NULL CHECK(result IN ('WIN','LOSS')),
        settled_at_utc TEXT NOT NULL,
        FOREIGN KEY(event_slug) REFERENCES paper_entries(event_slug)
    )""")


def _available_cash_cents(db):
    """Unspent simulated cash; no mark-to-market or assumed pending payouts."""
    bankroll = int(db.execute(
        "SELECT starting_bankroll_cents FROM paper_config WHERE id=1"
    ).fetchone()[0])
    spent = int(db.execute("SELECT COALESCE(SUM(stake_cents),0) FROM paper_entries"
                           ).fetchone()[0])
    wins = db.execute("""SELECT e.stake_cents, e.assumed_fill_price
        FROM paper_entries e INNER JOIN paper_settlements s
          ON e.event_slug=s.event_slug WHERE s.result='WIN'""").fetchall()
    return bankroll - spent + sum(_payout_cents(s, p) for s, p in wins)


def settle_pending(root=ROOT, *, team_csv=None, at=None):
    """Resolve by strict paired completed NBA team box scores, never guess.

    Never treats a missing/ambiguous/invalid result as a loss. Does not claim
    that the exchange itself has settled a contract.
    """
    from .moneyline_paper_track import _load_results, _season

    path = Path(root) / DEFAULT_DB
    if not path.is_file():
        return {"newly_settled": 0, "pending": 0, "paper_only": True}
    now = at or datetime.now(timezone.utc)
    if now.tzinfo is None:
        raise ValueError("Timezone-aware settlement time required")
    now = now.astimezone(timezone.utc)
    caches = {}
    newly = 0
    with sqlite3.connect(path, timeout=15) as db:
        db.execute("PRAGMA foreign_keys=ON")
        db.execute("BEGIN IMMEDIATE")
        _ensure_tables(db)
        outstanding = db.execute("""SELECT e.event_slug, e.team, e.captured_at_utc
            FROM paper_entries e LEFT JOIN paper_settlements s
                ON s.event_slug=e.event_slug
            WHERE s.event_slug IS NULL ORDER BY e.captured_at_utc""").fetchall()
        for slug, team, captured in outstanding:
            match = SLUG.fullmatch(slug)
            if not match:
                continue
            away, home, day = match.groups()
            if team not in (home.upper(), away.upper()):
                continue
            # The date must be reached; results must be from paired finished logs.
            if day > now.date().isoformat():
                continue
            try:
                captured_dt = datetime.fromisoformat(captured.replace("Z", "+00:00"))
            except (ValueError, AttributeError):
                continue
            if captured_dt.tzinfo is None or captured_dt.astimezone(timezone.utc) >= now:
                continue
            source = Path(team_csv) if team_csv else Path(root) / "data" / (
                f"nba_{_season(day)}_latest_raw.csv")
            if source not in caches:
                caches[source] = _load_results(source)
            key = f"{day}_{away.upper()}_{home.upper()}"
            matches = caches[source].get(key, [])
            if len(matches) != 1:
                continue
            result = matches[0]
            outcome = "WIN" if result["winner"] == team else "LOSS"
            db.execute("""INSERT INTO paper_settlements (
                event_slug,nba_game_id_verified,home_pts,away_pts,result,settled_at_utc
            ) VALUES(?,?,?,?,?,?)""", (
                slug, result["nba_game_id"], result["home_pts"],
                result["away_pts"], outcome, now.isoformat(),
            ))
            newly += 1
        pending = db.execute("""SELECT COUNT(*) FROM paper_entries e
            LEFT JOIN paper_settlements s ON s.event_slug=e.event_slug
            WHERE s.event_slug IS NULL""").fetchone()[0]
    return {"newly_settled": newly, "pending": pending, "paper_only": True,
            "verification": "paired_completed_NBA_team_box_scores_not_exchange"}


def portfolio_report(root=ROOT, *, max_rows=200):
    """Read-only view of the persistent paper journal, no network or mutations."""
    path = Path(root) / DEFAULT_DB
    default_summary = {
        "starting_bankroll": _money(STARTING_BANKROLL_CENTS),
        "available_cash": _money(STARTING_BANKROLL_CENTS),
        "reserved_open_stakes": 0.0,
        "book_value_paper_equity": _money(STARTING_BANKROLL_CENTS),
        "realized_paper_pnl": 0.0, "settled_paper_roi": None,
        "max_realized_drawdown": 0.0, "max_realized_drawdown_pct": 0.0,
        "open_positions": 0, "settled_trades": 0, "wins": 0, "losses": 0,
    }
    base = {"paper_only": True, "live_orders_submitted": 0,
            "status": "NOT_STARTED", "summary": default_summary,
            "positions": [], "equity_curve": [],
            "execution_cost_scenarios": [],
            "notice": ("Hypothetical fills at observed ask +1 cent. NBA paired-game "
                       "verification is NOT Polymarket settlement; unresolved "
                       "positions are valued at paper cost, not market price.")}
    if not path.exists():
        return base
    with sqlite3.connect(path, timeout=15) as db:
        db.row_factory = sqlite3.Row
        exists = db.execute("""SELECT 1 FROM sqlite_master
            WHERE type='table' AND name='paper_entries'""").fetchone()
        if not exists:
            return base
        conf = db.execute("""SELECT 1 FROM sqlite_master WHERE type='table'
            AND name='paper_config'""").fetchone()
        bankroll_cents = int(db.execute(
            "SELECT starting_bankroll_cents FROM paper_config WHERE id=1"
        ).fetchone()[0]) if conf else STARTING_BANKROLL_CENTS
        settlement = db.execute("""SELECT 1 FROM sqlite_master
            WHERE type='table' AND name='paper_settlements'""").fetchone()
        if settlement:
            rows = [dict(r) for r in db.execute("""SELECT
                e.*, s.result, s.settled_at_utc, s.nba_game_id_verified,
                s.home_pts, s.away_pts
                FROM paper_entries e LEFT JOIN paper_settlements s
                ON s.event_slug=e.event_slug ORDER BY e.captured_at_utc,
                e.event_slug""").fetchall()]
        else:
            rows = [dict(r) for r in db.execute("""SELECT *,
                NULL as result, NULL as settled_at_utc,
                NULL as nba_game_id_verified, NULL as home_pts,
                NULL as away_pts
                FROM paper_entries ORDER BY captured_at_utc,event_slug""").fetchall()]

    if not rows:
        base["status"] = "WAITING_FOR_ENTRIES"
        base["summary"]["starting_bankroll"] = _money(bankroll_cents)
        base["summary"]["available_cash"] = _money(bankroll_cents)
        base["summary"]["book_value_paper_equity"] = _money(bankroll_cents)
        return base
    open_cost = sum(int(r["stake_cents"]) for r in rows if r["result"] is None)
    total_spent = sum(int(r["stake_cents"]) for r in rows)
    settled = [r for r in rows if r["result"] in ("WIN", "LOSS")]
    win_payouts = sum(_payout_cents(r["stake_cents"], r["assumed_fill_price"])
                      for r in settled if r["result"] == "WIN")
    settled_cost = sum(int(r["stake_cents"]) for r in settled)
    pnl_cents = win_payouts - settled_cost
    cash_cents = bankroll_cents - total_spent + win_payouts
    # Open positions are held at paper cost because market marks are unavailable.
    book_equity = cash_cents + open_cost
    peak = bankroll_cents
    realized_equity = bankroll_cents
    max_dd = 0
    max_dd_pct = 0.0
    curve = [{"at_utc": None, "realized_equity": _money(realized_equity)}]
    for r in sorted(settled, key=lambda row: (
            row["settled_at_utc"], row["event_slug"])):
        payout = (_payout_cents(r["stake_cents"], r["assumed_fill_price"])
                  if r["result"] == "WIN" else 0)
        realized_equity += payout - int(r["stake_cents"])
        peak = max(peak, realized_equity)
        dd = peak - realized_equity
        max_dd = max(max_dd, dd)
        max_dd_pct = max(max_dd_pct, dd / peak if peak > 0 else 0)
        curve.append({"at_utc": r["settled_at_utc"],
                      "realized_equity": _money(realized_equity)})
    cost_scenarios = []
    for premium in (0, 1, 2):
        total_payout = sum(
            _payout_cents(r["stake_cents"], r["observed_ask"] + premium / 100)
            for r in settled if r["result"] == "WIN"
            and r["observed_ask"] + premium / 100 < 1
        )
        valid = all(r["observed_ask"] + premium / 100 < 1 for r in settled)
        profit = total_payout - settled_cost if valid else None
        cost_scenarios.append({
            "ask_premium_cents": premium,
            "settled_trades": len(settled),
            "simulated_profit": _money(profit) if profit is not None else None,
            "simulated_roi": round(profit / settled_cost, 6)
                if profit is not None and settled_cost else None,
            "price_basis": "observed_ask_plus_hypothetical_premium",
        })
    positions = []
    for r in reversed(rows[-max_rows:]):
        payout = (_payout_cents(r["stake_cents"], r["assumed_fill_price"])
                  if r["result"] == "WIN" else 0)
        per_trade_pnl = (payout - int(r["stake_cents"])) if r["result"] else None
        positions.append({
            "event_slug": r["event_slug"], "team": r["team"],
            "captured_at_utc": r["captured_at_utc"],
            "quote_observed_at_utc": r["quote_observed_at_utc"],
            "model_probability": r["model_probability"],
            "observed_ask": r["observed_ask"],
            "assumed_fill_price": r["assumed_fill_price"],
            "stressed_roi": r["stressed_roi"],
            "stake": _money(r["stake_cents"]), "assumed_shares": r["assumed_shares"],
            "status": r["result"] or "PENDING_NBA_VERIFICATION",
            "simulated_pnl": _money(per_trade_pnl) if per_trade_pnl is not None else None,
            "nba_game_id_verified": r["nba_game_id_verified"],
            "nba_final_score": [r["away_pts"], r["home_pts"]]
                if r["result"] else None,
            "settled_at_utc": r["settled_at_utc"],
        })
    base["status"] = "TRACKING"
    base["summary"] = {
        "starting_bankroll": _money(bankroll_cents),
        "available_cash": _money(cash_cents),
        "reserved_open_stakes": _money(open_cost),
        "book_value_paper_equity": _money(book_equity),
        "realized_paper_pnl": _money(pnl_cents),
        "settled_paper_roi": round(pnl_cents / settled_cost, 6)
            if settled_cost else None,
        "max_realized_drawdown": _money(max_dd),
        "max_realized_drawdown_pct": round(max_dd_pct, 6),
        "open_positions": sum(1 for r in rows if r["result"] is None),
        "settled_trades": len(settled),
        "wins": sum(1 for r in settled if r["result"] == "WIN"),
        "losses": sum(1 for r in settled if r["result"] == "LOSS"),
    }
    base["positions"] = positions
    base["equity_curve"] = curve
    base["execution_cost_scenarios"] = cost_scenarios
    return base
