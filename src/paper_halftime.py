"""Run a local paper account for Parlay Guard's halftime moneyline rule."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing, contextmanager
import csv
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
import sqlite3
import time

import requests

from .halftime_paper import (
    LEAGUES, MIN_PROBABILITY, MIN_SCREENED_EV, PAPER_STAKE, PROBABILITY_HAIRCUT, screen_halftime,
)
from .inplay_scan import utc
from .us_wager import GATEWAY, ONE, SLUG, WagerError, decimal

DEFAULT_DB = Path("data/paper_trades/halftime_account.sqlite3")
START_CENTS = 10_000
STAKE_CENTS = int(PAPER_STAKE * 100)
FRESH_SECONDS = 5
POLL_SECONDS = 2
SETTLEMENT_INTERVAL = 60


@contextmanager
def ledger(path):
    with closing(sqlite3.connect(path, timeout=10)) as db, db:
        db.row_factory = sqlite3.Row
        yield db


def init_account(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with ledger(path) as db:
        db.execute("""CREATE TABLE IF NOT EXISTS halftime_bets (
            league TEXT NOT NULL, fixture_id TEXT NOT NULL, event_slug TEXT NOT NULL,
            market_slug TEXT NOT NULL UNIQUE, is_long INTEGER NOT NULL,
            matchup TEXT NOT NULL, side TEXT NOT NULL, halftime_score TEXT NOT NULL,
            quote_utc TEXT NOT NULL, entered_utc TEXT NOT NULL, link TEXT NOT NULL,
            probability REAL NOT NULL, screened_ev REAL NOT NULL,
            ask TEXT NOT NULL, fee_inclusive_cost TEXT NOT NULL, shares TEXT NOT NULL,
            stake_cents INTEGER NOT NULL, settled_utc TEXT, payout_per_share TEXT, payout_cents INTEGER,
            PRIMARY KEY (league, fixture_id)
        )""")


def is_fresh(timestamp, now):
    observed = utc(timestamp)
    return observed is not None and -5 <= (now - observed).total_seconds() <= FRESH_SECONDS


def account_totals(db):
    count, stake, open_count, open_stake, payout = db.execute("""SELECT COUNT(*),
        COALESCE(SUM(stake_cents), 0),
        COUNT(*) FILTER (WHERE settled_utc IS NULL),
        COALESCE(SUM(stake_cents) FILTER (WHERE settled_utc IS NULL), 0),
        COALESCE(SUM(payout_cents), 0)
        FROM halftime_bets""").fetchone()
    return {
        "bets": count, "pending": open_count, "available_cents": START_CENTS - stake + payout,
        "pending_stake_cents": open_stake, "realized_cents": payout - (stake - open_stake),
    }


def recorded_games(path):
    with ledger(path) as db:
        return {(row[0], row[1]) for row in db.execute("SELECT league, fixture_id FROM halftime_bets")}


def record_scan(path, scan, *, now=None):
    """Lock available cash and deduplicate before committing any simulated entries."""
    written = []
    with ledger(path) as db:
        db.execute("BEGIN IMMEDIATE")
        cash = account_totals(db)["available_cents"]
        for pick in scan["picks"]:
            at = now if now is not None else datetime.now(timezone.utc)
            if cash < STAKE_CENTS or not is_fresh(scan.get("updated_utc"), at):
                break
            market = pick.get("market_slug") or ""
            game = pick.get("event") or ""
            fixture_id = pick.get("fixture_id") or ""
            if (pick.get("phase") != "halftime" or pick.get("league") not in LEAGUES
                    or not str(fixture_id).isdigit()
                    or not SLUG.fullmatch(game) or not SLUG.fullmatch(market) or market != f"aec-{game}"
                    or type(pick.get("long")) is not bool or not is_fresh(pick.get("quote_utc"), at)):
                continue
            try:
                probability, ask, cost = (decimal(pick.get(key)) for key in ("probability", "ask", "cost"))
            except WagerError:
                continue
            if not decimal(MIN_PROBABILITY) <= probability <= ONE or not 0 < ask <= cost < ONE:
                continue
            ev = (probability - decimal(PROBABILITY_HAIRCUT)) / cost - 1
            if ev < decimal(MIN_SCREENED_EV):
                continue
            result = db.execute("""INSERT INTO halftime_bets (
                league, fixture_id, event_slug, market_slug, is_long, matchup, side, halftime_score,
                quote_utc, entered_utc, link, probability, screened_ev, ask, fee_inclusive_cost,
                shares, stake_cents
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?) ON CONFLICT DO NOTHING""", (
                pick["league"], fixture_id, game, market, int(pick["long"]), pick["title"], pick["side"],
                pick["score_text"], pick["quote_utc"], at.isoformat(), pick["url"], float(probability),
                float(ev), str(ask), str(cost), str(PAPER_STAKE / cost), STAKE_CENTS,
            ))
            if result.rowcount:
                cash -= STAKE_CENTS
                written.append(pick)
    return written


def settlement_price(market):
    if not SLUG.fullmatch(market):
        return None
    response = requests.get(f"{GATEWAY}/v1/markets/{market}/settlement", timeout=(3, 5))
    if response.status_code == 404:
        return None
    response.raise_for_status()
    data = response.json()
    if data.get("slug") != market:
        return None
    price = decimal(data.get("settlement"))
    return price if 0 <= price <= ONE else None


def settle_pending(path):
    with ledger(path) as db:
        pending = db.execute("""SELECT market_slug, is_long, shares, matchup, side, stake_cents
            FROM halftime_bets WHERE settled_utc IS NULL ORDER BY entered_utc""").fetchall()

    def fetch(row):
        try:
            return settlement_price(row["market_slug"]), False
        except (requests.RequestException, ValueError, TypeError, AttributeError):
            return None, True

    changed = []
    failures = 0
    if not pending:
        return changed, failures
    with ThreadPoolExecutor(max_workers=4) as pool:
        outcomes = list(pool.map(fetch, pending))
    with ledger(path) as db:
        for row, (long_price, failed) in zip(pending, outcomes):
            failures += failed
            if long_price is None:
                continue
            side_price = long_price if row["is_long"] else ONE - long_price
            cents = int((Decimal(row["shares"]) * side_price * 100).quantize(Decimal(1), rounding=ROUND_HALF_UP))
            result = db.execute("""UPDATE halftime_bets SET settled_utc = ?, payout_per_share = ?, payout_cents = ?
                WHERE market_slug = ? AND settled_utc IS NULL""",
                (datetime.now(timezone.utc).isoformat(), str(side_price), cents, row["market_slug"]))
            if result.rowcount:
                changed.append((row, cents - row["stake_cents"]))
    return changed, failures


def print_status(path):
    with ledger(path) as db:
        stats = account_totals(db)
    print(f"Paper cash ${stats['available_cents']/100:.2f} | Reserved ${stats['pending_stake_cents']/100:.2f} | "
          f"Realized P&L {stats['realized_cents']/100:+.2f} USD | "
          f"{stats['bets']} bets, {stats['pending']} pending", flush=True)


def export_csv(path, destination):
    if destination.resolve() == path.resolve():
        raise ValueError("Choose a CSV path different from the account database")
    with ledger(path) as db:
        cursor = db.execute("SELECT * FROM halftime_bets ORDER BY entered_utc, league, fixture_id")
        rows = cursor.fetchall()
        columns = [field[0] for field in cursor.description]
    with destination.open("x", newline="", encoding="utf-8") as output:
        csv.writer(output).writerow(columns)
        csv.writer(output).writerows(rows)


def report_settlements(result):
    changed, failures = result
    for row, net in changed:
        print(f"Settled | {row['matchup']} | {row['side']} | P&L {net/100:+.2f} USD", flush=True)
    if failures:
        print(f"Settlement feed unavailable for {failures} bets. They remain pending.", flush=True)
    return bool(changed)


def run(path, *, leagues, once):
    print(f"Halftime paper account | {', '.join(leagues)} | ${PAPER_STAKE} per game | {POLL_SECONDS}s polling", flush=True)
    print_status(path)
    next_settlement = 0
    settlement = None
    last_status = None
    with ThreadPoolExecutor(max_workers=1) as executor:
        while True:
            started = time.monotonic()
            if settlement is not None and settlement.done():
                if report_settlements(settlement.result()):
                    print_status(path)
                settlement = None
            if settlement is None and started >= next_settlement:
                settlement = executor.submit(settle_pending, path)
                next_settlement = started + SETTLEMENT_INTERVAL
            scan = screen_halftime(leagues=leagues, include_live_inventory=False, recorded=recorded_games(path))
            bets = record_scan(path, scan)
            for pick in bets:
                print(f"Paper bet | {pick['league']} | {pick['title']} | {pick['side']} "
                      f"| Cost {pick['cost']:.3f} per share | Model {pick['probability']:.1%} "
                      f"| Screened EV {pick['screened_ev']:+.1%}\n  {pick['url']}", flush=True)
            if bets:
                print_status(path)
            status = scan["games_at_halftime"], len(scan["picks"]), tuple(scan["warnings"])
            if status != last_status:
                print(f"{status[0]} at halftime | {status[1]} qualifying new quotes", flush=True)
                for warning in status[2]:
                    print(warning, flush=True)
                last_status = status
            if once:
                if settlement is not None and report_settlements(settlement.result()):
                    print_status(path)
                return
            time.sleep(max(0, POLL_SECONDS - (time.monotonic() - started)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB, help="Local SQLite account path")
    parser.add_argument("--league", choices=(*LEAGUES, "all"), default="all")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--once", action="store_true", help="Scan once and exit")
    mode.add_argument("--status", action="store_true", help="View the saved paper balance without scanning")
    mode.add_argument("--settle-only", action="store_true", help="Check payouts without scanning for entries")
    mode.add_argument("--export-csv", type=Path, help="Write the account ledger to a new CSV file")
    args = parser.parse_args()
    init_account(args.db)
    try:
        if args.settle_only:
            report_settlements(settle_pending(args.db))
            print_status(args.db)
        elif args.status:
            print_status(args.db)
        elif args.export_csv:
            export_csv(args.db, args.export_csv)
            print(args.export_csv)
        else:
            run(args.db, leagues=LEAGUES if args.league == "all" else (args.league,), once=args.once)
    except KeyboardInterrupt:
        print("\nPaper account saved.", flush=True)
    except (FileExistsError, ValueError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
