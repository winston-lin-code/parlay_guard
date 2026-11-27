"""Replay the MLB/NHL entry rule against recorded Polymarket US books."""

import argparse
from bisect import bisect_right
from collections import Counter, defaultdict
import csv
from datetime import datetime, timedelta, timezone
from decimal import Decimal
import json
from pathlib import Path

from . import pro_inplay_model as model
from .inplay_scan import utc
from .pro_live_ev import (
    HALFTIME_WIN_FLOOR, MAX_BOOK_AGE, at_halftime, best_paper_quote, qualifies_screened_roi, verified_for_pricing,
)
from .us_wager import SLUG, WagerError

STAKE = Decimal("10")
SLIPPAGE = Decimal(".01")
BOOK_COLUMNS = (
    "market_slug", "timestamp", "local_timestamp", "event_type", "bid_prices", "bid_sizes",
    "offer_prices", "offer_sizes", "state",
)
STATE_FIELDS = ("score_away_home", "phase", "period", "outs", "bases")


def fixtures(path):
    output = {}
    with path.open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            event = row["event"]
            league = row["league"]
            if (event in output or league not in ("mlb", "nhl") or not SLUG.fullmatch(event)
                    or not event.startswith(league + "-")
                    or not row["official_game_id"] or row["home_long"] not in ("0", "1")
                    or row["winner"] not in ("away", "home") or not utc(row["settled_at"])):
                raise ValueError(f"Invalid or duplicate fixture {event}")
            if (not 0 < Decimal(row["tick"]) <= 1 or not 0 <= Decimal(row["fee_coefficient"]) <= 1
                    or not Decimal(row["minimum_quantity"]) > 0):
                raise ValueError(f"Invalid historical contract settings for {event}")
            output[event] = row
    return output


def verified_observations(path, games):
    states = defaultdict(dict)
    with path.open(encoding="utf-8-sig") as stream:
        for line in stream:
            if not line.strip():
                continue
            slate = json.loads(line)
            if slate.get("league") not in ("mlb", "nhl") or (
                "games" not in slate and slate.get("status") != "feed_unavailable"
            ):
                raise ValueError("Expected MLB/NHL score verifier JSONL")
            for row in slate.get("games", []):
                event = row["event"]
                if event not in games or str(row.get("official_game_id")) != games[event]["official_game_id"]:
                    continue
                official, market = row.get("official", {}), row.get("polymarket_us", {})
                official_at, market_at = utc(official.get("received_at")), utc(market.get("received_at"))
                if not official_at or not market_at:
                    continue
                later = max(official_at, market_at)
                if later not in states[event]:
                    states[event][later] = row
                elif states[event][later] != row:
                    raise ValueError(f"Conflicting score updates for {event} at {later.isoformat()}")
    return states


def confirmed_breaks(states, games, coverage):
    return confirmed_entries(states, games, coverage, "halftime")


def confirmed_entries(states, games, coverage, window):
    result = {}
    for event, game in games.items():
        league = game["league"]
        checks = sorted(states.get(event, {}).items())
        breaks = []
        for (first_at, first), (entry_at, second) in zip(checks, checks[1:]):
            if entry_at.year < 2025 or entry_at >= utc(game["settled_at"]):
                continue
            if not 0 < (entry_at - first_at).total_seconds() <= 10:
                continue
            if any(abs((utc(row["official"]["received_at"]) - utc(row["polymarket_us"]["received_at"])).total_seconds())
                   > MAX_BOOK_AGE for row in (first, second)):
                continue
            if not all(verified_for_pricing(row["official"], row["polymarket_us"]) for row in (first, second)):
                continue
            if any(first["official"].get(field) != second["official"].get(field) for field in STATE_FIELDS):
                continue
            if window == "halftime" and not at_halftime(league, second["official"]):
                continue
            if state := model.model_state(league, second["official"]):
                breaks.append((first_at, entry_at, second["official"], state))
        if breaks:
            coverage[league]["games_with_verified_break"] += 1
            result[event] = breaks
    return result


def snapshots(paths):
    for path in paths:
        if path.suffix.lower() == ".parquet":
            from pyarrow.parquet import ParquetFile

            for batch in ParquetFile(path).iter_batches(columns=BOOK_COLUMNS):
                for row in batch.to_pylist():
                    if row["event_type"] != "marketData":
                        continue
                    book = {"marketSlug": row["market_slug"], "transactTime": row["timestamp"],
                            "state": row["state"]}
                    for label, source in (("bids", "bid"), ("offers", "offer")):
                        prices, sizes = row[f"{source}_prices"], row[f"{source}_sizes"]
                        if prices is None or sizes is None:
                            break
                        if len(prices) != len(sizes):
                            raise ValueError(f"Mismatched depth columns in {path}")
                        book[label] = [{"px": {"value": price, "currency": "USD"}, "qty": size}
                                       for price, size in zip(prices, sizes)]
                    yield utc(row["local_timestamp"]), book
        else:
            with path.open(encoding="utf-8-sig") as stream:
                for line in stream:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    if row.get("kind") in ("book_snapshot", "book_update"):
                        seconds, ns = divmod(row["received_at_ns"], 1_000_000_000)
                        received = datetime.fromtimestamp(seconds, timezone.utc) + timedelta(microseconds=ns // 1000)
                        yield received, row["book"]


def available_books(breaks, records):
    checks = {event: [row[1] for row in rows] for event, rows in breaks.items()}
    selected = {}
    for received, book in records:
        slug = book.get("marketSlug") or ""
        event = slug.removeprefix("aec-")
        times = checks.get(event)
        observed = utc(book.get("transactTime"))
        if not times or not received or not observed or observed > received:
            continue
        index = bisect_right(times, received)
        if index >= len(times):
            continue
        key = event, index
        previous = selected.get(key)
        if previous is None or (observed, received) > previous[:2]:
            selected[key] = observed, received, book
        elif (observed, received) == previous[:2] and book != previous[2]:
            raise ValueError(f"Conflicting order-book snapshots for {slug} at {observed.isoformat()}")
    return selected


def simulate(games, breaks, books, coverage, *, stake=STAKE, high_ev=False):
    proposals = []
    for event, checks in breaks.items():
        game = games[event]
        league = game["league"]
        artifact = model.load(league)
        if not artifact:
            raise ValueError(f"{league} model unavailable")
        contract = {"slug": f"aec-{event}", "question": event, "orderPriceMinTickSize": game["tick"],
                    "minimumTradeQty": game["minimum_quantity"], "feeCoefficient": game["fee_coefficient"]}
        has_book = False
        for index, (first_at, entry_at, score, state) in enumerate(checks):
            archived = books.get((event, index))
            if not archived:
                continue
            observed, received, book = archived
            if (not first_at <= observed < entry_at or (entry_at - observed).total_seconds() > MAX_BOOK_AGE
                    or book.get("state") != "MARKET_STATE_OPEN"
                    or not all(isinstance(book.get(side), list) for side in ("bids", "offers"))):
                continue
            has_book = True
            home = model.forecast(state, artifact)
            probability = max(home, 1 - home)
            if probability <= HALFTIME_WIN_FLOOR:
                continue
            winner_side = "home" if home >= .5 else "away"
            long = (winner_side == "home") == (game["home_long"] == "1")
            try:
                quote = best_paper_quote(contract, {"long": long, "description": winner_side},
                                         probability, book, stake, SLIPPAGE, entry_at)
            except WagerError:
                continue
            if not quote:
                continue
            roi = quote["screened_ev"]
            if not qualifies_screened_roi(roi, high_ev=high_ev):
                continue
            won = winner_side == game["winner"]
            cost = Decimal(quote["total_estimate"])
            proposals.append({
                "league": league, "event": event, "official_game_id": game["official_game_id"],
                "entry_at": entry_at.isoformat(), "book_at": observed.isoformat(), "book_received_at": received.isoformat(),
                "settled_at": utc(game["settled_at"]).isoformat(),
                "score_away_home": "-".join(map(str, score["score_away_home"])),
                "phase": score["phase"], "period": score["period"], "checkpoint": int(at_halftime(league, score)),
                "selected_side": winner_side, "probability": probability, "screened_roi": roi,
                "ask_limit": quote["limit"], "quantity": quote["quantity"], "fee": quote["fee_estimate"],
                "cost": cost, "won": int(won), "pnl": int(won) * quote["quantity"] - cost,
            })
            break
        if has_book:
            coverage[league]["games_with_executable_book_window"] += 1
    unique = {}
    for row in sorted(proposals, key=lambda r: (r["entry_at"], r["event"])):
        unique.setdefault((row["league"], row["official_game_id"]), row)
    return list(unique.values())


def write_report(output, games, trades, coverage, *, window="halftime"):
    output.mkdir(parents=True, exist_ok=False)
    fields = ("league event official_game_id entry_at book_at book_received_at settled_at score_away_home phase period "
              "checkpoint selected_side probability screened_roi ask_limit quantity fee cost won pnl").split()
    with (output / "trades.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(trades)
    with (output / "summary.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["league", "settled_manifest_games", "verified_breaks" if window == "halftime" else "verified_games",
                         "book_windows", "trades",
                         "wins", "win_rate", "spend", "net_pnl", "net_roi", "max_drawdown"])
        for league in ("mlb", "nhl"):
            sample = [t for t in trades if t["league"] == league]
            spent, pnl, wins = (sum(t[field] for t in sample) for field in ("cost", "pnl", "won"))
            settled = defaultdict(Decimal)
            for trade in sample:
                settled[trade["settled_at"]] += trade["pnl"]
            running = peak = drawdown = Decimal(0)
            for timestamp in sorted(settled):
                running += settled[timestamp]
                peak = max(peak, running)
                drawdown = max(drawdown, peak - running)
            writer.writerow([league, sum(g["league"] == league for g in games.values()),
                             coverage[league]["games_with_verified_break"],
                             coverage[league]["games_with_executable_book_window"], len(sample), wins,
                             wins / len(sample) if sample else None, spent, pnl if sample else None,
                             pnl / spent if spent else None, drawdown if sample else None])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--states", required=True, type=Path)
    parser.add_argument("--settlements", required=True, type=Path)
    parser.add_argument("--books", nargs="*", type=Path, default=[])
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--window", choices=("halftime", "any", "compare"), default="halftime",
                        help="Replay halftime only, any supported live game state, or both")
    parser.add_argument("--stake", type=Decimal, default=STAKE, help="All-in cap per entry in USD (default: $10)")
    parser.add_argument("--high-ev", action="store_true",
                        help="Require at least 5%% screened ROI instead of the default 0%%–5%% band")
    args = parser.parse_args()
    if not args.stake.is_finite() or args.stake <= 0 or args.stake % Decimal(".01"):
        parser.error("--stake must be a positive USD amount with at most two decimal places")
    missing = [str(path) for path in (args.states, args.settlements, *args.books) if not path.is_file()]
    if missing:
        parser.exit(2, "Missing replay files: " + ", ".join(missing) + "\n"
                    "Supply recorded input files or run python -m src.pro_live_ev --watch --paper-any-high-ev "
                    "to collect new paper entries.\n")
    try:
        if args.output.exists():
            raise FileExistsError(f"Output folder already exists: {args.output}")
        games = fixtures(args.settlements)
        states = verified_observations(args.states, games)
        windows = ("any", "halftime") if args.window == "compare" else (args.window,)
        for window in windows:
            coverage = defaultdict(Counter)
            entries = confirmed_entries(states, games, coverage, window)
            books = available_books(entries, snapshots(args.books))
            trades = simulate(games, entries, books, coverage, stake=args.stake, high_ev=args.high_ev)
            output = args.output / window if len(windows) > 1 else args.output
            write_report(output, games, trades, coverage, window=window)
            print(f"{window}: {len(trades)} simulated trades written to {output}")
    except (OSError, ValueError, ArithmeticError, KeyError, ImportError) as exc:
        parser.exit(2, f"Backtest stopped: {exc}\n")


if __name__ == "__main__":
    main()
