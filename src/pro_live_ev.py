"""Paper-only MLB and NHL in-game moneyline forecasts and Polymarket US EV."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from decimal import Decimal, ROUND_CEILING
import json
from pathlib import Path
import re
import sqlite3
from time import monotonic, sleep

from . import pro_inplay_model as model
from .pro_live_scores import (
    DATA_ERRORS, LEAGUES, MLB, NHL, SPORTS, agreement, assess, league_json, league_listings, market_state, mlb_state,
    nhl_state, now, received_state,
)
from .inplay_scan import utc
from .predict_today import ET
from .us_wager import CENT, WagerError, decimal, public, quote_order

MAX_BOOK_AGE = 5
MIN_SCREENED_ROI = .05
IDLE_HEARTBEAT_SECONDS = 60
HALFTIME_CHECKPOINTS = {"mlb": ("end", 5), "nhl": ("intermission", 2)}
HALFTIME_WIN_FLOOR = .60
DEFAULT_PAPER_DB = Path("data/paper_trades/pro_halftime.sqlite3")
OUTCOME_SCOPE = {
    "mlb": r"extra innings (?:are )?included",
    "nhl": r"overtime and (?:any )?shootout (?:are )?included",
}


def contract_for(league, event):
    slug = f"aec-{event['slug']}"
    matches = [m for m in event.get("markets") or [] if m.get("slug") == slug
               and m.get("active") is True and m.get("closed") is False
               and m.get("marketType") == "moneyline"
               and m.get("sportsMarketType") == f"{SPORTS[league]}_team_full_game_winner"
               and re.search(OUTCOME_SCOPE[league], m.get("description") or "", flags=re.I)]
    if len(matches) != 1:
        return None
    match = matches[0]
    sides = {(side.get("team") or {}).get("ordering"): side for side in match.get("marketSides") or []}
    if set(sides) != {"home", "away"} or {side.get("long") for side in sides.values()} != {True, False}:
        return None
    return match, sides


def current_official(league, game_id):
    if league == "mlb":
        state = mlb_state(league_json(f"{MLB}/v1.1/game/{game_id}/feed/live"))
    else:
        state = nhl_state(league_json(f"{NHL}/gamecenter/{game_id}/landing"))
    return received_state(state, now())


def current_market(league, slug):
    event = public(f"/v1/events/slug/{slug}", fresh_score=True)["event"]
    if event.get("slug") != slug:
        raise WagerError("Polymarket US returned a different event.")
    return event, received_state(market_state(league, event), now())


def verified_for_pricing(official, market):
    if agreement(official, market) != "score_and_phase_agree":
        return False
    if official["phase"] in ("top", "bottom"):
        return all(state.get("bases") is not None and type(state.get("outs")) is int for state in (official, market))
    return True


def best_paper_quote(contract, side, p, book, stake, slippage, reference):
    pick = {"side": (side.get("team") or {}).get("name") or side.get("description"), "probability": p}
    minimum = max(1, int(decimal(contract["minimumTradeQty"]).to_integral_value(rounding=ROUND_CEILING)))
    low, high = minimum, min(int(stake / CENT), 10_000)
    largest = None
    while low <= high:
        quantity = (low + high) // 2
        try:
            _, quote = quote_order(pick, contract, side["long"], quantity, book, reference, slippage=slippage)
        except WagerError:
            high = quantity - 1
            continue
        if decimal(quote["total_estimate"]) <= stake:
            largest = quote
            low = quantity + 1
        else:
            high = quantity - 1
    if not largest:
        return None
    quote = best = largest
    # Smaller quantities can avoid paying through the top of a thin book.
    while quote["screened_ev"] < MIN_SCREENED_ROI and quote["quantity"] > minimum:
        _, quote = quote_order(pick, contract, side["long"], quote["quantity"] - 1, book, reference,
                               slippage=slippage)
        if quote["screened_ev"] > best["screened_ev"]:
            best = quote
    cost = float(best["total_estimate"])
    return {**best, "probability": p, "slippage_per_contract": float(slippage),
            "estimated_net_roi": p * best["quantity"] / cost - 1,
            "meets_paper_roi_floor": best["screened_ev"] >= MIN_SCREENED_ROI}


def evaluate(league, event, verified, artifact, stake, slippage):
    state = model.model_state(league, verified.get("official", {}))
    row = {**verified, "paper": True}
    if not state:
        return {**row, "pricing_status": "unsupported_game_state"}
    home = model.forecast(state, artifact)
    row["win_probability"] = {"home": home, "away": 1 - home}
    if not verified_for_pricing(verified["official"], verified["polymarket_us"]):
        return {**row, "pricing_status": "score_or_phase_unverified"}
    matched = contract_for(league, event)
    if not matched:
        return {**row, "pricing_status": "unverified_moneyline_terms"}
    contract, _ = matched
    try:
        book = public(f"/v1/markets/{contract['slug']}/book", fresh_book=True)["marketData"]
        with ThreadPoolExecutor(max_workers=2) as pool:
            official = pool.submit(current_official, league, verified["official_game_id"])
            latest = pool.submit(current_market, league, event["slug"])
            official, (latest, current) = official.result(), latest.result()
    except DATA_ERRORS as exc:
        return {**row, "pricing_status": "quote_or_recheck_unavailable", "error": str(exc)}
    state = model.model_state(league, official)
    new_contract = contract_for(league, latest)
    if (not state or not new_contract or not verified_for_pricing(official, current)
            or any(current.get(k) != verified["polymarket_us"].get(k)
                   for k in ("score_away_home", "phase", "period", "outs", "bases"))):
        return {**row, "pricing_status": "game_state_changed_during_quote"}
    observed, reference = utc(book.get("transactTime")), now()
    score_received = utc(verified["polymarket_us"].get("received_at"))
    official_received = utc(verified["official"].get("received_at"))
    if (not observed or not 0 <= (reference - observed).total_seconds() <= MAX_BOOK_AGE
            or not score_received or not official_received or observed < max(score_received, official_received)):
        return {**row, "pricing_status": "no_fresh_post_score_book"}
    home = model.forecast(state, artifact)
    row["win_probability"] = {"home": home, "away": 1 - home}
    row["official"] = official
    row["polymarket_us"] = current
    row["quotes"] = {}
    contract, sides = new_contract
    for label, p in row["win_probability"].items():
        side = sides[label]
        if side.get("tradable") is True:
            try:
                row["quotes"][label] = best_paper_quote(contract, side, p, book, stake, slippage, reference)
            except WagerError:
                row["quotes"][label] = None
    row["pricing_status"] = "priced" if any(row["quotes"].values()) else "no_executable_quote_at_stake"
    return row


def at_halftime(league, official):
    return (official.get("phase"), official.get("period")) == HALFTIME_CHECKPOINTS[league]


def log_paper_entry(path, game, stake):
    if game["pricing_status"] != "priced" or not at_halftime(game["league"], game["official"]):
        return {"status": "no_verified_halftime_quote"}
    qualifies = [(side, quote) for side, quote in game["quotes"].items()
                 if quote and quote["probability"] > HALFTIME_WIN_FLOOR
                 and 0 < quote["screened_ev"] < MIN_SCREENED_ROI and decimal(quote["total_estimate"]) <= stake]
    if not qualifies:
        return {"status": "outside_probability_or_roi_band"}
    side, quote = max(qualifies, key=lambda pair: pair[1]["screened_ev"])
    quoted, recorded = utc(quote["quote_utc"]), now()
    if not quoted or not 0 <= (recorded - quoted).total_seconds() <= MAX_BOOK_AGE:
        return {"status": "quote_expired_before_paper_entry"}
    entry = {"mode": "paper", "event": game["event"], "league": game["league"],
             "official_game_id": game["official_game_id"], "outcome": side,
             "checkpoint": game["official"]["phase"], "period": game["official"]["period"],
             "score_away_home": game["official"]["score_away_home"], "recorded_at": recorded.isoformat(),
             "stake_cap": str(stake), "quote": quote}
    path.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(path, timeout=5)) as db, db:
        db.execute("""CREATE TABLE IF NOT EXISTS pro_halftime_entries (
            league TEXT NOT NULL, official_game_id TEXT NOT NULL, event TEXT NOT NULL UNIQUE,
            recorded_at TEXT NOT NULL, detail TEXT NOT NULL, PRIMARY KEY (league, official_game_id)
        )""")
        inserted = db.execute("INSERT OR IGNORE INTO pro_halftime_entries VALUES (?, ?, ?, ?, ?)",
                              (game["league"], str(game["official_game_id"]), game["event"],
                               entry["recorded_at"], json.dumps(entry))).rowcount
    return {"status": "paper_entry_logged", "entry": entry} if inserted else {"status": "already_logged"}


def scan(league, stake, slippage, *, paper_db=None):
    artifact = model.load(league)
    if not artifact:
        return {"league": league, "status": "model_unavailable"}
    try:
        listings = list(league_listings(league))
        events = {slug: (event, received) for slug, event, received in listings if event.get("live") is True}
        if not events:
            return idle_status(league, listings)
        observed = assess(league, events)
    except DATA_ERRORS as exc:
        return {"league": league, "status": "feed_unavailable", "error": str(exc)}
    results = []
    for verified in observed["games"]:
        event = events[verified["event"]][0]
        if paper_db is not None and not at_halftime(league, verified.get("official", {})):
            results.append({**verified, "pricing_status": "outside_halftime_checkpoint"})
            continue
        try:
            row = evaluate(league, event, verified, artifact, stake, slippage)
        except DATA_ERRORS as exc:
            row = {**verified, "paper": True, "pricing_status": "quote_or_recheck_unavailable", "error": str(exc)}
        if paper_db is not None:
            row["halftime_paper"] = log_paper_entry(paper_db, row, stake)
        results.append(row)
    return {"league": league, "listings_checked": len(events), "games": results}


def idle_status(league, listings):
    scheduled = []
    for slug, event, received in listings:
        if event.get("ended") is True or event.get("closed") is True or event.get("active") is not True:
            continue
        start = utc(event.get("startTime"))
        if start and start >= received:
            scheduled.append((start, slug, event.get("title")))
    earliest = min((start for start, _, _ in scheduled), default=None)
    return {"league": league, "status": "no_live_listings", "listings_checked": 0,
            "upcoming_listings": len(scheduled), "next_start_et": earliest.astimezone(ET).isoformat() if earliest else None,
            "next_games": [{"event": slug, "title": title} for start, slug, title in sorted(scheduled, key=lambda r: r[1])
                           if start == earliest], "games": []}


def should_report(result, previous, elapsed):
    return result.get("status") != "no_live_listings" or (
        result != previous or elapsed >= IDLE_HEARTBEAT_SECONDS
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--leagues", nargs="+", choices=LEAGUES, default=LEAGUES)
    parser.add_argument("--stake", help="Maximum paper spend including costs, default $10 for halftime or $2 otherwise")
    parser.add_argument("--slippage", default=".01", help="Adverse price allowance per contract in dollars")
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--interval", type=int, default=5)
    parser.add_argument("--paper-halftime", action="store_true",
                        help="Log halftime paper entries with win probability above 60%% and positive ROI below 5%%")
    parser.add_argument("--paper-db", type=Path, help="SQLite ledger for --paper-halftime")
    args = parser.parse_args()
    if args.paper_db and not args.paper_halftime:
        parser.error("--paper-db requires --paper-halftime")
    paper_db = (args.paper_db or DEFAULT_PAPER_DB) if args.paper_halftime else None
    try:
        stake = decimal(args.stake or ("10.00" if args.paper_halftime else "2.00"))
        slippage = decimal(args.slippage)
        if stake < CENT or not 0 <= slippage < 1 or args.interval < 3:
            raise ValueError("Use a stake of at least one cent, slippage below one dollar, and interval of at least 3 seconds.")
    except ValueError as exc:
        parser.error(str(exc))
    leagues = list(dict.fromkeys(args.leagues))
    last_output = {}
    try:
        with ThreadPoolExecutor(max_workers=len(leagues)) as pool:
            while True:
                started = monotonic()
                for result in pool.map(lambda league: scan(league, stake, slippage, paper_db=paper_db), leagues):
                    league = result["league"]
                    previous, emitted = last_output.get(league, (None, started))
                    if should_report(result, previous, started - emitted):
                        print(json.dumps(result), flush=True)
                        last_output[league] = result, started
                if not args.watch:
                    break
                sleep(max(0, args.interval - (monotonic() - started)))
    except KeyboardInterrupt:
        pass
    except sqlite3.Error as exc:
        parser.exit(1, f"Paper ledger unavailable: {exc}\n")


if __name__ == "__main__":
    main()
