"""Paper-only MLB and NHL in-game moneyline forecasts and Polymarket US EV."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from decimal import Decimal, ROUND_CEILING
import json
import re
from time import monotonic, sleep

from . import pro_inplay_model as model
from .pro_live_scores import (
    DATA_ERRORS, LEAGUES, MLB, NHL, SPORTS, agreement, assess, league_json, market_events, market_state, mlb_state,
    nhl_state, now, received_state,
)
from .inplay_scan import utc
from .us_wager import CENT, WagerError, decimal, public, quote_order

MAX_BOOK_AGE = 5
MIN_SCREENED_ROI = .05
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


def scan(league, stake, slippage):
    artifact = model.load(league)
    if not artifact:
        return {"league": league, "status": "model_unavailable"}
    try:
        events = market_events(league)
        observed = assess(league, events)
    except DATA_ERRORS as exc:
        return {"league": league, "status": "feed_unavailable", "error": str(exc)}
    results = []
    for verified in observed["games"]:
        event = events[verified["event"]][0]
        try:
            results.append(evaluate(league, event, verified, artifact, stake, slippage))
        except DATA_ERRORS as exc:
            results.append({**verified, "paper": True, "pricing_status": "quote_or_recheck_unavailable",
                            "error": str(exc)})
    return {"league": league, "listings_checked": len(events), "games": results}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--leagues", nargs="+", choices=LEAGUES, default=LEAGUES)
    parser.add_argument("--stake", default="2.00", help="Maximum paper spend including fees and slippage")
    parser.add_argument("--slippage", default=".01", help="Adverse price allowance per contract in dollars")
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--interval", type=int, default=5)
    args = parser.parse_args()
    try:
        stake, slippage = decimal(args.stake), decimal(args.slippage)
        if stake < CENT or not 0 <= slippage < 1 or args.interval < 3:
            raise ValueError("Use a stake of at least one cent, slippage below one dollar, and interval of at least 3 seconds.")
    except ValueError as exc:
        parser.error(str(exc))
    leagues = list(dict.fromkeys(args.leagues))
    try:
        with ThreadPoolExecutor(max_workers=len(leagues)) as pool:
            while True:
                started = monotonic()
                for result in pool.map(lambda league: scan(league, stake, slippage), leagues):
                    print(json.dumps(result), flush=True)
                if not args.watch:
                    break
                sleep(max(0, args.interval - (monotonic() - started)))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
