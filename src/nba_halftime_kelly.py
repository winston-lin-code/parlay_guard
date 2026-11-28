"""Submit quarter-Kelly NBA halftime moneylines on Polymarket US."""

import argparse
from decimal import Decimal, ROUND_CEILING, ROUND_DOWN
from pathlib import Path
import sqlite3
from threading import Event, Lock, Thread
from time import monotonic

from . import nba_moneyline_shadow as nba
from .inplay_scan import PROBABILITY_HAIRCUT
from .pregame_capture import signing_credentials
from .pro_halftime_orders import AutomaticOrders, LIVE_STAKE
from .pro_live_ev import MAX_BOOK_AGE
from .run_research import ROOT, live_client, live_credentials
from .tennis_wager import BookStream
from .us_wager import CENT, ONE, WagerError, decimal, quote_order
from .wcbb_anytime import MIN_EV, SCORE_DELAY, SLIPPAGE, display_team, game_state, now

LIVE_DB = ROOT / "data" / "live_orders" / "nba_halftime_kelly.sqlite3"
FRACTION = Decimal(".25")
MAX_BANKROLL_FRACTION = Decimal(".02")


def buying_power(client):
    try:
        with client() as exchange:
            response = exchange.account.balances()
        # The account endpoint lists currencies. The SDK also documents a single balance.
        balances = [item for item in response.get("balances", [response]) if item.get("currency") == "USD"]
        if len(balances) != 1:
            raise ValueError("Expected one USD balance")
        balance = decimal(balances[0]["buyingPower"])
    except Exception as exc:
        raise WagerError(f"Polymarket US USD buying power unavailable ({type(exc).__name__})") from None
    return max(Decimal(0), balance)


def stake_cap(bankroll):
    return min(LIVE_STAKE, bankroll * MAX_BANKROLL_FRACTION).quantize(CENT, rounding=ROUND_DOWN)


def quarter_kelly_quote(game, contract, market, bankroll, *, explain=False):
    maximum = stake_cap(bankroll)
    if maximum == 0:
        reason = ("No positive USD buying power reported by Polymarket US" if bankroll == 0
                  else "USD buying power is too small for a one-cent per-game cap")
        return None, reason
    pick = {"side": display_team(game["teams"][market["selected"]]), "probability": market["probability"]}
    adjusted = max(Decimal(0), market["probability"] - decimal(PROBABILITY_HAIRCUT))
    minimum = max(1, int(decimal(contract["minimumTradeQty"]).to_integral_value(rounding=ROUND_CEILING)))
    cheapest, best_ev, closest = None, None, None
    book_error = "Exchange minimum exceeds the maximum stake at the current ask"
    for quantity in range(int(LIVE_STAKE / market["ask"]), minimum - 1, -1):
        try:
            payload, quote = quote_order(pick, contract, market["side"]["long"], quantity, market["book"],
                                         market["received"], slippage=SLIPPAGE)
        except WagerError as exc:
            book_error = str(exc)
            continue
        total = decimal(quote["total_estimate"])
        cheapest = total if cheapest is None else min(cheapest, total)
        if total > maximum:
            continue
        ev = adjusted * quantity / total - ONE
        best_ev = ev if best_ev is None else max(best_ev, ev)
        if adjusted * quantity < (ONE + MIN_EV) * total:
            continue
        unit_cost = total / quantity
        allocation = bankroll * FRACTION * (adjusted - unit_cost) / (ONE - unit_cost)
        cap = min(maximum, allocation).quantize(CENT, rounding=ROUND_DOWN)
        if total <= cap:
            return (payload, quote, cap), None
        miss = (total - cap, total, cap)
        if closest is None or miss < closest:
            closest = miss
    if cheapest is None:
        return None, book_error
    if best_ev is None:
        reason = "Minimum executable order exceeds the 2%/$5 cap"
        detail = f"${cheapest:.2f} all-in vs ${maximum:.2f} cap"
    elif closest is not None:
        _, total, cap = closest
        reason = "Quarter-Kelly allocation too small for an EV-qualified order"
        detail = f"${total:.2f} all-in vs ${cap:.2f} Kelly budget"
    else:
        reason = "Best affordable screened EV below 5%"
        detail = f"{best_ev:+.2%}, {market['probability']:.1%} win probability, ${market['ask']:.3f} best ask"
    return None, f"{reason} ({detail})" if explain else reason


def verify_checkpoint(game):
    date = game["tip"].astimezone(nba.halftime.ET).date()
    for event in nba.retry_espn(nba.scoreboard, date):
        if str(event.get("id")) != game["id"]:
            continue
        window = nba.halftime.game_state("NBA", event)
        current = game_state(event, league="NBA")
        return (window is not None and current is not None and window["checkpoint"] == game["checkpoint"]
                and current["scores"] == game["scores"] and current["period"] == game["period"]
                and -2 <= game["clock"] - current["clock"] <= SCORE_DELAY)
    return False


def scan(orders, client, stream, *, explain=False):
    # Read buying power and submit one game at a time so concurrent candidates
    # use the updated account balance after any preceding fill.
    execution = Lock()

    def check(game, match, books):
        if not game["inside_window"]:
            return 0, "Outside the halftime window"
        if match is None:
            return 0, "No uniquely matched Polymarket US moneyline"
        event, contract, assignments = match
        previous = orders.attempts("NBA", [game["id"]]).get(game["id"])
        if previous is not None and not orders.can_retry(previous):
            return 0, None
        try:
            market, reason = nba.market_at_score(game, contract, assignments, books)
            if market is None:
                return 0, reason
            with execution:
                bankroll = buying_power(client)
                sized, reason = quarter_kelly_quote(game, contract, market, bankroll, explain=explain)
                if sized is None:
                    return 0, reason
                if not verify_checkpoint(game):
                    return 0, "ESPN score or halftime checkpoint changed before submission"
                payload, quote, cap = sized
                recorded = now()
                entry = {"event": event["slug"], "league": "NBA", "official_game_id": game["id"],
                         "mode": "live", "strategy": "quarter_kelly", "checkpoint": game["checkpoint"],
                         "clock": game["clock_text"], "score_away_home": [game["scores"]["away"], game["scores"]["home"]],
                         "outcome": quote["side"], "probability": float(market["probability"]),
                         "score_utc": market["score_at"].isoformat(), "book_source": market["source"],
                         "recorded_at": recorded.isoformat(), "stake_cap": str(cap), "quote": quote}
                result = orders.place(entry, payload, retry_for=previous["order_id"] if previous else None)
        except nba.halftime.FEED_ERRORS as exc:
            return 0, nba.feed_issue("NBA live pricing or account check", exc)
        if result["status"] in ("live_order_recorded", "live_order_unknown"):
            status = result.get("state") or result["status"]
            label = f"{quote['side']} · {quote['quantity']} shares · ${quote['total_estimate']} screened cost · {status}"
            print(label, flush=True)
            return 1, None
        return 0, result["status"].replace("_", " ")

    return nba.scan_games(check, stream=stream, explain=explain)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--live", action="store_true", help="Enable real-money automatic orders")
    mode.add_argument("--check-account", action="store_true", help="Check authentication and USD buying power")
    parser.add_argument("--env-file", type=Path, help="Use this credentials file instead of the project .env")
    parser.add_argument("--explain", action="store_true", help="Show each considered game's scan status")
    args = parser.parse_args()
    try:
        key, secret = live_credentials(args.env_file)
        client = live_client(key, secret)
        bankroll = buying_power(client)
        if args.check_account:
            print("Polymarket US authentication verified.", flush=True)
            print(f"Available USD buying power · ${bankroll:,.2f}", flush=True)
            print(f"Per-game cap before quarter Kelly · ${stake_cap(bankroll):.2f}", flush=True)
            return
        stream = BookStream(signing_credentials(key, secret), Event(), request_id="nba-halftime-kelly")
        orders = AutomaticOrders(LIVE_DB, client, max_book_age=MAX_BOOK_AGE, retry_expired=True)
    except (ImportError, OSError, ValueError, sqlite3.Error) as exc:
        parser.error(str(exc))

    stop = Event()
    worker = Thread(target=stream.follow, args=(stop,), daemon=True)
    worker.start()
    print(f"LIVE · NBA halftime quarter Kelly · $5 and 2% per-game caps · {LIVE_DB}", flush=True)
    if stake_cap(bankroll) == 0:
        print("Current per-game cap is $0.00. Run --check-account to see available USD buying power.", flush=True)
    last_refresh = float("-inf")
    try:
        while True:
            started = monotonic()
            games, attempts, messages = scan(orders, client, stream, explain=args.explain)
            print(f"{games} live NBA games · {attempts} new live order attempts", flush=True)
            for message in messages:
                print(message, flush=True)
            if started - last_refresh >= 30:
                try:
                    orders.refresh()
                except Exception as exc:
                    print(f"Order status refresh unavailable ({type(exc).__name__})", flush=True)
                last_refresh = started
            stop.wait(max(1, nba.SCAN_INTERVAL - (monotonic() - started)))
    except KeyboardInterrupt:
        return
    finally:
        stop.set()
        worker.join(timeout=2)


if __name__ == "__main__":
    main()
