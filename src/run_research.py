"""Start the Parlay Guard dashboard and its live sports scanners."""

import argparse
import base64
import os
from pathlib import Path
import sqlite3

import uvicorn

from .env_config import env_value
from .pregame_capture import KEY_ID_VAR, SECRET_VAR, read_env_file

ROOT = Path(__file__).resolve().parents[1]


def live_client(env_file):
    from polymarket_us import PolymarketUS

    if env_file:
        key, secret = read_env_file(env_file)
    else:
        key, secret = env_value(KEY_ID_VAR), env_value(SECRET_VAR)
        if not (key and secret):
            key, secret = env_value("POLYMARKET_KEY_ID"), env_value("POLYMARKET_SECRET_KEY")
        if not (key and secret):
            raise ValueError(f"Set {KEY_ID_VAR} and {SECRET_VAR} in the project .env or environment.")
    try:
        size = len(base64.b64decode(secret, validate=True))
    except ValueError:
        size = 0
    if size not in (32, 64):
        raise ValueError("Invalid Polymarket US secret key format.")

    def client():
        return PolymarketUS(key_id=key, secret_key=secret, timeout=6, max_retries=0)

    return client


def verify_account(client):
    try:
        with client() as exchange:
            response = exchange.account.balances()
        if not isinstance(response, dict) or not isinstance(response.get("balances"), list):
            raise ValueError("Unexpected balance response")
    except Exception as exc:
        raise ValueError(f"Polymarket US account check failed ({type(exc).__name__}). "
                         "Check your API key and connection.") from None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--live-halftime", action="store_true",
                        help="Automatically submit real MLB/NHL halftime orders with a $5 all-in cap per game")
    parser.add_argument("--live-wcbb", action="store_true",
                        help="Automatically submit real WCBB anytime orders with a $5 all-in cap per game")
    parser.add_argument("--env-file", type=Path, help="Override the project .env for Polymarket US credentials")
    args = parser.parse_args()
    if args.env_file and not (args.live_halftime or args.live_wcbb):
        parser.error("--env-file requires --live-halftime or --live-wcbb")
    os.environ["PG_DISABLE_AUTO_REFRESH"] = "0"
    os.environ["PG_ENABLE_PRO_HALFTIME_PAPER"] = "1"
    os.environ["PARLAY_GUARD_US_ORDERS"] = "0"

    from .api import app, inplay_scanner
    from .pro_halftime_orders import DEFAULT_LIVE_DB, AutomaticOrders
    from .wcbb_anytime import BOOK_MAX_AGE, LIVE_DB_PATH

    if args.live_halftime or args.live_wcbb:
        try:
            client = live_client(args.env_file)
            if args.live_wcbb:
                verify_account(client)
                inplay_scanner.wcbb_orders = AutomaticOrders(ROOT / LIVE_DB_PATH, client, max_book_age=BOOK_MAX_AGE)
            if args.live_halftime:
                inplay_scanner.pro_halftime_orders = AutomaticOrders(ROOT / DEFAULT_LIVE_DB, client)
        except ImportError:
            parser.error("Install the project requirements to use Polymarket US live orders.")
        except (OSError, ValueError, sqlite3.Error) as exc:
            parser.error(str(exc))

    if args.live_wcbb:
        print("Polymarket US authentication verified. WCBB real-money auto orders enabled.", flush=True)
    uvicorn.run(app, host=args.host, port=args.port, access_log=False)


if __name__ == "__main__":
    main()
