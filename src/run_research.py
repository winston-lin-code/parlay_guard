"""Start the Parlay Guard dashboard and the MLB/NHL halftime scanner."""

import argparse
import base64
import os
from pathlib import Path
import sqlite3

import uvicorn

from .pregame_capture import KEY_ID_VAR, SECRET_VAR, read_env_file

ROOT = Path(__file__).resolve().parents[1]


def live_client(env_file):
    from polymarket_us import PolymarketUS

    key, secret = os.getenv(KEY_ID_VAR), os.getenv(SECRET_VAR)
    if not (key and secret):
        key, secret = os.getenv("POLYMARKET_KEY_ID"), os.getenv("POLYMARKET_SECRET_KEY")
    if env_file or not (key and secret):
        path = env_file or ROOT / ".env"
        if not path.is_file():
            raise ValueError(f"Set {KEY_ID_VAR} and {SECRET_VAR} in the project .env or environment.")
        key, secret = read_env_file(path)
    try:
        size = len(base64.b64decode(secret, validate=True))
    except ValueError:
        size = 0
    if size not in (32, 64):
        raise ValueError("Invalid Polymarket US secret key format.")

    def client():
        return PolymarketUS(key_id=key, secret_key=secret, timeout=6, max_retries=0)

    return client


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--live-halftime", action="store_true",
                        help="Automatically submit real MLB/NHL halftime orders with a $5 all-in cap per game")
    parser.add_argument("--env-file", type=Path, help="Polymarket US credentials for --live-halftime")
    args = parser.parse_args()
    if args.env_file and not args.live_halftime:
        parser.error("--env-file requires --live-halftime")
    os.environ["PG_DISABLE_AUTO_REFRESH"] = "0"
    os.environ["PG_ENABLE_PRO_HALFTIME_PAPER"] = "1"
    os.environ["PARLAY_GUARD_US_ORDERS"] = "0"

    from .api import app, inplay_scanner
    from .pro_halftime_orders import DEFAULT_LIVE_DB, HalftimeOrders

    if args.live_halftime:
        try:
            inplay_scanner.pro_halftime_orders = HalftimeOrders(ROOT / DEFAULT_LIVE_DB, live_client(args.env_file))
        except ImportError:
            parser.error("Install the project requirements to use Polymarket US live orders.")
        except (OSError, ValueError, sqlite3.Error) as exc:
            parser.error(str(exc))

    uvicorn.run(app, host=args.host, port=args.port, access_log=False)


if __name__ == "__main__":
    main()
