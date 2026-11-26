"""Start all Parlay Guard dashboard workers with MLB/NHL halftime paper logging."""

import argparse
import os

import uvicorn


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    os.environ["PG_DISABLE_AUTO_REFRESH"] = "0"
    os.environ["PG_ENABLE_PRO_HALFTIME_PAPER"] = "1"
    os.environ["PARLAY_GUARD_US_ORDERS"] = "0"
    uvicorn.run("src.api:app", host=args.host, port=args.port, access_log=False)


if __name__ == "__main__":
    main()
