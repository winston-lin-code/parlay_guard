"""Watch the NBA and NFL halftime paper rule from the terminal."""

import argparse
from datetime import datetime, timezone
import os
import time

from .halftime_paper import (
    LEAGUES, MIN_PROBABILITY, MIN_SCREENED_EV, PAPER_STAKE, PROBABILITY_HAIRCUT, prepare_selection,
    price_selection, screen_halftime,
)
from .inplay_scan import ESPN_SCOREBOARD, read_json, utc
from .nfl_inplay_scan import NFL_SCOREBOARD
from .predict_today import ET

POLL_SECONDS = 2
MAX_ALERT_AGE = 5
PREPARED_SECONDS = 30


def fast_scores(league, day):
    endpoint = ESPN_SCOREBOARD if league == "NBA" else NFL_SCOREBOARD
    return read_json(
        "GET", endpoint, max_age=MAX_ALERT_AGE, headers={"Cache-Control": "no-cache"},
        params={"dates": day.strftime("%Y%m%d"), "limit": 100, "_": time.time_ns()},
    )["events"]


class WatchPrices:
    """Keep model and market mappings briefly. Always fetch the order book anew."""

    def __init__(self):
        self.prepared = {}

    def price(self, game):
        key = (game["league"], game["id"], game["tip"], game["scores"]["home"], game["scores"]["away"])
        now = time.monotonic()
        cached = self.prepared.get(key)
        if cached is None or cached[0] <= now:
            selection = prepare_selection(game)
            expires = time.monotonic() + (PREPARED_SECONDS if selection is not None else POLL_SECONDS)
            self.prepared[key] = expires, selection
        else:
            _, selection = cached
        return price_selection(game, selection) if selection is not None else None

    def discard_expired(self):
        now = time.monotonic()
        self.prepared = {key: value for key, value in self.prepared.items() if value[0] > now}


def signal_key(pick):
    return pick["event"], pick["side"]


def alertable_picks(picks, now):
    for pick in picks:
        quoted = utc(pick["quote_utc"])
        if quoted is not None and -5 <= (now - quoted).total_seconds() <= MAX_ALERT_AGE:
            yield pick


def emit(pick, *, bell):
    now = datetime.now(timezone.utc)
    quoted = utc(pick["quote_utc"])
    age = (now - quoted).total_seconds()
    if not -5 <= age <= MAX_ALERT_AGE:
        return False
    maximum_cost = (pick["probability"] - PROBABILITY_HAIRCUT) / (1 + MIN_SCREENED_EV)
    print(f"[{now.astimezone(ET):%H:%M:%S %Z}] {pick['league']} | {pick['title']} | {pick['score_text']} at halftime\n"
          f"  {pick['side']} moneyline | Model {pick['probability']:.1%} | Screened EV {pick['screened_ev']:+.1%} | "
          f"Quote {max(0, age):.1f}s old\n"
          f"  Avg. ask {pick['ask']:.3f} | Fee-inclusive {pick['cost']:.3f} per share | "
          f"Maximum fee-inclusive cost under rule {maximum_cost:.3f}\n"
          f"  {pick['url']}", flush=True)
    if bell:
        if os.name == "nt":
            import winsound
            try:
                winsound.MessageBeep()
            except RuntimeError:
                print("\a", end="", flush=True)
        else:
            print("\a", end="", flush=True)
    return True


def watch(*, leagues, interval, once, bell):
    prices = WatchPrices()
    active = {}
    last_status = None
    print(f"Halftime watcher | {', '.join(leagues)} | {MIN_PROBABILITY:.0%} model, "
          f"{MIN_SCREENED_EV:+.0%} screened EV | ${PAPER_STAKE} paper stake | {interval:g}s polling", flush=True)
    while True:
        start = time.monotonic()
        prices.discard_expired()
        snapshot = screen_halftime(leagues=leagues, score_feed=fast_scores, price_feed=prices.price)
        current = {
            signal_key(pick): pick for pick in alertable_picks(snapshot["picks"], datetime.now(timezone.utc))
        }
        for key, pick in active.items():
            if key not in current:
                print(f"Quote no longer verified | {pick['title']} | {pick['side']}", flush=True)
        for key, pick in current.items():
            previous = active.get(key)
            if previous is None:
                if emit(pick, bell=bell):
                    active[key] = pick
                continue
            cost_drop = previous["cost"] - pick["cost"]
            if abs(cost_drop) >= .01 or previous["score_text"] != pick["score_text"]:
                if emit(pick, bell=bell and cost_drop >= .01):
                    active[key] = pick
        active = {key: pick for key, pick in active.items() if key in current}
        status = snapshot["games_at_halftime"], len(active), tuple(snapshot["warnings"])
        if status != last_status:
            print(f"{status[0]} at halftime | {status[1]} qualifying", flush=True)
            for warning in status[2]:
                print(warning, flush=True)
            last_status = status
        if once:
            return
        time.sleep(max(0, interval - (time.monotonic() - start)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--league", choices=(*LEAGUES, "all"), default="all")
    parser.add_argument("--interval", type=float, default=POLL_SECONDS, help="Seconds between scan starts, minimum 2")
    parser.add_argument("--once", action="store_true", help="Run one scan and exit")
    parser.add_argument("--bell", action="store_true", help="Play an alert sound for new qualifying quotes")
    args = parser.parse_args()
    if not 2 <= args.interval <= 60:
        parser.error("--interval must be between 2 and 60 seconds")
    try:
        watch(leagues=LEAGUES if args.league == "all" else (args.league,),
              interval=args.interval, once=args.once, bell=args.bell)
    except KeyboardInterrupt:
        print("\nWatcher stopped.", flush=True)


if __name__ == "__main__":
    main()
