"""Compare ITF point-score freshness across Polymarket US and Live Tennis API."""

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from getpass import getpass
import json
import os
import re
from statistics import median
from time import monotonic, sleep

import requests

from .tennis_elo import player_key
from .tennis_wager import SCORE_MAX_AGE, WagerError, now, stream_catalog, tennis_contract, us_score, utc

LIVE_MATCHES = "https://api.livetennisapi.com/api/public/v1/matches"
MAX_START_GAP = timedelta(hours=3)


def alternate_catalog(key):
    response = requests.get(
        LIVE_MATCHES, headers={"X-API-Key": key},
        params={"tour": "itf", "draw": "singles", "status": "live", "limit": 200},
        timeout=(4, 6),
    )
    response.raise_for_status()
    catalog = response.json()
    if len(catalog["data"]) >= 200:
        raise ValueError("The alternate live catalog reached its page limit. Coverage is incomplete.")
    return catalog["data"], now()


def feed_names(match):
    players = match.get("players") or {}
    return tuple(player_key((players.get(side) or {}).get("name") or "") for side in ("p1", "p2"))


def matched_feeds(events, other_events):
    paired = []
    for slug, event in events.items():
        if event.get("live") is not True:
            continue
        contract = tennis_contract(event)
        scheduled = utc(event.get("startTime"))
        if not contract or not scheduled:
            continue
        players = {team.get("ordering"): player_key(team.get("name") or "")
                   for side in contract.get("marketSides") or [] if (team := side.get("team"))}
        if set(players) != {"home", "away"} or not all(players.values()) or len(set(players.values())) != 2:
            continue
        names = players["home"], players["away"]
        gender = "men" if slug.startswith("itfme-") else "women"
        matches = []
        for row in other_events:
            start = utc(row.get("scheduled_time"))
            if (row.get("id") is not None and row.get("status") == "live" and not row.get("event_status")
                    and row.get("draw") == "singles" and row.get("gender") == gender and start
                    and abs(start - scheduled) <= MAX_START_GAP and set(feed_names(row)) == set(names)):
                matches.append(row)
        if len(matches) == 1:
            row = matches[0]
            paired.append((slug, event, row, feed_names(row) != names))
    match_counts = Counter(row["id"] for _, _, row, _ in paired)
    return [pair for pair in paired if match_counts[pair[2]["id"]] == 1]


def score_token(games, points):
    if not 1 <= len(games) <= 3 or not all(re.fullmatch(r"[0-7]-[0-7]", pair) for pair in games):
        return None
    if not re.fullmatch(r"(?:[0-9]{1,2}|AD)-(?:[0-9]{1,2}|AD)", points):
        return None
    if games[-1] != "6-6" and any(p not in {"0", "15", "30", "40", "AD"} for p in points.split("-")):
        return None
    return f"{','.join(games)}:{points}"


def common_score(event, row, reverse):
    us = us_score(event, event["score_received_at"])
    if not us:
        return None, None
    games, sep, points = us["score_text"].upper().partition(":")
    # Compare standard set scoring and regular tiebreaks with explicit points.
    sets = [part.strip() for part in games.split(",")]
    period = f"TB{len(sets)}" if sets[-1] == "6-6" else f"S{len(sets)}"
    us_token = score_token(sets, points.strip()) if sep and us["period_text"].upper() == period else None

    score = row.get("score") or {}
    timestamp = utc(score.get("timestamp"))
    games, points = score.get("games"), score.get("points")
    if (not timestamp or score.get("stale") is True or score.get("verdict") or not isinstance(games, list)
            or len(games) != 2 or not all(isinstance(side, list) for side in games)
            or len(games[0]) != len(games[1]) or not isinstance(points, list) or len(points) != 2):
        return us_token, None
    if reverse:
        games, points = games[::-1], points[::-1]
    return us_token, score_token([f"{a}-{b}" for a, b in zip(*games)], "-".join(map(str, points)).upper())


class Comparison:
    def __init__(self):
        self.initial = set()
        self.samples = {}
        self.coverage = Counter({
            name: 0 for name in ("us_live_match_polls", "matched_live_polls", "point_score_pairs",
                                 "us_within_20_seconds", "alternate_within_20_seconds",
                                 "alternate_older_state_recently_confirmed")
        })

    def observe(self, events, alternate_events, alternate_received):
        self.coverage["us_live_match_polls"] += sum(e.get("live") is True for e in events.values())
        for slug, event, row, reverse in matched_feeds(events, alternate_events):
            self.coverage["matched_live_polls"] += 1
            key = (slug, row["id"])
            us, alt = common_score(event, row, reverse)
            if us is None or alt is None:
                continue
            self.coverage["point_score_pairs"] += 1
            if key not in self.samples:
                self.samples[key] = {}
                self.initial.update(((key, us), (key, alt)))
            for source, token, received in (
                ("us", us, event["score_received_at"]), ("alternate", alt, alternate_received),
            ):
                if (key, token) not in self.initial:
                    self.samples[key].setdefault(token, {}).setdefault(source, received)
            for source, stamp, received in (
                ("us", (event.get("eventState") or {}).get("updatedAt"), event["score_received_at"]),
                ("alternate", row["score"]["timestamp"], alternate_received),
            ):
                timestamp = utc(stamp)
                if timestamp and 0 <= (received - timestamp).total_seconds() <= SCORE_MAX_AGE:
                    self.coverage[f"{source}_within_20_seconds"] += 1
            alternate_stamp = utc(row["score"]["timestamp"])
            confirmed_age = row["score"].get("observed_age_seconds")
            if ((alternate_received - alternate_stamp).total_seconds() > SCORE_MAX_AGE
                    and isinstance(confirmed_age, (int, float)) and 0 <= confirmed_age <= SCORE_MAX_AGE):
                self.coverage["alternate_older_state_recently_confirmed"] += 1

    def result(self):
        leads = [(pair["us"] - pair["alternate"]).total_seconds()
                 for match in self.samples.values() for pair in match.values() if len(pair) == 2]
        return {
            **self.coverage, "matches_with_point_score_pairs": len(self.samples),
            "distinct_changes_seen_on_both": len(leads), "alternate_first": sum(lead > 0 for lead in leads),
            "us_first": sum(lead < 0 for lead in leads),
            "median_alternate_lead_seconds": round(median(leads), 2) if leads else None,
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=60, help="Number of polls, up to 100 per run")
    parser.add_argument("--interval", type=int, default=5, help="Seconds between polls, minimum 3")
    args = parser.parse_args()
    if not 1 <= args.samples <= 100 or args.interval < 3:
        parser.error("use 1 to 100 samples and an interval of at least 3 seconds")
    key = os.getenv("LIVE_TENNIS_API_KEY") or getpass("Live Tennis API key: ").strip()
    if not key:
        parser.error("a Live Tennis API key is required")
    comparison = Comparison()
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            for i in range(args.samples):
                started = monotonic()
                us = pool.submit(stream_catalog)
                other = pool.submit(alternate_catalog, key)
                comparison.observe(us.result(), *other.result())
                if (i + 1) % 10 == 0:
                    print(f"{i + 1}/{args.samples} polls complete", flush=True)
                if i + 1 < args.samples:
                    sleep(max(0, args.interval - (monotonic() - started)))
    except KeyboardInterrupt:
        pass
    except requests.HTTPError as exc:
        status = exc.response.status_code
        print(json.dumps(comparison.result(), indent=2))
        parser.exit(1, f"A feed returned HTTP {status}. Check API access or remaining quota.\n")
    except (requests.RequestException, WagerError, ValueError, KeyError, TypeError) as exc:
        parser.exit(1, f"Score comparison stopped ({type(exc).__name__}).\n")
    print(json.dumps(comparison.result(), indent=2))


if __name__ == "__main__":
    main()
