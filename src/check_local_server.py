"""Report the state of a running local Parlay Guard dashboard using HTTP GETs."""

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import re
from urllib.error import HTTPError
from urllib.request import ProxyHandler, build_opener


def read_json(opener, url):
    try:
        with opener.open(url, timeout=4) as response:
            data = json.load(response)
    except HTTPError as exc:
        return None, f"HTTP {exc.code}"
    except OSError:
        return None, "connection failed or timed out"
    except ValueError:
        return None, "expected a JSON response"
    if not isinstance(data, dict):
        return None, "expected a JSON object"
    return data, None


def update_status(stamp, now, stale_after):
    if not stamp:
        return "no update reported"
    try:
        age = (now - datetime.fromisoformat(stamp.replace("Z", "+00:00"))).total_seconds()
    except (AttributeError, TypeError, ValueError):
        return "invalid update timestamp"
    if age < -5:
        return "update timestamp is ahead of this computer's clock"
    status = "stale" if age > stale_after else "fresh"
    return f"{status}, last update {max(age, 0):.0f}s ago"


def print_scanners(data, now):
    print(f"Combined EV scanner: {update_status(data.get('updated_utc'), now, 35)}")
    leagues = Counter(game.get("league") or "Unknown" for game in data.get("games", []) if game.get("phase") == "live")
    labels = ", ".join(f"{league} ({count})" for league, count in sorted(leagues.items()))
    print(f"Live games in EV scanner: {labels or 'none reported'}")
    print(f"Sports listings: {data.get('sports', {}).get('status', 'status unavailable')}")

    halftime = data.get("halftime_paper")
    if halftime is not None:
        print(f"Other halftime scanner: {update_status(halftime.get('updated_utc'), now, 35)}")
        if halftime.get("leagues"):
            print("  Leagues: " + ", ".join(halftime["leagues"]))
        if halftime.get("warnings"):
            print(f"  Scanner warnings: {len(halftime['warnings'])}")

    wcbb = data.get("wcbb_anytime")
    if wcbb is not None:
        live = wcbb.get("live_ordering") is True
        mode = "REAL-MONEY AUTO ORDERS ENABLED" if live else "PAPER MODE"
        print(f"WCBB anytime scanner: {mode}")
        print(f"  Feed: {update_status(wcbb.get('updated_utc'), now, 35)}")
        if live:
            orders = wcbb.get("orders", [])
            print(f"  Recent real orders: {len(orders)}")
            if orders:
                latest = orders[0]
                print(f"  Latest: {latest['side']} | {latest['state']} | {latest.get('order_id', 'no order ID')}")

    pro = data.get("pro_halftime")
    if pro is None:
        print("NHL/MLB halftime scanner: status unavailable on this server")
        return
    if pro.get("live_ordering") is True:
        mode = "REAL-MONEY AUTO ORDERS ENABLED"
    elif pro.get("paper_logging") is True:
        mode = "PAPER MODE"
    elif not pro.get("updated_utc"):
        mode = "WAITING FOR FIRST SCAN OR SCANNER STOPPED"
    else:
        mode = "RESEARCH PREVIEW"
    print(f"NHL/MLB halftime scanner: {mode}")
    print(f"  Feed: {update_status(pro.get('updated_utc'), now, 35)}")
    if pro.get("stake_cap") is not None:
        print(f"  Per-game cap: ${pro['stake_cap']}")
    for scan in pro.get("scans", []):
        status = scan.get("status")
        checked = scan.get("listings_checked")
        label = status.replace("_", " ") if status else "scan complete"
        count = f" ({checked} live listings checked)" if checked is not None else ""
        upcoming = f" | next start ET {scan['next_start_et']}" if scan.get("next_start_et") else ""
        print(f"  {scan['league'].upper()}: {label}{count}{upcoming}")
    entries = pro.get("entries", [])
    print(f"  Recent ledger entries: {len(entries)}")
    if entries:
        latest = entries[0]
        state = latest.get("state") or "paper entry logged"
        print(f"  Latest: {latest.get('event', 'unknown game')} | {state} | {latest.get('recorded_at', 'time unavailable')}")
    if pro.get("warning"):
        print("  Halftime ledger warning. See the dashboard for details.")
    if pro.get("live_ordering"):
        print("  Exchange authentication and fills are unverified by this check.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8000, help="Loopback port (default: 8000)")
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")
    base = f"http://127.0.0.1:{args.port}"
    opener = build_opener(ProxyHandler({}))

    health, error = read_json(opener, base + "/api/health")
    if error or "live_dashboard_available" not in health or "refresh" not in health:
        print(f"Parlay Guard at {base}: {error or 'unexpected health response'}")
        return 1
    print(f"Parlay Guard at {base}: reachable")
    print("Dashboard page: " + ("available" if health["live_dashboard_available"] else "unavailable"))
    now = datetime.now(timezone.utc)
    refresh = health["refresh"]
    freshness = "stale" if refresh.get("stale") is True else "fresh" if refresh.get("stale") is False else "unknown"
    last_refresh = update_status(refresh.get("refresh_finished_utc"), now, 1200)
    print(f"NBA dashboard data: {freshness} | refresh {'running' if refresh.get('refreshing') else 'idle'}")
    print(f"  Refresh worker: {last_refresh}")
    if error := refresh.get("last_error"):
        summary = re.match(r"[A-Za-z]+Error(?:: [1-5][0-9]{2}\b)?", error)
        print(f"  Last refresh failed{f' ({summary[0]})' if summary else ''}.")

    data, error = read_json(opener, base + "/api/inplay/opportunities")
    if error:
        print(f"Scanner status: {error}")
    else:
        print_scanners(data, datetime.now(timezone.utc))

    orders, _ = read_json(opener, base + "/api/inplay/us-wager/status")
    enabled = orders.get("direct_orders") if orders is not None else None
    mode = "enabled" if enabled is True else "disabled" if enabled is False else "status unavailable"
    print(f"Manual order endpoint: {mode}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
