"""Discover scheduled sports games on Gamma across leagues and group their markets."""

import re
from datetime import datetime, timedelta, timezone

from .inplay_scan import GAMMA, read_json, utc
from .college_sources import LEAGUES
from .pro_ev_scan import PRO_SERIES

SPORTS_TAG = 1
SPORT_LABELS = {
    "american-football": "American football", "football": "American football",
    "nfl": "American football", "cfb": "American football", "ncaaf": "American football",
    "basketball": "Basketball", "baseball": "Baseball", "hockey": "Hockey",
    "ice-hockey": "Hockey", "soccer": "Soccer", "tennis": "Tennis", "esports": "Esports",
    "mma": "MMA", "boxing": "Boxing", "cricket": "Cricket", "rugby": "Rugby",
    "golf": "Golf", "volleyball": "Volleyball", "handball": "Handball",
    "formula1": "Motorsport", "motorsports": "Motorsport",
}
SAFE_SLUG = re.compile(r"[a-z0-9][a-z0-9-]*")
DATED_FIXTURE = re.compile(r"^(.+?-\d{4}-\d{2}-\d{2})(?:-|$)")
PAST = timedelta(hours=18)
AHEAD = timedelta(hours=36)
MAX_PAGES = 100
COLLEGE_SERIES = {data["series"]: sport for sport, data in LEAGUES.items()}
COLLEGE_FIELDS = ("id", "question", "outcomes", "clobTokenIds", "line", "sportsMarketType",
                  "feesEnabled", "feeSchedule", "gameStartTime", "description")


def league_index(records):
    leagues = {}
    for entry in records if isinstance(records, list) else []:
        if not isinstance(entry, dict):
            continue
        series = str(entry.get("series") or "")
        label = str(entry.get("name") or entry.get("sport") or "").strip()
        if series.isdigit() and label:
            leagues[series] = label
    return leagues


def _fixture(event, now, leagues):
    slug = event.get("slug") or ""
    if not SAFE_SLUG.fullmatch(slug) or event.get("closed") or event.get("ended"):
        return None
    markets = [m for m in event.get("markets") or [] if m.get("active") is True and not m.get("closed")]
    if not markets:
        return None
    market_times = [utc(m.get("gameStartTime")) for m in markets]
    market_start = next((time for time in market_times if time is not None), None)
    tags = {x.get("slug"): x.get("label") for x in event.get("tags") or [] if x.get("slug")}
    if "games" not in tags and market_start is None:
        return None
    start = utc(event.get("startTime") or event.get("gameStartTime")) or market_start
    if start is None or not now - PAST <= start <= now + AHEAD:
        return None
    series = event.get("series") or []
    series_slug = event.get("seriesSlug") or (series[0].get("slug") if series else "")
    series_id = str(series[0].get("id") or "") if series else ""
    college = COLLEGE_SERIES.get(series_id)
    pro = PRO_SERIES.get(series_id)
    league = (leagues.get(series_id) or tags.get(series_slug) or
              (series[0].get("title") if series else None) or series_slug)
    sport = next((label for tag, label in SPORT_LABELS.items() if tag != "football" and tag in tags),
                 SPORT_LABELS.get("football") if "football" in tags else league or "Other sports")
    league = league or sport
    if "nba" in tags:
        sport, league = "Basketball", "NBA"
    match = DATED_FIXTURE.match(slug)
    key = match[1] if match else slug
    phase = "live" if event.get("live") is True else "pregame" if start > now else "unknown"
    active = {str(m["id"]) for m in markets if m.get("id") and m.get("acceptingOrders") is True
              and m.get("enableOrderBook") is True}
    title = str(event.get("title") or "").strip()
    if not title:
        return None
    return {"event": key, "slug": slug, "title": title, "sport": sport, "league": league,
            "phase": phase, "start_utc": start.isoformat(),
            "score_text": str(event.get("score") or "") if phase == "live" else "",
            "clock_text": str(event.get("elapsed") or "") if phase == "live" else "",
            "period_text": str(event.get("period") or "") if phase == "live" else "",
            "open_market_ids": active, "college": college, "pro": pro,
            "college_markets": [{field: market.get(field) for field in COLLEGE_FIELDS}
                                for market in markets if (college or pro) and str(market.get("id")) in active
                                and market.get("sportsMarketType") in ("moneyline", "spreads", "totals")]}


def add_fixtures(fixtures, events, now, leagues):
    for event in events:
        row = _fixture(event, now, leagues)
        if row is None:
            continue
        key = row["event"]
        previous = fixtures.get(key)
        if previous is None:
            fixtures[key] = row
            continue
        # Prefer the main matchup over a child event titled for one market.
        preferred = row if row["slug"] == key else previous
        other = previous if preferred is row else row
        preferred["open_market_ids"].update(other["open_market_ids"])
        if (other["college"] or other["pro"]) == (preferred["college"] or preferred["pro"]):
            preferred["college_markets"] = list({market["id"]: market for market in
                                                 [*other["college_markets"], *preferred["college_markets"]]}.values())
        if preferred["phase"] == "unknown" and other["phase"] == "live":
            for field in ("phase", "score_text", "clock_text", "period_text"):
                preferred[field] = other[field]
        fixtures[key] = preferred


def _snapshot(fixtures, now, status):
    rows = [{key: value for key, value in fixture.items() if key != "open_market_ids"} |
            {"open_markets": len(fixture["open_market_ids"])} for fixture in fixtures.values()]
    return {"status": status, "observed_utc": now.isoformat(), "games": rows}


def _catalog_page(params, stop):
    try:
        return read_json("GET", GAMMA + "/events/keyset", params=params)
    except OSError:
        if stop.wait(1):
            raise
        return read_json("GET", GAMMA + "/events/keyset", params=params)


def scan_sports_events(stop, on_progress=None):
    now = datetime.now(timezone.utc)
    try:
        leagues = league_index(read_json("GET", GAMMA + "/sports"))
    except (ValueError, TypeError, OSError, KeyError):
        leagues = {}
    params = {"tag_id": SPORTS_TAG, "active": "true", "closed": "false", "limit": 50,
              "end_date_min": (now - PAST).isoformat(),
              "end_date_max": (now + AHEAD + timedelta(days=1)).isoformat()}
    fixtures, cursors = {}, set()
    status = "partial"
    for _ in range(MAX_PAGES):
        if stop.is_set():
            break
        try:
            page = _catalog_page(params, stop)
            if not isinstance(page.get("events"), list):
                raise ValueError("Missing sports events")
            add_fixtures(fixtures, page["events"], now, leagues)
        except (ValueError, TypeError, OSError, KeyError):
            break
        cursor = page.get("next_cursor")
        if not cursor:
            status = "complete"
            break
        if cursor in cursors:
            break
        if on_progress and fixtures:
            on_progress(_snapshot(fixtures, now, "scanning"))
        cursors.add(cursor)
        params["after_cursor"] = cursor
    return _snapshot(fixtures, now, status)
