"""Parlay Guard: read-only NBA player-prop market scanner for Polymarket.

Discovery: Gamma NBA GAME series (default 10345), not general NBA futures.
Pricing:   CLOB POST /books with batches of Yes/No token IDs.

Install: pip install requests pandas numpy pyarrow
Usage:
    python polymarket_nba_ev.py --event-slug nba-gsw-mem-2026-11-19
    python polymarket_nba_ev.py --pages 2 --max-games 12
    python polymarket_nba_ev.py --pages 2 --debug --no-orderbooks

CSV: data/polymarket_nba_ev.csv

Important: Player markets saying 'O/U' can use outcome names Yes/No.
The scanner verifies Yes=Over per market using Gamma settlement text, and
leaves the outcome as YES/NO when resolution wording is insufficient.
EV is computed only for description-verified rows with current projections.
Be especially cautious: some markets resolve NO if a player does not play,
so a conditional-on-playing model may overstate a YES probability.

Optional data/current_projections.csv columns:
    player_name,stat,prediction,as_of_utc
(e.g. 'Ty Jerome,PTS,18.4,2026-11-19T09:00:00Z')
Historical prediction residuals (if available):
    data/processed/player_predictions_2025-26.parquet
with PTS/PRED_PTS, REB/PRED_REB, AST/PRED_AST, FG3M/PRED_FG3M.

Market quotes, model probabilities and estimated EV are research outputs;
they are not trading instructions, guarantees, or validated profitability.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

GAMMA = "https://gamma-api.polymarket.com"
CLOB = "https://clob.polymarket.com"
NBA_GAME_SERIES = "10345"
HISTORY = Path("data/processed/player_predictions_2025-26.parquet")
PROJECTIONS = Path("data/current_projections.csv")
OUTPUT = Path("data/polymarket_nba_ev.csv")

STAT_TYPES = {
    "points": ("PTS", "Points"),
    "rebounds": ("REB", "Rebounds"),
    "assists": ("AST", "Assists"),
    "threes": ("FG3M", r"(?:3-Pointers Made|3 Pointers Made|Three-Pointers Made|Threes)"),
}
HIST_COLUMNS = {
    "PTS": ("PTS", "PRED_PTS"),
    "REB": ("REB", "PRED_REB"),
    "AST": ("AST", "PRED_AST"),
    "FG3M": ("FG3M", "PRED_FG3M"),
}
COLUMNS = [
    "observed_utc", "event", "event_slug", "game_start_utc",
    "market_id", "market_question", "player_name", "stat", "line",
    "outcome", "side", "outcome_mapping", "no_play_resolves_no", "token_id",
    "ask_price", "ask_size_shares", "bid_price", "bid_size_shares",
    "gamma_indicative_price", "book_status", "book_timestamp",
    "fee_allowance_per_share", "effective_cost", "break_even_probability",
    "projection", "model_probability", "estimated_ev_per_dollar",
    "market_url",
]


def make_session() -> requests.Session:
    session = requests.Session()
    session.headers.update({"User-Agent": "ParlayGuardResearch/0.2 (read-only)"})
    retries = Retry(
        total=2,
        connect=2,
        read=2,
        status=2,
        backoff_factor=0.6,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=frozenset(["GET", "POST"]),
        respect_retry_after_header=True,
    )
    adapter = HTTPAdapter(max_retries=retries, pool_connections=8, pool_maxsize=8)
    session.mount("https://", adapter)
    return session


def request_json(session: requests.Session, method: str, url: str, **kwargs):
    response = session.request(method, url, timeout=(5, 12), **kwargs)
    response.raise_for_status()
    return response.json()


def as_list(value) -> list:
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except (TypeError, ValueError):
            return []
        return value if isinstance(value, list) else []
    return []


def clean_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(name).casefold())


def parse_player_market(market: dict):
    """Parse observed Gamma format: 'Ty Jerome: Points O/U 22.5'."""
    kind = str(market.get("sportsMarketType") or "").lower()
    if kind not in STAT_TYPES:
        return None
    stat, word_pattern = STAT_TYPES[kind]
    question = str(market.get("question") or "").strip()
    pattern = rf"(?P<player>[^:]+):\s*{word_pattern}\s+O/U\s+(?P<line>\d+(?:\.\d+)?)"
    match = re.fullmatch(pattern, question, flags=re.IGNORECASE)
    if not match:
        return None
    line = float(match.group("line"))
    # Integer lines may push / differ in settlement; support unambiguous half-lines only.
    if not math.isclose(line % 1, 0.5):
        return None
    return match.group("player").strip(), stat, line


def verified_over_mapping(market: dict, player: str, line: float) -> tuple[str, bool]:
    """Verify the actual YES/NO rule for this individual market.

    Require both halves of the market's resolution text to agree with the
    exact half-point line. Do not infer a mapping from the question alone.
    """
    description = str(market.get("description") or "")
    inactive_no = bool(re.search(
        r"(?:inactive|does not take the court|does not play).*?resolve\s+[\"'“”]?No\b",
        description, re.I | re.S,
    ))
    if not description:
        return "unverified", False
    # Require the rule to refer to the same named player, not another prop
    # in a shared general-purpose description.
    if clean_name(player) not in clean_name(description):
        return "unverified", inactive_no
    marker = r"(?:this market|the market)\s+will\s+resolve\s+to\s*[\"'“”]?"
    yes = re.search(
        marker + r"Yes[\"'“”]?\s+if\s+(.*?)" +
        r"(?=" + marker + r"No\b|\Z)",
        description, re.I | re.S,
    )
    no = re.search(
        marker + r"No[\"'“”]?\s+if\s+(.*?)" +
        r"(?=The entire game|If this market|If the game|If the player|" +
        r"The resolution source|\Z)",
        description, re.I | re.S,
    )
    if not yes or not no:
        return "unverified", inactive_no
    line_re = re.escape(f"{line:g}") + r"(?!\d)"
    yes_over = re.search(r"\b(?:more than|over|greater than)\s+" + line_re,
                         yes.group(1), re.I)
    no_under = re.search(line_re +
                         r"\s+(?:(?:[\w-]+\s+){0,4})?or\s+(?:fewer|less)\b",
                         no.group(1), re.I)
    if yes_over and no_under:
        return "description_yes_over", inactive_no
    return "unverified", inactive_no


def parse_utc(value):
    if not value:
        return None
    parsed = pd.to_datetime(value, utc=True, errors="coerce")
    if pd.isna(parsed):
        return None
    try:
        return parsed.to_pydatetime()
    except (AttributeError, ValueError, OverflowError):
        return None


def market_tipoff(event: dict, market: dict | None = None):
    """Use gameStartTime only; event startDate can mean market creation time."""
    values = []
    if market:
        values.append(market.get("gameStartTime"))
    values.append(event.get("gameStartTime"))
    for value in values:
        dt = parse_utc(value)
        if dt:
            return dt
    return None


def event_tipoff(event: dict):
    dt = market_tipoff(event)
    if dt:
        return dt
    for market in event.get("markets") or []:
        if isinstance(market, dict):
            dt = market_tipoff(event, market)
            if dt:
                return dt
    return None


def discover_events(session, *, pages, series_id, event_slug, debug):
    if event_slug:
        url = GAMMA + "/events/slug/" + event_slug
        event = request_json(session, "GET", url)
        if isinstance(event, list):
            return event
        if isinstance(event, dict):
            return [event]
        raise ValueError("Unexpected event-by-slug response")

    events = []
    for page in range(pages):
        params = {
            "series_id": series_id,
            "active": "true",
            "closed": "false",
            "limit": 100,
            "offset": page * 100,
        }
        try:
            batch = request_json(session, "GET", GAMMA + "/events", params=params)
        except requests.HTTPError as exc:
            if events:
                print(f"WARNING: stopped Gamma pagination at page {page + 1}: {exc}")
                break
            raise
        except requests.RequestException as exc:
            if events:
                print(f"WARNING: stopped Gamma pagination at page {page + 1}: {exc}")
                break
            raise
        if not isinstance(batch, list):
            raise ValueError("Expected an array from Gamma /events")
        events.extend(batch)
        if debug:
            print(f"Gamma page {page + 1}: {len(batch)} events")
        if len(batch) < 100:
            break
    return events


def gather_candidates(events, now, max_games, debug):
    # Deduplicate and select nearest future games rather than request thousands
    # of books for historical/in-progress events returned as active by Gamma.
    future = []
    skipped_unverified = 0
    skipped_started = 0
    seen_events = set()
    for event in events:
        event_key = str(event.get("id") or event.get("slug") or "")
        if event_key in seen_events:
            continue
        seen_events.add(event_key)
        tipoff = event_tipoff(event)
        if tipoff is None:
            skipped_unverified += 1
            continue
        if tipoff <= now:
            skipped_started += 1
            continue
        future.append((tipoff, event))

    future.sort(key=lambda item: item[0])
    if max_games:
        future = future[:max_games]

    rows = []
    seen_markets = set()
    unmatched_examples = []
    counts = Counter()
    for tipoff, event in future:
        if debug:
            types = Counter(str(m.get("sportsMarketType") or "unknown")
                            for m in event.get("markets", []) if isinstance(m, dict))
            print(f"GAME {event.get('title')} ({tipoff.isoformat()}), market types: {dict(types)}")
        for market in event.get("markets") or []:
            if not isinstance(market, dict):
                continue
            if market.get("closed") or market.get("active") is False:
                continue
            parsed = parse_player_market(market)
            if parsed is None:
                if debug and str(market.get("sportsMarketType") or "").lower() in STAT_TYPES and len(unmatched_examples) < 5:
                    unmatched_examples.append(str(market.get("question")))
                continue
            player, stat, line = parsed
            rule_mapping, inactive_no = verified_over_mapping(market, player, line)
            market_id = str(market.get("id") or market.get("conditionId") or "")
            if not market_id:
                continue
            if market_id in seen_markets:
                continue
            game_time = market_tipoff(event, market) or tipoff
            if game_time <= now:
                continue
            tokens = as_list(market.get("clobTokenIds"))
            outcomes = as_list(market.get("outcomes"))
            indicative_prices = as_list(market.get("outcomePrices"))
            if len(tokens) != 2 or len(outcomes) != 2 or {str(x).lower() for x in outcomes} != {"yes", "no"}:
                counts["bad_token_mapping"] += 1
                continue
            seen_markets.add(market_id)
            counts[stat] += 1
            for idx, (outcome, token) in enumerate(zip(outcomes, tokens)):
                token = str(token).strip()
                if not token:
                    continue
                indicative = indicative_prices[idx] if len(indicative_prices) == 2 else None
                try:
                    indicative = float(indicative) if indicative is not None else None
                except (ValueError, TypeError):
                    indicative = None
                rows.append({
                    "event": event.get("title", ""),
                    "event_slug": event.get("slug", ""),
                    "game_start_utc": game_time.isoformat(),
                    "market_id": market_id,
                    "market_question": market.get("question", ""),
                    "player_name": player,
                    "stat": stat,
                    "line": line,
                    "outcome": str(outcome).upper(),
                    "outcome_mapping": rule_mapping,
                    "no_play_resolves_no": inactive_no,
                    "token_id": token,
                    "gamma_indicative_price": indicative,
                    "market_url": "https://polymarket.com/event/" + str(event.get("slug") or ""),
                })

    print(f"Events discovered: {len(events)} | Future games selected: {len(future)} "
          f"| Skipped started: {skipped_started} | Unknown tipoff: {skipped_unverified}")
    print(f"Eligible player markets: {sum(counts[s] for s in HIST_COLUMNS)} "
          f"| Breakdown: {dict(counts)} | Outcome tokens: {len(rows)}")
    if debug and unmatched_examples:
        print("Unparsed player markets (samples):", unmatched_examples)
    return rows


def best_level(book: dict, key: str, *, is_ask: bool):
    levels = []
    for item in book.get(key) or []:
        try:
            price, size = float(item["price"]), float(item["size"])
            if 0 < price <= 1 and size > 0:
                levels.append((price, size))
        except (KeyError, TypeError, ValueError):
            continue
    if not levels:
        return None, None
    return (min(levels) if is_ask else max(levels))


def fetch_books(session, token_ids: list[str], batch_size: int, debug: bool):
    """Return a map token_id -> book; partial data survives failed chunks."""
    unique_tokens = list(dict.fromkeys(token_ids))
    books = {}
    failures = 0
    chunks = (len(unique_tokens) + batch_size - 1) // batch_size
    for idx in range(chunks):
        batch = unique_tokens[idx * batch_size : (idx + 1) * batch_size]
        print(f"CLOB batch {idx + 1}/{chunks}: requesting {len(batch)} token books...", flush=True)
        batch_books, failed = fetch_book_chunk(session, batch, debug)
        books.update(batch_books)
        failures += failed
        print(f"  Returned: {len(batch_books)} books | Missing/failed tokens in batch: "
              f"{len(batch) - len(batch_books)}", flush=True)
    if failures:
        print(f"WARNING: {failures} CLOB token requests failed; corresponding rows remain in CSV without an ask price.")
    return books


def fetch_book_chunk(session, tokens: list[str], debug=False):
    payload = [{"token_id": token} for token in tokens]
    try:
        data = request_json(session, "POST", CLOB + "/books", json=payload)
        if not isinstance(data, list):
            raise ValueError("CLOB /books returned a non-array response")
    except requests.HTTPError as exc:
        # One invalid token can reject a batch: bisect only for 400/422.
        code = exc.response.status_code if exc.response is not None else None
        if code in {400, 422} and len(tokens) > 1:
            mid = len(tokens) // 2
            left, lfail = fetch_book_chunk(session, tokens[:mid], debug)
            right, rfail = fetch_book_chunk(session, tokens[mid:], debug)
            left.update(right)
            return left, lfail + rfail
        if debug:
            print(f"  CLOB request failed (HTTP {code}): {str(exc)[:160]}")
        return {}, len(tokens)
    except (requests.RequestException, ValueError) as exc:
        if debug:
            print(f"  CLOB request failed: {str(exc)[:160]}")
        return {}, len(tokens)

    wanted = set(tokens)
    books = {}
    for book in data:
        if not isinstance(book, dict):
            continue
        token = str(book.get("asset_id") or book.get("token_id") or "")
        if token in wanted:
            books[token] = book
    return books, len(tokens) - len(books)


def load_projections(now: datetime):
    if not PROJECTIONS.exists():
        return {}
    df = pd.read_csv(PROJECTIONS)
    required = {"player_name", "stat", "prediction", "as_of_utc"}
    if not required.issubset(df.columns):
        raise ValueError(f"{PROJECTIONS} is missing: {sorted(required - set(df.columns))}")
    df["stat"] = df["stat"].astype(str).str.upper().str.strip()
    df["as_of_utc"] = pd.to_datetime(df["as_of_utc"], utc=True, errors="coerce")
    df["prediction"] = pd.to_numeric(df["prediction"], errors="coerce")
    df["normalized_player"] = df["player_name"].map(clean_name)
    cutoff = pd.Timestamp(now) - pd.Timedelta(hours=36)
    df = df[
        df["stat"].isin(HIST_COLUMNS)
        & df["prediction"].notna()
        & df["as_of_utc"].notna()
        & (df["as_of_utc"] <= pd.Timestamp(now))
        & (df["as_of_utc"] >= cutoff)
    ]
    df = df.sort_values("as_of_utc").drop_duplicates(["normalized_player", "stat"], keep="last")
    return {(r.normalized_player, r.stat): (float(r.prediction), r.as_of_utc.to_pydatetime())
            for r in df.itertuples()}


def load_residuals():
    if not HISTORY.exists():
        print(f"NOTE: historical residual file not found: {HISTORY}; EV unavailable.")
        return {}
    # Read only columns actually present; newer players might not be in history.
    try:
        import pyarrow.parquet as pq
        available = set(pq.ParquetFile(HISTORY).schema.names)
        wanted = [x for stat_pair in HIST_COLUMNS.values() for x in stat_pair if x in available]
        df = pd.read_parquet(HISTORY, columns=sorted(set(wanted)))
    except (ImportError, OSError, ValueError, KeyError) as exc:
        print(f"NOTE: historical file could not be loaded ({exc}); EV unavailable.")
        return {}
    residuals = {}
    for stat, (actual, pred) in HIST_COLUMNS.items():
        if actual not in df or pred not in df:
            continue
        actuals = pd.to_numeric(df[actual], errors="coerce")
        predictions = pd.to_numeric(df[pred], errors="coerce")
        values = (actuals - predictions).to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        if len(values) >= 100:
            residuals[stat] = np.sort(values)
    return residuals


def probability_over(residuals: np.ndarray, projection: float, line: float):
    threshold = line - projection
    count = len(residuals) - np.searchsorted(residuals, threshold, side="right")
    return float((count + 1) / (len(residuals) + 2))


def enrich_rows(rows, books, *, now, fee_per_share, projections, residuals, no_orderbooks):
    for row in rows:
        row["observed_utc"] = now.isoformat()
        mapping_verified = row["outcome_mapping"] == "description_yes_over"
        row["side"] = ("OVER" if row["outcome"] == "YES" else "UNDER") if mapping_verified else row["outcome"]
        book = books.get(row["token_id"])
        if book is None:
            row["book_status"] = "not_requested" if no_orderbooks else "unavailable"
            row["ask_price"] = None
            row["ask_size_shares"] = None
            row["bid_price"] = None
            row["bid_size_shares"] = None
            row["book_timestamp"] = None
        else:
            row["ask_price"], row["ask_size_shares"] = best_level(book, "asks", is_ask=True)
            row["bid_price"], row["bid_size_shares"] = best_level(book, "bids", is_ask=False)
            row["book_status"] = "priced" if row["ask_price"] is not None else "no_asks"
            row["book_timestamp"] = book.get("timestamp")
        row["fee_allowance_per_share"] = fee_per_share
        price = row["ask_price"]
        row["effective_cost"] = (price + fee_per_share) if price is not None else None
        row["break_even_probability"] = row["effective_cost"] if price is not None else None
        row["projection"] = None
        row["model_probability"] = None
        row["estimated_ev_per_dollar"] = None
        pred_record = projections.get((clean_name(row["player_name"]), row["stat"]))
        if pred_record is None:
            continue
        prediction, as_of = pred_record
        start = parse_utc(row["game_start_utc"])
        if start is None or not (as_of < start):
            continue
        row["projection"] = prediction
        if not mapping_verified or price is None or row["stat"] not in residuals:
            continue
        over = probability_over(residuals[row["stat"]], prediction, row["line"])
        p = over if row["outcome"] == "YES" else 1 - over
        cost = row["effective_cost"]
        row["model_probability"] = p
        # If cost >= $1, the price cannot provide a positive return on a $1 payout.
        row["estimated_ev_per_dollar"] = (p - cost) / cost if cost > 0 else None
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pages", type=int, default=2, help="Gamma pages to inspect (100 events/page; default 2)")
    parser.add_argument("--event-slug", help="Inspect one specific game event")
    parser.add_argument("--series-id", default=NBA_GAME_SERIES, help="NBA game series (default 10345)")
    parser.add_argument("--max-games", type=int, default=12, help="Nearest future games to price (default 12)")
    parser.add_argument("--batch-size", type=int, default=100, help="CLOB tokens per POST /books (1..500; default 100)")
    parser.add_argument("--no-orderbooks", action="store_true", help="Parse props without requesting CLOB prices")
    parser.add_argument("--debug", action="store_true", help="Show discovery and parsing diagnostics")
    parser.add_argument("--fee-per-share", type=float, default=0.0, help="Manual fee allowance in dollars per share; default 0")
    parser.add_argument("--output", type=Path, default=OUTPUT, help="CSV output path")
    args = parser.parse_args(argv)
    if args.pages < 1 or args.max_games < 1 or not 1 <= args.batch_size <= 500:
        parser.error("--pages and --max-games must be >= 1; --batch-size must be 1..500")
    if not 0 <= args.fee_per_share < 1:
        parser.error("--fee-per-share must be >= 0 and < 1")

    now = datetime.now(timezone.utc)
    session = make_session()
    try:
        events = discover_events(session, pages=args.pages, series_id=args.series_id,
                                 event_slug=args.event_slug, debug=args.debug)
        rows = gather_candidates(events, now, max_games=args.max_games if not args.event_slug else 1,
                                 debug=args.debug)
        tokens = [row["token_id"] for row in rows]
        books = {} if args.no_orderbooks or not tokens else fetch_books(session, tokens, args.batch_size, args.debug)
    except requests.RequestException as exc:
        print(f"ERROR: Polymarket API request failed: {exc}")
        return 1
    finally:
        session.close()

    projections = load_projections(now)
    residuals = load_residuals() if projections else {}
    enrich_rows(rows, books, now=now, fee_per_share=args.fee_per_share,
                projections=projections,
                residuals=residuals, no_orderbooks=args.no_orderbooks)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows, columns=COLUMNS)
    df.to_csv(args.output, index=False)
    print(f"Saved: {args.output}")
    if df.empty:
        print("No eligible upcoming player markets found. Try --debug or --event-slug.")
        return 0

    priced = int((df["book_status"] == "priced").sum())
    print(f"Priced outcomes: {priced}/{len(df)}")
    verified_count = int((df["outcome_mapping"] == "description_yes_over").sum())
    print(f"Settlement verified against market descriptions: {verified_count}/{len(df)} outcomes")
    if verified_count < len(df):
        print("Unverified outcomes remain YES/NO and their EV is withheld.")
    if df["no_play_resolves_no"].any():
        print("CAUTION: Inactive/no-play rules may resolve to NO, not a void. Model EV must account for this.")
    if not projections:
        print(f"No fresh current projections in {PROJECTIONS}; EV unavailable.")
    if df["estimated_ev_per_dollar"].notna().any():
        print("Exploratory EV estimates available; fees/slippage/model calibration are not validated.")
        display = df.sort_values("estimated_ev_per_dollar", ascending=False, na_position="last")
    else:
        display = df.sort_values(["game_start_utc", "player_name", "stat", "line", "outcome"])
    shown = ["event", "player_name", "stat", "line", "outcome", "ask_price",
             "ask_size_shares", "book_status", "model_probability", "estimated_ev_per_dollar"]
    print(display[shown].head(25).to_string(index=False))
    print("Note: Gamma indicative prices are NOT executable CLOB ask quotes.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
