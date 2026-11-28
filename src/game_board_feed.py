"""Polymarket sports updates and fresh two-sided indicative moneyline asks."""

import json
import time
from datetime import datetime, timedelta, timezone

import requests
from websockets.exceptions import WebSocketException
from websockets.sync.client import connect

from .inplay_scan import MAX_AGE, _books, as_num, utc

SPORTS_SOCKET = "wss://sports-api.polymarket.com/ws"
MAX_SCORE_AGE = timedelta(minutes=3)
FINAL_SCORE_AGE = timedelta(hours=6)
MAX_FIXTURES = 48


def parse_score(message):
    if not isinstance(message, dict) or type(message.get("gameId")) is not int:
        return None
    if message.get("ended") is True:
        phase = "final"
    elif message.get("live") is True:
        phase = "live"
    else:
        return None
    return message["gameId"], {
        "event_slug": str(message.get("slug") or ""),
        "score_league": str(message.get("leagueAbbreviation") or "").upper(),
        "score_columns": [str(message.get(side) or "") for side in ("homeTeam", "awayTeam")],
        "phase": phase, "score_text": str(message.get("score") or ""),
        "score_order": " / ".join(str(message.get(side) or "") for side in ("homeTeam", "awayTeam")).strip(" /"),
        "period_text": str(message.get("period") or "") if phase == "live" else "",
        "clock_text": str(message.get("elapsed") or "") if phase == "live" else "",
        "score_utc": datetime.now(timezone.utc).isoformat(),
    }


def follow_scores(stop, publish):
    while not stop.is_set():
        try:
            with connect(SPORTS_SOCKET, open_timeout=8, close_timeout=1, max_size=64 * 1024) as socket:
                last_frame = time.monotonic()
                while not stop.is_set():
                    try:
                        message = socket.recv(timeout=2)
                    except TimeoutError:
                        if time.monotonic() - last_frame > 30:
                            break
                        continue
                    last_frame = time.monotonic()
                    if message == "ping":
                        socket.send("pong")
                        continue
                    try:
                        state = parse_score(json.loads(message))
                    except (ValueError, TypeError):
                        continue
                    if state:
                        publish(*state)
        except (OSError, WebSocketException, TimeoutError):
            stop.wait(5)


def best_ask(book, now):
    timestamp = as_num(book.get("timestamp"))
    if timestamp is None or not -5 <= now.timestamp() - timestamp / 1000 <= MAX_AGE:
        return None
    asks = []
    for level in book.get("asks") or []:
        price, size = as_num(level.get("price")), as_num(level.get("size"))
        if price is not None and size is not None and 0 < price < 1 and size > 0:
            asks.append(price)
    if asks:
        return {"ask": min(asks), "quote_utc": datetime.fromtimestamp(timestamp / 1000, timezone.utc).isoformat()}


def moneyline_fixtures(fixtures, now):
    upcoming = []
    for fixture in fixtures:
        tip = utc(fixture.get("start_utc"))
        if fixture.get("moneyline") and tip is not None and now - timedelta(hours=8) <= tip <= now + timedelta(hours=36):
            upcoming.append((tip, fixture))
    upcoming.sort(key=lambda item: (item[1].get("phase") != "live", abs(item[0] - now)))
    return [fixture for _, fixture in upcoming[:MAX_FIXTURES]]


def scan_moneylines(fixtures, *, books_reader=None):
    selected = moneyline_fixtures(fixtures, datetime.now(timezone.utc))
    tokens = list(dict.fromkeys(token for fixture in selected for _, token in fixture["moneyline"]))
    try:
        books = (books_reader or _books)(tokens) if tokens else {}
    except (requests.RequestException, OSError, ValueError, KeyError, TypeError):
        return {}
    now = datetime.now(timezone.utc)
    result = {}
    for fixture in selected:
        prices = []
        for label, token in fixture["moneyline"]:
            quote = best_ask(books.get(token, {}), now)
            if quote:
                prices.append({"side": label, "token_id": token, **quote})
        if prices:
            result[fixture["event"]] = prices
    return result
