import asyncio
import json
from datetime import datetime, timedelta, timezone

from fastapi import FastAPI

from src import inplay_api
from src.game_board_feed import best_ask, parse_score
from src.inplay_api import ScanState, agrees_with_board
from src.sports_inventory import _fixture


def test_public_moneyline_requires_distinct_valid_sides():
    now = datetime.now(timezone.utc)
    start = (now + timedelta(hours=2)).isoformat()
    market = {
        "id": "1", "question": "Aces vs. Liberty", "sportsMarketType": "moneyline",
        "outcomes": ["Aces", "Liberty"], "clobTokenIds": ["10", "20"], "gameStartTime": start,
        "active": True, "enableOrderBook": True, "acceptingOrders": True,
    }
    event = {"slug": "wnba-lva-nyl-2026-11-25", "title": "Aces vs. Liberty", "gameId": 42,
             "startTime": start, "markets": [market]}
    fixture = _fixture(event, now, {})
    assert fixture["moneyline"] == [("Aces", "10"), ("Liberty", "20")]
    assert fixture["sports_game_id"] == 42
    market["clobTokenIds"] = ["10", "10"]
    assert _fixture(event, now, {})["moneyline"] == []


def test_live_score_invalidates_pregame_pick_without_hiding_asks():
    now = datetime.now(timezone.utc)
    event = "nba-away-home"
    upcoming = {"event": event, "phase": "pregame", "start_utc": (now + timedelta(minutes=1)).isoformat()}
    state = ScanState()
    quote = {"quote_utc": now.isoformat(), "ask": .54, "side": "Home", "token_id": "10"}
    state.sports["games"] = [{**upcoming, "sports_game_id": 42, "moneyline": [("Home", "10")]}]
    state.snapshot = {"updated_utc": now.isoformat(), "games": [upcoming], "opportunities": [
        {**quote, "event": event, "phase": "pregame", "probability": .7, "screened_ev": .1}
    ]}
    state.moneyline_quotes[event] = [quote]
    assert state.read()["top_picks"] == {event: "10"}
    update = {"gameId": 42, "live": True, "score": "28-21", "period": "Q1",
              "homeTeam": "Home", "awayTeam": "Away"}
    state.update_score(*parse_score(update))
    snapshot = state.read()
    assert snapshot["top_picks"] == {}
    assert snapshot["board"][0]["score_text"] == "28-21"
    assert snapshot["board"][0]["moneyline_quotes"] == [quote]
    assert "sports_game_id" not in snapshot["sports"]["games"][0]
    state.update_score(*parse_score({**update, "ended": True}))
    assert state.read()["board"][0]["moneyline_quotes"] == []


def test_best_ask_ignores_empty_price_levels_and_expired_books():
    now = datetime.now(timezone.utc)
    book = {"timestamp": now.timestamp() * 1000,
            "asks": [{"price": ".41", "size": "0"}, {"price": ".62", "size": "100"}, {"price": ".48", "size": "2"}]}
    assert best_ask(book, now)["ask"] == .48
    assert best_ask(book, now + timedelta(seconds=36)) is None


def test_newer_score_rejects_mismatched_live_ev():
    now = datetime.now(timezone.utc)
    score = {"phase": "live", "score_text": "38-30", "score_utc": now.isoformat()}
    assert not agrees_with_board({"phase": "live", "score": [30, 35]}, score, now)
    assert agrees_with_board({"phase": "live", "score": [30, 38]}, score, now)
    assert agrees_with_board({"phase": "live", "score": [30, 42]}, score, now + timedelta(seconds=1))


def test_dashboard_socket_reprices_from_book_updates_and_keeps_expired_pick(monkeypatch):
    now = datetime.now(timezone.utc)
    event, token = "nba-away-home", "10"
    stamp = int(now.timestamp() * 1000)
    state = ScanState()
    view = {"event": event, "phase": "live", "start_utc": now.isoformat(), "priced_sides": 0}
    pick = {"event": event, "phase": "live", "token_id": token, "side": "Home",
            "market": "Away vs Home", "probability": .75, "score": [22, 28], "period_text": "Q1"}
    state.sports["games"] = [{**view, "sports_game_id": 42, "moneyline": [("Home", token)]}]
    state.snapshot = {"updated_utc": now.isoformat(), "games": [view], "opportunities": [],
                      "rest_books": {}, "quote_candidates": [(pick, (0, 1))]}
    state.market_books.observe_market_terms({token: {"min_order_size": 5}})
    state.market_books.apply({"event_type": "book", "asset_id": token, "timestamp": stamp,
                              "asks": [{"price": ".55", "size": "25"}, {"price": ".80", "size": "50"}]}, {token})
    monkeypatch.setattr(inplay_api, "scanner", state)
    app = FastAPI()
    app.include_router(inplay_api.router)

    def change(timestamp, price, size):
        state.market_books.apply({"event_type": "price_change", "timestamp": timestamp, "price_changes": [
            {"asset_id": token, "price": price, "size": size, "side": "SELL"}]}, {token})

    async def exercise():
        incoming, outgoing = asyncio.Queue(), asyncio.Queue()
        scope = {"type": "websocket", "path": "/ws/inplay", "headers": [(b"host", b"testserver")],
                 "query_string": b"", "scheme": "ws"}
        await incoming.put({"type": "websocket.connect"})
        task = asyncio.create_task(app(scope, incoming.get, outgoing.put))

        async def receive():
            return await asyncio.wait_for(outgoing.get(), timeout=3)

        try:
            assert (await receive())["type"] == "websocket.accept"
            frame = json.loads((await receive())["text"])
            assert frame["top_picks"] == {event: token}
            assert frame["opportunities"][0]["book_source"] == "websocket"
            assert frame["board"][0]["moneyline_quotes"][0]["ask"] == .55
            assert "quote_candidates" not in frame and "rest_books" not in frame

            change(stamp + 1, ".55", "0")
            frame = json.loads((await receive())["text"])
            assert frame["opportunities"] == []
            assert frame["recent_picks"][event][0]["ask"] == .55
            assert frame["board"][0]["moneyline_quotes"][0]["ask"] == .80

            change(stamp - 1, ".40", "100")
            assert state.read()["opportunities"] == []
            change(stamp + 2, ".55", "25")
            assert json.loads((await receive())["text"])["top_picks"] == {event: token}

            state.update_score(*parse_score({"gameId": 42, "slug": event, "live": True,
                                            "score": "28-22", "period": "Q2"}))
            frame = json.loads((await receive())["text"])
            assert frame["opportunities"] == []
            assert frame["recent_picks"][event][0]["period_text"] == "Q1"
            assert state.score_wake["nba"].is_set()
        finally:
            await incoming.put({"type": "websocket.disconnect", "code": 1000})
            await asyncio.wait_for(task, timeout=3)

    asyncio.run(exercise())
    assert state.remember_picks([], {}, now + timedelta(seconds=123)) == {}
