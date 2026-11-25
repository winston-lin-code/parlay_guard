from datetime import datetime, timedelta, timezone

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src import us_wager_api as api
from src.us_wager import WagerError, match_contract, quote_order

EVENT = "nba-chi-cha-2026-11-25"
MARKET = "tsc-nba-chi-cha-2026-11-25-231pt5"
START = datetime(2026, 11, 26, tzinfo=timezone.utc)
PICK = {"event": EVENT, "market_type": "totals", "side": "Under", "market": "Bulls vs Hornets: O/U 231.5",
        "line": 231.5, "probability": .85}
CONTRACT = {"slug": MARKET, "gameStartTime": START.isoformat(), "active": True, "closed": False,
            "sportsMarketType": "basketball_team_full_game_total", "marketType": "totals", "line": 231.5,
            "question": "Will these teams combine for over 231.5?", "minimumTradeQty": .01,
            "orderPriceMinTickSize": .01, "marketSides": [
                {"long": True, "description": "Over", "tradable": True},
                {"long": False, "description": "Under", "tradable": True},
            ]}
US_EVENT = {"slug": EVENT, "active": True, "closed": False, "markets": [CONTRACT]}


def test_matches_exact_line_and_buy_short_uses_complement_price():
    contract, long = match_contract(PICK, US_EVENT, START)
    assert long is False
    book = {"marketSlug": MARKET, "state": "MARKET_STATE_OPEN", "transactTime": START.isoformat(),
            "bids": [{"px": {"value": ".41", "currency": "USD"}, "qty": "2"},
                     {"px": {"value": ".39", "currency": "USD"}, "qty": "15"}]}
    body, summary = quote_order(PICK, contract, long, 10, book, START)
    assert body["intent"] == "ORDER_INTENT_BUY_SHORT"
    assert body["price"]["value"] == "0.39"
    assert summary["limit"] == "0.61"
    assert summary["total_estimate"] == "6.28"
    assert body["tif"] == "TIME_IN_FORCE_FILL_OR_KILL"
    with pytest.raises(WagerError):
        quote_order(PICK, contract, long, 10, book, START + timedelta(seconds=16))
    with pytest.raises(WagerError):
        match_contract({**PICK, "line": 232.5}, US_EVENT, START)
    with pytest.raises(WagerError):
        match_contract(PICK, {**US_EVENT, "markets": [CONTRACT, CONTRACT]}, START)
    with pytest.raises(WagerError):
        match_contract(PICK, {**US_EVENT, "markets": [
            {**CONTRACT, "sportsMarketType": "basketball_team_first_half_total"}]}, START)


def test_preview_and_confirm_is_opt_in_and_single_use(tmp_path, monkeypatch):
    app = FastAPI()
    app.include_router(api.router)
    browser = TestClient(app)
    quantity = 10
    body = {"marketSlug": MARKET, "intent": "ORDER_INTENT_BUY_LONG", "quantity": quantity,
            "price": {"value": "0.25"}, "tif": "TIME_IN_FORCE_FILL_OR_KILL"}
    summary = {"quote_utc": datetime.now(timezone.utc).isoformat(), "screened_ev": .2}
    monkeypatch.setattr(api, "current_order", lambda _: (body, dict(summary)))
    selection = {"event": EVENT, "token_id": "123", "quantity": quantity}
    route = "/api/inplay/us-wager/"
    assert browser.post(route + "preview", json=selection).json()["ticket"] is None
    assert browser.post(route + "confirm", json={"ticket": "a"*48}).status_code == 403
    for name, value in {"PARLAY_GUARD_US_ORDERS": "1", "PARLAY_GUARD_ORDER_TOKEN": "a"*32,
                        "PARLAY_GUARD_ORDER_ORIGIN": "https://parlay.example",
                        "PARLAY_GUARD_ORDER_DB": str(tmp_path / "orders.db"),
                        "POLYMARKET_KEY_ID": "fake", "POLYMARKET_SECRET_KEY": "fake"}.items():
        monkeypatch.setenv(name, value)
    submissions = []

    class Exchange:
        orders = property(lambda self: self)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def preview(self, value):
            return {"order": value["request"]}

        def create(self, value):
            submissions.append(value)
            return {"id": "test-order", "executions": []}

    monkeypatch.setattr(api, "client", lambda _: Exchange())
    assert browser.post(route + "preview", json=selection).status_code == 403
    headers = {"Origin": "https://parlay.example", "X-Order-Token": "a"*32}
    preview = browser.post(route + "preview", json=selection, headers=headers)
    assert preview.status_code == 200
    ticket = preview.json()["ticket"]
    assert submissions == []
    assert browser.post(route + "confirm", json={"ticket": ticket}, headers=headers).json()["id"] == "test-order"
    assert browser.post(route + "confirm", json={"ticket": ticket}, headers=headers).status_code == 409
    assert submissions == [body]
    ticket = browser.post(route + "preview", json=selection, headers=headers).json()["ticket"]
    monkeypatch.setattr(api, "current_order", lambda _: ({**body, "price": {"value": ".30"}}, dict(summary)))
    assert browser.post(route + "confirm", json={"ticket": ticket}, headers=headers).status_code == 409
    assert submissions == [body]


def test_preview_uses_cached_listing_with_separate_fresh_book(monkeypatch):
    import requests

    from src import us_wager

    now = datetime.now(timezone.utc)
    start = now + timedelta(hours=1)
    contract = {**CONTRACT, "gameStartTime": start.isoformat()}
    event = {**US_EVENT, "markets": [contract]}
    row = {**PICK, "phase": "pregame", "token_id": "123", "screened_ev": .2}
    game = {"event": EVENT, "phase": "pregame", "start_utc": start.isoformat()}
    monkeypatch.setattr(api.scanner, "read", lambda: {"games": [game], "opportunities": [row]})

    calls = []

    class Response:
        status_code = 200

        def __init__(self, age, data):
            self.headers = {"Age": str(age)}
            self.data = data

        def raise_for_status(self):
            pass

        def json(self):
            return self.data

    def get(url, **options):
        calls.append(options)
        if "/events/" in url:
            assert "params" not in options
            return Response(23, {"event": event})
        assert options["params"]["_"]
        return Response(0, {"marketData": {
            "marketSlug": MARKET, "state": "MARKET_STATE_OPEN", "transactTime": now.isoformat(),
            "bids": [{"px": {"value": ".40", "currency": "USD"}, "qty": "10"}],
        }})

    monkeypatch.setattr(requests, "get", get)
    selection = api.Selection(event=EVENT, token_id="123", quantity=3)
    _, quote = api.current_order(selection)
    assert quote["limit"] == "0.60"
    api.current_order(selection)
    assert calls[1]["params"]["_"] != calls[3]["params"]["_"]

    monkeypatch.setattr(requests, "get", lambda *_, **kwargs: Response(23, {}))
    with pytest.raises(WagerError, match="stale order book"):
        us_wager.public("/v1/markets/" + MARKET + "/book", fresh_book=True)
