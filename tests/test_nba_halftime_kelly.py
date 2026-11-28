from copy import deepcopy
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from unittest.mock import MagicMock

import pytest

from src import nba_halftime_kelly as live, nba_moneyline_shadow as nba, pro_halftime_orders
from src.us_wager import WagerError


NOW = datetime(2026, 11, 28, 1, tzinfo=timezone.utc)
EVENT = "nba-pho-hou-2026-11-27"
MARKET = "aec-" + EVENT


@pytest.fixture
def fixture():
    tip = NOW - timedelta(minutes=40)
    event = {"id": "40123", "date": tip.isoformat(), "season": {"type": 2}, "competitions": [{
        "status": {"period": 2, "clock": 222, "displayClock": "3:42",
                   "type": {"state": "in", "name": "STATUS_IN_PROGRESS"}},
        "competitors": [{"homeAway": "home", "team": {"id": "10", "abbreviation": "HOU"}, "score": "58"},
                        {"homeAway": "away", "team": {"id": "20", "abbreviation": "PHX"}, "score": "51"}],
    }]}
    contract = {"slug": MARKET, "question": "Suns vs Rockets", "marketType": "moneyline",
                "sportsMarketType": "basketball_team_full_game_winner", "minimumTradeQty": "1",
                "orderPriceMinTickSize": ".01", "feeCoefficient": ".11", "gameStartTime": tip.isoformat(),
                "description": "Overtime is included.", "active": True, "closed": False, "marketSides": [
                    {"long": True, "tradable": True, "team": {"abbreviation": "PHX"}},
                    {"long": False, "tradable": True, "team": {"abbreviation": "HOU"}},
                ]}
    summary = {"header": {"id": "40123"}, "plays": [{"id": "play-1", "homeScore": 58, "awayScore": 51,
               "wallclock": (NOW - timedelta(seconds=20)).isoformat(), "period": {"number": 2}}],
               "winprobability": [{"playId": "play-1", "homeWinPercentage": .76}]}
    book = {"marketSlug": MARKET, "transactTime": NOW.isoformat(), "state": "MARKET_STATE_OPEN",
            "bids": [{"px": {"value": ".50", "currency": "USD"}, "qty": "50"}], "offers": []}
    return event, contract, summary, book


def test_kelly_uses_fees_and_slippage_in_the_edge_and_fraction(fixture):
    event, contract, _, book = fixture
    game = live.game_state(event, league="NBA")
    market = {"selected": "home", "probability": Decimal(".611"), "side": contract["marketSides"][1],
              "ask": Decimal(".49"), "book": book, "received": NOW}
    book["bids"][0]["px"]["value"] = ".51"
    sized, reason = live.quarter_kelly_quote(game, contract, market, Decimal(100))
    assert reason is None
    payload, quote, cap = sized
    assert payload["quantity"] == 3
    assert Decimal(payload["price"]["value"]) == Decimal(".51")
    assert quote["total_estimate"] == "1.59"
    assert quote["fee_estimate"] == "0.09"
    assert cap == Decimal("1.64")
    book["bids"][0]["px"]["value"] = ".50"
    market["ask"] = Decimal(".50")
    assert live.quarter_kelly_quote(game, contract, market, Decimal(100), explain=True) == (
        None, "Best affordable screened EV below 5% (+3.89%, 61.1% win probability, $0.500 best ask)")


@pytest.mark.parametrize(("bankroll", "reason"), [
    ("20", "Minimum executable order exceeds the 2%/$5 cap ($0.53 all-in vs $0.40 cap)"),
    ("32", "Quarter-Kelly allocation too small for an EV-qualified order ($0.53 all-in vs $0.52 Kelly budget)"),
])
def test_kelly_explain_separates_hard_stake_cap_from_kelly_allocation(fixture, bankroll, reason):
    event, contract, _, book = fixture
    game = live.game_state(event, league="NBA")
    book["bids"][0]["px"]["value"] = ".51"
    market = {"selected": "home", "probability": Decimal(".611"), "side": contract["marketSides"][1],
              "ask": Decimal(".49"), "book": book, "received": NOW}
    assert live.quarter_kelly_quote(game, contract, market, Decimal(bankroll), explain=True) == (None, reason)


def test_kelly_reports_the_exchange_minimum_when_the_book_is_too_thin(fixture):
    event, contract, _, book = fixture
    game = live.game_state(event, league="NBA")
    contract["minimumTradeQty"] = "2"
    book["bids"][0]["qty"] = "1"
    market = {"selected": "home", "probability": Decimal(".76"), "side": contract["marketSides"][1],
              "ask": Decimal(".50"), "book": book, "received": NOW}
    assert live.quarter_kelly_quote(game, contract, market, Decimal(100)) == (
        None, "Insufficient order-book depth for this quantity.")


@pytest.mark.parametrize("recheck", ["same", "score_changed", "outside_window", "expired_book"])
def test_live_runner_rechecks_before_submitting_one_fok_order(fixture, monkeypatch, tmp_path, recheck):
    event, contract, summary, book = fixture
    now = NOW
    submitted = []
    snapshots = []
    latest = deepcopy(event)
    if recheck == "score_changed":
        latest["competitions"][0]["competitors"][0]["score"] = "60"
    elif recheck == "outside_window":
        latest["competitions"][0]["status"]["clock"] = 301

    def scoreboard(_):
        nonlocal now
        snapshots.append(True)
        if len(snapshots) == 2:
            if recheck == "expired_book":
                now += timedelta(seconds=6)
            return [latest]
        return [event]

    def create(payload):
        submitted.append(payload)
        return {"id": "order-1", "executions": [{"order": {
            "id": "order-1", "marketSlug": MARKET, "state": "ORDER_STATE_FILLED",
            "cumQuantity": payload["quantity"]}}]}

    client = MagicMock()
    exchange = client.return_value.__enter__.return_value
    exchange.orders.create.side_effect = create
    exchange.account.balances.return_value = {"balances": [{"currency": "USD", "buyingPower": 100}]}
    monkeypatch.setattr(nba, "now", lambda: now)
    monkeypatch.setattr(live, "now", lambda: now)
    monkeypatch.setattr(pro_halftime_orders, "now", lambda: now)
    monkeypatch.setattr(nba, "scoreboard", scoreboard)
    monkeypatch.setattr(nba, "fresh_book", lambda _: book)
    monkeypatch.setattr(nba.halftime, "live_inventory", lambda *args, **kwargs: ({"NBA": [(EVENT, contract)]}, []))
    monkeypatch.setattr(nba.halftime, "read_json", lambda *args, **kwargs: summary)
    path = tmp_path / "orders.sqlite3"
    orders = pro_halftime_orders.AutomaticOrders(path, client, max_book_age=live.MAX_BOOK_AGE)
    _, attempts, messages = live.scan(orders, client, None)
    if recheck != "same":
        assert attempts == 0 and not submitted and not orders.recent()
        assert messages
        return
    assert attempts == 1 and not messages
    assert submitted[0]["quantity"] == 3
    assert submitted[0]["tif"] == "TIME_IN_FORCE_FILL_OR_KILL"
    assert submitted[0]["manualOrderIndicator"] == "MANUAL_ORDER_INDICATOR_AUTOMATIC"
    assert Decimal(submitted[0]["price"]["value"]) == Decimal(".50")
    saved, = orders.recent()
    assert saved["stake_cap"] == "2.00" and saved["quote"]["total_estimate"] == "1.62"
    assert saved["state"] == "ORDER_STATE_FILLED"
    assert live.scan(pro_halftime_orders.AutomaticOrders(path, client), client, None)[1] == 0
    assert len(submitted) == 1


def test_buying_power_requires_valid_usd_balance():
    client = MagicMock()
    account = client.return_value.__enter__.return_value.account
    account.balances.return_value = {"balances": [{"currency": "USD", "currentBalance": 100}]}
    with pytest.raises(WagerError, match="USD buying power unavailable"):
        live.buying_power(client)
    account.balances.return_value = {"currency": "USD", "buyingPower": 100, "currentBalance": 200}
    assert live.buying_power(client) == Decimal("100")
