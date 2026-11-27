from datetime import datetime, timedelta, timezone

import pytest

from src import wcbb_anytime as paper
from src.inplay_api import ScanState

NOW = datetime(2026, 11, 27, 18, tzinfo=timezone.utc)
EVENT = "wcbb-duke-unc-2026-11-27"
MARKET = "aec-" + EVENT


def fixture(period=1, score=16):
    return {
        "id": "40123", "date": (NOW - timedelta(minutes=20)).isoformat(),
        "competitions": [{
            "status": {"period": period, "clock": 100, "displayClock": "1:40",
                       "type": {"state": "in", "name": "STATUS_IN_PROGRESS"}},
            "competitors": [
                {"homeAway": "home", "score": str(score),
                 "team": {"id": "1", "abbreviation": "UNC", "shortDisplayName": "North Carolina"}},
                {"homeAway": "away", "score": "10",
                 "team": {"id": "2", "abbreviation": "DUKE", "shortDisplayName": "Duke"}},
            ],
        }],
    }


def inputs(period, probability, bid, size=20):
    contract = {
        "slug": MARKET, "question": "Duke vs North Carolina",
        "active": True, "closed": False, "marketType": "moneyline",
        "sportsMarketType": "basketball_team_full_game_winner", "description": "Overtime is included.",
        "gameStartTime": fixture()["date"], "orderPriceMinTickSize": ".01", "minimumTradeQty": "1",
        "feeCoefficient": ".11",
        "marketSides": [
            {"long": False, "tradable": True, "team": {"name": "North Carolina (W)"}},
            {"long": True, "tradable": True, "team": {"name": "Duke (W)"}},
        ],
    }
    summary = {
        "header": {"id": "40123"},
        "plays": [{"id": "play-1", "wallclock": (NOW - timedelta(seconds=20)).isoformat(),
                   "period": {"number": period}, "homeScore": 16, "awayScore": 10}],
        "winprobability": [{"playId": "play-1", "homeWinPercentage": probability}],
    }
    book = {"marketSlug": MARKET, "state": "MARKET_STATE_OPEN", "transactTime": NOW.isoformat(),
            "bids": [{"px": {"currency": "USD", "value": bid}, "qty": str(size)}]}
    return contract, summary, book


@pytest.mark.parametrize(("period", "probability", "bid", "status"), [
    (1, .73, ".50", "pending_recheck"),
    (4, .73, ".50", "pending_recheck"),
    (5, .73, ".50", "pending_recheck"),
    (3, .60, ".50", "below_probability"),
    (3, .73, ".81", "below_price"),
    (3, .66, ".40", "below_ev"),
])
def test_anytime_entry_thresholds_and_depth(monkeypatch, period, probability, bid, status):
    contract, summary, book = inputs(period, probability, bid, size=4)
    monkeypatch.setattr(paper, "now", lambda: NOW)
    monkeypatch.setattr(paper, "read_json", lambda *args, **kwargs: summary)
    monkeypatch.setattr(paper, "public", lambda *args, **kwargs: {"marketData": book})
    row, entry = paper.scan_game(paper.game_state(fixture(period)), [(EVENT, contract)], {})
    assert row["status"] == status
    if status == "pending_recheck":
        assert entry["quantity"] == 4
        assert entry["cost"] == "2.15"
        assert entry["side"] == "North Carolina"
        assert entry["long"] is False
        assert entry["screened_ev"] >= .05
    else:
        assert entry is None


def test_rejects_newer_or_changed_state(monkeypatch):
    _, summary, _ = inputs(4, .75, ".50")
    game = paper.game_state(fixture(4))
    assert paper.forecast(summary, game, NOW)
    summary["plays"].append({**summary["plays"][0], "id": "play-2", "homeScore": 18,
                             "wallclock": (NOW - timedelta(seconds=2)).isoformat()})
    assert paper.forecast(summary, game, NOW) is None
    summary["plays"] = [summary["plays"][-1]]
    assert paper.forecast(summary, paper.game_state(fixture(4, 18)), NOW) is None


def test_discovers_open_moneyline_before_exchange_marks_game_live(monkeypatch):
    contract, _, _ = inputs(1, .73, ".50")
    delayed = {"slug": EVENT, "live": False, "ended": False, "markets": [contract]}
    finished = {**delayed, "ended": True}
    outside = {**delayed, "markets": [{**contract, "gameStartTime": (NOW - timedelta(days=3)).isoformat()}]}
    start_window = (NOW - timedelta(days=1), NOW + timedelta(days=1))

    def inventory_response(*args, params, **kwargs):
        assert params["tagSlug"] == "wcbb"
        assert "live" not in params
        assert params["startDateMin"] == start_window[0].isoformat()
        assert params["startDateMax"] == start_window[1].isoformat()
        return {"events": [delayed, finished, outside]}

    monkeypatch.setattr(paper.halftime_paper, "read_json", inventory_response)
    listings, live = paper.halftime_paper.live_inventory("WCBB", start_window=start_window)
    assert live == []
    game = {**paper.game_state(fixture()), "listed_moneylines": listings["WCBB"]}
    event, market, assignments = paper.halftime_paper.contract_and_sides(game)
    assert (event["slug"], market["slug"], assignments) == (EVENT, MARKET, {"away": True, "home": False})


def test_scan_persists_once_and_uses_exchange_settlement(monkeypatch, tmp_path):
    path = tmp_path / "wcbb.sqlite3"
    contract, summary, book = inputs(4, .75, ".50")
    monkeypatch.setattr(paper, "now", lambda: NOW)
    monkeypatch.setattr(paper.halftime_paper, "live_inventory",
                        lambda league, *, start_window: ({"WCBB": [(EVENT, contract)]}, []))
    monkeypatch.setattr(paper, "read_json", lambda *args, **kwargs: summary)
    monkeypatch.setattr(paper, "public", lambda *args, **kwargs: {"marketData": book})
    calls = iter([fixture(4), fixture(4, 18)])
    monkeypatch.setattr(paper.halftime_paper, "scores", lambda *args: [next(calls)])
    rejected = paper.scan(path)
    assert rejected["games"][0]["status"] == "state_changed"
    assert rejected["entries"] == []
    monkeypatch.setattr(paper.halftime_paper, "scores", lambda *args: [fixture(4)])
    accepted = paper.scan(path)
    assert accepted["performance"]["open"] == 1
    assert accepted["performance"]["roi"] is None
    assert len(paper.scan(path)["entries"]) == 1
    monkeypatch.setattr(paper, "now", lambda: NOW + timedelta(minutes=60))
    def market_result(status):
        return lambda route: ({"slug": MARKET, "settlement": "0"} if route.endswith("/settlement")
                              else {"market": {"slug": MARKET, "status": status}})

    monkeypatch.setattr(paper, "public", market_result("MARKET_STATUS_OPEN"))
    paper.settle(path)
    assert paper.read_ledger(path)[0]["settlement"] is None
    monkeypatch.setattr(paper, "public", market_result("MARKET_STATUS_RESOLVED"))
    paper.settle(path)
    settled = paper.scan(path)
    assert settled["performance"]["wins"] == 1
    assert settled["performance"]["open"] == 0
    assert settled["performance"]["roi"] > 0
    assert settled["entries"][0]["result"] == "Won"
    state = ScanState()
    state.wcbb = settled
    assert state.read()["wcbb_anytime"] == settled
