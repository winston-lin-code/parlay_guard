from datetime import datetime, timedelta, timezone

import pytest

from src import halftime_paper as paper
from src.inplay_api import ScanState


NOW = datetime(2026, 11, 26, 3, tzinfo=timezone.utc)
SLUG = "aec-nba-pho-hou-2026-11-25"


def event():
    teams = [("away", "PHX", "Suns", "60"), ("home", "HOU", "Rockets", "67")]
    return {
        "id": "nba-1", "date": "2026-11-26T01:30:00Z", "season": {"type": 2},
        "competitions": [{
            "status": {"period": 2, "clock": 0, "type": {"state": "in", "name": "STATUS_HALFTIME"}},
            "competitors": [
                {"homeAway": side, "score": score, "team": {"abbreviation": code, "shortDisplayName": label}}
                for side, code, label, score in teams
            ],
        }],
    }


def market():
    return {
        "slug": SLUG, "marketType": "moneyline", "sportsMarketType": "basketball_team_full_game_winner",
        "active": True, "closed": False, "gameStartTime": event()["date"],
        "description": "Overtime is included if played.",
        "marketSides": [
            {"id": "101", "long": True, "tradable": True, "team": {"safeName": "Suns"}},
            {"id": "102", "long": False, "tradable": True, "team": {"safeName": "Rockets"}},
        ],
    }


def order_book(bid="0.50", qty="8"):
    def level(price, size):
        return {"px": {"value": price, "currency": "USD"}, "qty": size}
    return {
        "marketSlug": SLUG, "state": "MARKET_STATE_OPEN", "transactTime": NOW.isoformat(),
        "bids": [level(bid, qty)], "offers": [level("0.43", qty)],
    }


@pytest.fixture
def clock(monkeypatch):
    class FrozenClock(datetime):
        @staticmethod
        def now(tz):
            return NOW
    monkeypatch.setattr(paper, "datetime", FrozenClock)


def test_requires_confirmed_halftime():
    fixture = event()
    status = fixture["competitions"][0]["status"]
    assert paper.game_state("NBA", fixture)
    status["type"] = {"state": "in", "shortDetail": "End of 2nd"}
    assert paper.game_state("NBA", fixture)
    status["period"] = 3
    assert paper.game_state("NBA", fixture) is None
    status.update(period=2, clock=7)
    assert paper.game_state("NBA", fixture) is None
    status.update(clock=0, type={"state": "post", "name": "STATUS_HALFTIME"})
    assert paper.game_state("NBA", fixture) is None


@pytest.mark.parametrize(("p", "bid", "selected"), [
    (.599, "0.50", None),
    (.60, "0.50", "Rockets"),
    (.60, "0.47", None),
    (.60, "0.48", None),  # Passes 5% before fees, falls below it after fees.
    (.40, "0.50", "Suns"),
])
def test_applies_model_and_fee_adjusted_ev_thresholds(monkeypatch, clock, p, bid, selected):
    monkeypatch.setattr(paper, "home_probability", lambda game, now: p)
    monkeypatch.setattr(paper, "public", lambda path, **kwargs:
                        {"marketData": order_book(bid)} if kwargs else {"market": market()})
    row = paper.price_game(paper.game_state("NBA", event()))
    if selected is None:
        assert row is None
    else:
        assert row["side"] == selected
        assert row["screened_ev"] >= .05
        assert row["cost"] > row["ask"]
        assert "outcomeId=" + ("102" if selected == "Rockets" else "101") in row["url"]


def test_paper_quote_walks_depth_and_rejects_stale_or_paused_books():
    book = order_book(qty="1")
    assert paper.paper_quote(book, SLUG, False, NOW) is None
    book["bids"].append({"px": {"value": ".40", "currency": "USD"}, "qty": "10"})
    quote = paper.paper_quote(book, SLUG, False, NOW)
    assert .50 < quote["ask"] < .60
    assert quote["cost"] > quote["ask"]
    assert paper.paper_quote(book, SLUG, False, NOW + timedelta(seconds=16)) is None
    assert paper.paper_quote({**book, "state": "MARKET_STATE_PAUSED"}, SLUG, False, NOW) is None


def test_scoreboard_recheck_removes_picks_after_halftime_and_ranks_current_games(monkeypatch, clock):
    def game_by_id(game_id, period):
        fixture = event()
        fixture["id"] = game_id
        fixture["competitions"][0]["status"]["period"] = period
        return fixture

    responses = iter([
        [game_by_id("1", 2), game_by_id("2", 2), game_by_id("3", 2)],
        [game_by_id("1", 3), game_by_id("2", 2), game_by_id("3", 2)],
    ])
    monkeypatch.setattr(paper, "scores", lambda league, day: next(responses)
                        if league == "NBA" and day.isoformat() == "2026-11-25" else [])
    monkeypatch.setattr(paper, "price_game", lambda game: {
        "event": game["id"], "probability": .7, "quote_utc": NOW.isoformat(),
        "screened_ev": {"1": .7, "2": .1, "3": .25}[game["id"]],
    })
    assert [pick["event"] for pick in paper.screen_halftime()["picks"]] == ["3", "2"]


def test_api_never_surfaces_expired_halftime_quotes():
    state = ScanState()
    state.halftime = {"updated_utc": NOW.isoformat(),
                      "picks": [{"quote_utc": NOW.isoformat()}], "warnings": []}
    assert state.read()["halftime_paper"]["picks"] == []
