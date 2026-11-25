from datetime import datetime, timedelta, timezone
import threading

import pytest

from src import inplay_scan as scan
from src.inplay_api import ScanState, rank_opportunities

NOW = datetime(2026, 11, 25, 1, tzinfo=timezone.utc)
PROJECTION = {"home_margin": 4, "projected_total": 224, "home_win_probability": 0.60}
METRICS = {"home_margin_holdout_mae": 11, "total_holdout_mae": 15}


def event(home_score=80, away_score=70, period=3, clock=285, phase="in"):
    return {"id": "espn1", "date": "2026-11-24T23:30:00Z", "season": {"type": 2},
            "competitions": [{"status": {"type": {"state": phase}, "period": period, "clock": clock,
                                         "displayClock": "4:45"},
                              "competitors": [
                                  {"homeAway": "home", "score": str(home_score), "team": {"abbreviation": "NY"}},
                                  {"homeAway": "away", "score": str(away_score), "team": {"abbreviation": "GS"}},
                              ]}]}


def test_live_score_and_clock_move_win_probability():
    def probability(**kwargs):
        game, = scan.parse_games([event(**kwargs)], NOW).values()
        return scan.probability_frame(game, PROJECTION, METRICS)[0]
    assert probability(home_score=80, away_score=70) > probability(home_score=70, away_score=80)
    assert probability(period=4, clock=120) > probability(period=3, clock=285)
    assert not scan.parse_games([event(period=4, clock=30), event(period=5)], NOW)
    assert not scan.parse_games([event(phase="post")], NOW)


@pytest.mark.parametrize(("kind", "question", "labels", "line"), [
    ("moneyline", "Warriors vs. Knicks", ["Warriors", "Knicks"], None),
    ("spreads", "Spread: Warriors (-2.5)", ["Warriors", "Knicks"], -2.5),
    ("totals", "Warriors vs. Knicks: O/U 220.5", ["Under", "Over"], 220.5),
])
def test_market_token_mapping_and_complements(kind, question, labels, line):
    game, = scan.parse_games([event()], NOW).values()
    frame = scan.probability_frame(game, PROJECTION, METRICS)
    market = {"sportsMarketType": kind, "question": question, "outcomes": labels,
              "line": line, "clobTokenIds": ["1", "2"]}
    sides = scan.market_probabilities(market, game, frame)
    assert sides[0][0:2] == ("1", labels[0])
    assert sum(side[2] for side in sides) == pytest.approx(1)
    if kind == "spreads":
        assert sides[0][2] < 0.5
        assert scan.market_probabilities({**market, "outcomes": labels[::-1]}, game, frame) is None
    assert scan.market_probabilities({**market, "description": "First half only"}, game, frame) is None


def test_depth_fees_and_quote_expiry():
    book = {"timestamp": str(int(NOW.timestamp() * 1000)), "min_order_size": 5,
            "asks": [{"price": "0.50", "size": "4"}, {"price": "0.60", "size": "20"}]}
    result = scan.priced_side(book, 0.7, (0.05, 1), NOW)
    assert result["ask"] > result["best_ask"]
    assert 0 < result["screened_ev"] < result["ev"]
    assert result["break_even_probability"] > result["ask"]
    assert abs(result["screened_probability"] / result["break_even_probability"] - 1 - result["screened_ev"]) < 1e-10
    assert scan.priced_side({**book, "asks": book["asks"][:1]}, 0.7, (0, 1), NOW) is None
    assert scan.priced_side(book, 0.7, (0, 1), NOW + timedelta(seconds=36)) is None
    assert scan.fee_schedule({"feesEnabled": True}) is None


def test_active_catalog_paginates_and_deduplicates(monkeypatch):
    def response(method, url, params):
        if "after_cursor" not in params:
            return {"markets": [{"id": "1"}], "next_cursor": "page-2"}
        assert params["after_cursor"] == "page-2"
        return {"markets": [{"id": "1"}, {"id": "2", "acceptingOrders": True, "enableOrderBook": True}]}
    monkeypatch.setattr(scan, "read_json", response)
    states = []
    scan.scan_catalog(threading.Event(), states.append)
    assert states[-1] == {"cataloged": 2, "open_order_books": 1, "status": "complete"}


def test_scan_discards_quotes_if_score_moves(monkeypatch):
    class FrozenClock(datetime):
        @staticmethod
        def now(tz):
            return NOW
    monkeypatch.setattr(scan, "datetime", FrozenClock)
    # The actual request parser uses pandas, independent of the scanner clock.
    day = NOW.astimezone(scan.ET).date()
    responses = iter([[event()], [event(home_score=82)]])
    monkeypatch.setattr(scan, "scoreboard", lambda d: next(responses) if d == day else [])
    monkeypatch.setattr(scan, "game_prior", lambda g: (PROJECTION, METRICS))
    monkeypatch.setattr(scan, "event_markets", lambda g: [
        {"sportsMarketType": "moneyline", "question": "Warriors vs. Knicks",
         "outcomes": ["Warriors", "Knicks"], "clobTokenIds": ["1", "2"], "feesEnabled": False},
    ])
    monkeypatch.setattr(scan, "_books", lambda tokens: {"2": {
        "timestamp": NOW.timestamp()*1000, "min_order_size": 5,
        "asks": [{"price": "0.45", "size": 100}],
    }})
    assert scan.scan_opportunities()["opportunities"] == []
    responses = iter([[event()], [event()]])
    result = scan.scan_opportunities()
    assert result["opportunities"][0]["side"] == "Knicks"
    assert result["opportunities"][0]["phase"] == "live"


def test_expired_snapshot_never_returns_opportunities():
    state = ScanState()
    state.snapshot = {"updated_utc": NOW.isoformat(), "opportunities": [{"quote_utc": NOW.isoformat()}]}
    assert state.read()["stale"]
    assert state.read()["opportunities"] == []


def test_one_paper_pick_per_current_fixture():
    game = {"event": "game-a", "phase": "live"}
    pregame = {"event": "game-b", "phase": "pregame",
               "start_utc": (NOW + timedelta(minutes=10)).isoformat()}
    started = {**pregame, "event": "started", "start_utc": NOW.isoformat()}
    row = {"event": game["event"], "phase": "live", "quote_utc": NOW.isoformat(),
           "screened_ev": 0.15, "probability": 0.4, "token_id": "1"}
    ranked, picks = rank_opportunities([
        row,
        {**row, "token_id": "2", "screened_ev": 0.12, "probability": 0.6},
        {**row, "token_id": "3", "screened_ev": 0.25, "quote_utc": (NOW - timedelta(seconds=36)).isoformat()},
        {**row, "event": pregame["event"], "phase": "pregame", "token_id": "4"},
        {**row, "event": started["event"], "phase": "pregame", "token_id": "5"},
        {**row, "token_id": "6", "screened_ev": 0},
    ], [game, pregame, started], NOW)
    assert [entry["token_id"] for entry in ranked] == ["1", "4", "2"]
    assert picks == {"game-a": "1", "game-b": "4"}
