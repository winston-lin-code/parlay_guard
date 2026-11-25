from datetime import datetime, timedelta, timezone

import pytest

from src import college_inplay_scan, nfl_inplay_scan
from src.college_inplay_model import load, score_clock


@pytest.mark.parametrize(("sport", "period", "clock", "remaining", "markets"), [
    ("mbb", 1, "10:00", 1800, {"moneyline", "spreads", "totals"}),
    ("mbb", 2, "10:00", 600, {"spreads", "totals"}),
    ("cfb", 1, "10:00", 3300, {"moneyline"}),
    ("cfb", 2, "10:00", 2400, {"moneyline", "totals"}),
    ("cfb", 3, "10:00", 1500, {"totals"}),
    ("cfb", 4, "10:00", 600, {"totals"}),
])
def test_only_validated_periods_and_markets(sport, period, clock, remaining, markets):
    artifact = load(sport)
    assert artifact["games"][-1] >= 750
    start = "2026-11-25T20:00:00+00:00"
    game = {"start_utc": start, "score_text": "15-15", "clock_text": clock, "period_text": f"Q{period}",
            "home": {"id": "1", "location": "Home Team"}, "away": {"id": "2", "location": "Away Team"}}
    state = score_clock(game, sport)
    assert state == (15, 15, remaining, period)
    common = {"feesEnabled": False, "clobTokenIds": ["1", "2"], "outcomes": ["Home Team", "Away Team"]}
    line = {1800: 135.5, 600: 65.5}.get(remaining, 52.5) if sport == "mbb" else 52.5
    fixture = {"start_utc": start, "college_markets": [
        {**common, "sportsMarketType": "moneyline", "question": "Home Team vs Away Team"},
        {**common, "sportsMarketType": "spreads", "question": "Spread: Home Team (-1.5)", "line": -1.5},
        {**common, "sportsMarketType": "totals", "question": f"Home Team vs Away Team: O/U {line}",
         "outcomes": ["Over", "Under"], "line": line},
    ]}
    result = list(college_inplay_scan.candidates(fixture, game, state, artifact))
    assert {market["sportsMarketType"] for market, _, _ in result} == markets
    assert all(sum(leg[2] for leg in sides) == pytest.approx(1) for _, sides, _ in result)


def test_stale_score_transition_and_late_game(monkeypatch):
    now = datetime.now(timezone.utc)
    start = (now - timedelta(hours=1)).isoformat()
    fixture = {"event": "mbb-home-away", "title": "Home Team vs Away Team", "start_utc": start, "college": "mbb",
               "college_markets": [
                   {"sportsMarketType": "moneyline", "question": "Home Team vs Away Team",
                    "outcomes": ["Home Team", "Away Team"], "clobTokenIds": ["1", "2"], "feesEnabled": False},
               ]}
    game = {"espn_id": "matched", "start_utc": start, "period_text": "Q1", "clock_text": "12:00",
            "score_text": "30-21", "home": {"id": "1", "location": "Home Team", "abbr": "HOM"},
            "away": {"id": "2", "location": "Away Team", "abbr": "AWY"}}
    sports = {"observed_utc": now.isoformat(), "games": [fixture]}
    monkeypatch.setattr(college_inplay_scan, "scoreboard", lambda *_: [game])
    monkeypatch.setattr(nfl_inplay_scan, "_books", lambda tokens: {token: {
        "timestamp": now.timestamp()*1000, "min_order_size": 1, "asks": [{"price": .2, "size": 200}]
    } for token in tokens})
    result = college_inplay_scan.scan_college_live(sports, "mbb")
    assert result["games"][0]["score_text"] == "21–30"
    assert result["opportunities"]
    assert all(not row["half_payout_tie"] for row in result["opportunities"])
    updates = iter([[game], [{**game, "score_text": "33-21"}]])
    monkeypatch.setattr(college_inplay_scan, "scoreboard", lambda *_: next(updates))
    assert not college_inplay_scan.scan_college_live(sports, "mbb")["opportunities"]
    assert score_clock({**game, "period_text": "Q2", "clock_text": "01:59"}, "mbb") is None
    assert score_clock({**game, "period_text": "Q3"}, "mbb") is None
