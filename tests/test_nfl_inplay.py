from datetime import datetime, timedelta, timezone

from src import nfl_inplay_scan
from src.nfl_inplay_model import load, score_clock


def test_regulation_clock_and_late_game_cutoff():
    start = {"score_text": "14-10", "period_text": "Q3", "clock_text": "10:30"}
    assert score_clock(start) == (14, 10, 1530)
    assert score_clock({**start, "period_text": "HT"}) == (14, 10, 1800)
    assert score_clock({**start, "period_text": "Q4", "clock_text": "01:59"}) is None
    assert score_clock({**start, "period_text": "Q5"}) is None
    assert score_clock({**start, "clock_text": "15:01"}) is None


def test_live_reprice_and_drop_when_score_changes(monkeypatch):
    now = datetime.now(timezone.utc)
    start = (now - timedelta(hours=1)).isoformat()
    fixture = {"event": "nfl-chi-det", "title": "Bears vs. Lions", "start_utc": start, "league": "NFL",
               "pro": "nfl", "college_markets": [
                   {"sportsMarketType": "moneyline", "question": "Bears vs. Lions",
                    "description": "If the game ends in a tie, this market will resolve 50-50.",
                    "outcomes": ["Bears", "Lions"], "clobTokenIds": ["1", "2"], "feesEnabled": False},
                   {"sportsMarketType": "spreads", "question": "Spread: Bears (-3.5)", "line": -3.5,
                    "outcomes": ["Bears", "Lions"], "clobTokenIds": ["3", "4"], "feesEnabled": False},
                   {"sportsMarketType": "totals", "question": "Bears vs. Lions: O/U 40.5", "line": 40.5,
                    "outcomes": ["Over", "Under"], "clobTokenIds": ["5", "6"], "feesEnabled": False},
               ]}
    game = {"espn_id": "verified", "start_utc": start, "period_text": "Q3", "clock_text": "10:00",
            "score_text": "27-13", "home": {"id": "1", "name": "Lions", "abbr": "DET"},
            "away": {"id": "2", "name": "Bears", "abbr": "CHI"}}
    monkeypatch.setattr(nfl_inplay_scan, "scoreboard", lambda *_: [game])
    monkeypatch.setattr(nfl_inplay_scan, "_books", lambda tokens: {token: {
        "timestamp": now.timestamp()*1000, "min_order_size": 1, "asks": [{"price": .1, "size": 200}]
    } for token in tokens})
    sports = {"observed_utc": now.isoformat(), "games": [fixture]}
    result = nfl_inplay_scan.scan_nfl(sports)
    assert load()["holdout_games"] == 272
    assert result["games"][0]["live_model"]
    assert result["opportunities"]
    assert all("Spread" not in row["market"] for row in result["opportunities"])
    updates = iter([[game], [{**game, "score_text": "34-13"}]])
    monkeypatch.setattr(nfl_inplay_scan, "scoreboard", lambda *_: next(updates))
    assert nfl_inplay_scan.scan_nfl(sports)["opportunities"] == []
