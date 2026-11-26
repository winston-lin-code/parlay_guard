from datetime import datetime, timedelta, timezone

import pytest

from src.game_board_feed import parse_score
from src.inplay_api import ScanState
from src.tennis_live import live_win_probability


@pytest.mark.parametrize(("score", "period", "doubles", "expected"), [
    ("4-6, 0-0:0-0", "S2", False, 0.25),
    ("6-3, 0-0:0-0", "S2", False, 0.75),
    ("6-6:3-4", "TB1", False, 0.34375),
    ("6-3, 6-5:0-0", "S2", False, 0.875),
    ("6-3, 3-6, 0-0:3-4", "TB3", True, 0.38720703125),
    ("0-0:40-AD", "S1", False, 0.4384765625),
    ("4-3:40-40", "S1", True, 0.6875),
    ("6-3, 6-0:0-0", "S2", False, None),
    ("6-3, 6-3, 0-0:0-0", "S3", False, None),
    ("6-6:7-5", "TB1", False, None),
    ("3-3:AD-AD", "S1", False, None),
])
def test_itf_score_states(score, period, doubles, expected):
    assert live_win_probability(score, period, doubles=doubles) == expected


def test_scanner_projects_fresh_itf_catalog_and_socket_scores():
    now = datetime.now(timezone.utc)
    state = ScanState()
    state.sports = {"observed_utc": now.isoformat(), "games": [{
        "event": "itfwo-test", "league": "ITFWO", "title": "One vs. Two",
        "phase": "live", "score_text": "6-3, 0-0", "period_text": "S2", "sports_game_id": 42,
    }]}
    assert state.read()["board"][0]["tennis_projection"]["first_probability"] == 0.75
    state.sports["observed_utc"] = (now - timedelta(minutes=4)).isoformat()
    assert state.read()["board"] == []

    update = {"gameId": 42, "live": True, "score": "3-6, 0-0:0-0", "period": "S2"}
    state.update_score(*parse_score(update))
    assert state.read()["board"][0]["tennis_projection"]["first_probability"] == 0.25
    state.update_score(*parse_score({**update, "ended": True}))
    assert state.read()["board"][0]["tennis_projection"] is None
