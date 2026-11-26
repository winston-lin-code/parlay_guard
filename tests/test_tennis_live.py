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
    assert state.read()["board"][0]["tennis_projection"]["first_probability"] > 0.75
    state.sports["observed_utc"] = (now - timedelta(minutes=4)).isoformat()
    assert state.read()["board"] == []

    update = {"gameId": 42, "live": True, "score": "3-6, 0-0:0-0", "period": "S2"}
    state.update_score(*parse_score(update))
    assert state.read()["board"][0]["tennis_projection"]["first_probability"] < 0.25
    state.update_score(*parse_score({**update, "ended": True}))
    assert state.read()["board"][0]["tennis_projection"] is None


@pytest.mark.parametrize("league", ("ITFME", "ITFWO"))
def test_trained_priors_match_artifact_at_set_break(league):
    from math import exp

    from src.tennis_live import load_models, tennis_projection

    model = load_models()[league]
    game = {"event": f"{league.lower()}-test", "title": "One vs. Two"}
    score = {"phase": "live", "score_text": "6-2, 0-0:0-0", "period_text": "S2"}
    result = tennis_projection(game, score)
    weight = model["set_breaks"]["1"]["weights"][0]
    assert result["historical_prior"]
    assert result["first_probability"] == pytest.approx(1 / (1 + exp(-4 * weight)))
    reverse = tennis_projection(game, {**score, "score_text": "2-6, 0-0:0-0"})
    assert reverse["first_probability"] == pytest.approx(result["second_probability"])
    doubles = tennis_projection({**game, "title": "A - B vs. C - D"}, score)
    assert not doubles["historical_prior"]
    assert doubles["first_probability"] == 0.75


def test_halftime_inventory_rows_include_live_tennis_model():
    now = datetime.now(timezone.utc)
    state = ScanState()
    game = {"event": "itfme-test", "title": "One vs. Two", "phase": "live",
            "score_text": "7-6(7-4), 0-0:0-0", "period_text": "S2"}
    state.halftime = {"updated_utc": now.isoformat(), "picks": [], "warnings": [], "live_games": [game]}
    row, = state.read()["halftime_paper"]["live_games"]
    assert row["tennis_projection"]["historical_prior"]
    assert row["tennis_projection"]["first_probability"] > .5
    assert "tennis_projection" not in game


def test_observed_first_set_break_expires_and_survives_score_changes():
    from src.inplay_scan import utc
    from src.tennis_live import first_set_checkpoint, observe_first_set_break

    now = datetime.now(timezone.utc)
    first = {"phase": "live", "period_text": "S1", "score_text": "5-4:40-0",
             "score_utc": (now - timedelta(seconds=30)).isoformat()}
    second = {"phase": "live", "period_text": "S2", "score_text": "6-4, 0-0:0-0", "score_utc": now.isoformat()}
    assert first_set_checkpoint(first, now) == "Late first set · one game from set"
    assert observe_first_set_break({}, second, utc) is None
    second["first_set_break_utc"] = observe_first_set_break(first, second, utc)
    assert second["first_set_break_utc"] == now.isoformat()
    assert observe_first_set_break(second, {**second, "score_text": "6-4, 0-1:0-0"}, utc) == now.isoformat()
    assert first_set_checkpoint(second, now + timedelta(minutes=5))
    assert first_set_checkpoint(second, now + timedelta(minutes=5, seconds=1)) is None


def test_elo_prior_uses_identified_score_column_order():
    from src.tennis_live import tennis_projection

    game = {"event": "itfme-test", "title": "Robert Strombachs vs. Yash Yadav"}
    score = {"phase": "live", "period_text": "S2", "score_text": "6-4, 0-0:0-0"}
    forward = tennis_projection(game, {**score, "score_order": "Robert Strombachs / Yash Yadav"})
    reverse = tennis_projection(game, {**score, "score_text": "4-6, 0-0:0-0",
                                        "score_order": "Yash Yadav / Robert Strombachs"})
    assert forward["elo_prior"] and reverse["elo_prior"]
    assert forward["first_probability"] == pytest.approx(reverse["second_probability"])
    assert not tennis_projection(game, score)["elo_prior"]
