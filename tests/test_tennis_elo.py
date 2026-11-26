from datetime import datetime, timedelta

import pytest

from src.tennis_elo import predict
from src.train_tennis_elo import Match, replay


def test_tournament_results_do_not_change_predictions_until_lag_expires():
    start = datetime(2025, 1, 1)

    def match(day):
        return Match(start + timedelta(days=day), "Clay", "1", "2", "Player One", "Player Two")

    metrics, _ = replay([match(0), match(0), match(13)], 32, .5, [2025])
    assert metrics["accuracy"] == .5
    assert metrics["both_players_rated"] == 0

    metrics, _ = replay([match(0), match(0), match(13), match(14)], 32, .5, [2025])
    assert metrics["accuracy"] == .625
    assert metrics["both_players_rated"] == 1


def test_predict_uses_surface_and_resolves_players_conservatively():
    model = {"surface_weight": .5, "last_tournament_start": "2025-01-01", "players": {
        "1": {"name": "Player One", "overall": 1500, "surfaces": {"Clay": 1700}, "matches": 10},
        "2": {"name": "Player Two", "overall": 1500, "surfaces": {}, "matches": 8},
    }}
    assert predict(model, " Player ONE ", "2")["players"][0]["win_probability"] == .5
    assert predict(model, "1", "2", "Clay")["players"][0]["elo"] == 1600
    assert predict(model, "1", "2", "Clay")["players"][0]["win_probability"] > .5
    with pytest.raises(ValueError, match="missing from the ratings"):
        predict(model, "Unknown", "2")
    model["players"]["3"] = model["players"]["1"]
    with pytest.raises(ValueError, match="ambiguous"):
        predict(model, "Player One", "2")
