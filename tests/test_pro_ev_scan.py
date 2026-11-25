import pytest

from src.college_ev_scan import candidates_for_fixture, match_fixture
from src.pro_score_calibration import load, predict


@pytest.mark.parametrize("sport,enabled", [
    ("nfl", {"moneyline", "spreads", "totals"}),
    ("nhl", {"moneyline", "spreads"}),
    ("mlb", {"moneyline", "spreads", "totals"}),
])
def test_trained_artifacts_screen_only_validated_markets(sport, enabled):
    model = load(sport)
    assert model["holdout_games"] >= 272
    home_id, away_id = tuple(model["teams"])[:2]
    start = "2027-01-01T17:00:00+00:00"
    game = {"espn_id": "known-fixture", "start_utc": start, "day": start[:10], "neutral": False,
            "home": {"id": home_id, "name": "Home mascot"},
            "away": {"id": away_id, "name": "Away mascot"}}
    base = {"outcomes": ["Away mascot", "Home mascot"], "clobTokenIds": ["1", "2"],
            "feesEnabled": False}
    moneyline = {**base, "sportsMarketType": "moneyline", "question": "Away mascot vs. Home mascot",
                 "description": "If the game ends in a tie, this market will resolve 50-50."}
    fixture = {"pro": sport, "start_utc": start, "college_markets": [
        moneyline,
        {**base, "sportsMarketType": "spreads", "line": -1.5, "question": "Spread: Away mascot (-1.5)"},
        {**base, "sportsMarketType": "totals", "line": 6.5, "outcomes": ["Over", "Under"],
         "question": "Away mascot vs. Home mascot: O/U 6.5"},
    ]}
    assert match_fixture(fixture, [game]) == game
    assert match_fixture(fixture, [game, {**game, "espn_id": "ambiguous"}]) is None
    candidates = {contract["sportsMarketType"]: sides for contract, sides, _ in
                  candidates_for_fixture(fixture, game, model)}
    assert set(candidates) == enabled
    assert all(sum(side[2] for side in sides) == pytest.approx(1) for sides in candidates.values())
    if sport == "nfl":
        away_win = 1 - predict(game, model)[0]
        tie = model["tie_rate"]
        assert candidates["moneyline"][0][2] == pytest.approx(away_win * (1 - tie) + .5 * tie)
        moneyline["description"] = "Unverified tie payout"
        assert all(market["sportsMarketType"] != "moneyline" for market, _, _ in
                   candidates_for_fixture(fixture, game, model))
