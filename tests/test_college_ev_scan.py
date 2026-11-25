from datetime import datetime, timedelta, timezone

from src.college_ev_scan import candidates_for_fixture, match_fixture
from src.college_score_calibration import load_score_artifact, predict_game
from src.inplay_api import ScanState


def test_artifacts_cover_real_heldout_seasons_and_consistent_probabilities():
    for sport, count in (("mbb", 5299), ("cfb", 587)):
        bundle = load_score_artifact(sport)
        assert bundle["holdout_games"] == count
        assert bundle["holdout_season"] > bundle["calibration_season"]
        season = bundle["holdout_season"] + 1
        teams = [key for key, state in bundle["teams"].items() if state["season"] == season and state["games"] >= 3]
        game = {"sport": sport, "home": {"id": teams[0]}, "away": {"id": teams[1]},
                "start_utc": "2026-11-26T01:00:00+00:00", "season": season, "neutral": False}
        home_probability, margin, total = predict_game(game, bundle)
        assert 1 - margin.cdf(-3.5) >= home_probability >= 1 - margin.cdf(3.5)
        assert total.cdf(total.mean - 2) <= total.cdf(total.mean + 2)


def test_fixture_requires_two_teams_and_holds_failed_validation_out():
    bundle = load_score_artifact("mbb")
    season = bundle["holdout_season"] + 1
    teams = [key for key, state in bundle["teams"].items() if state["season"] == season and state["games"] >= 3]
    tip = "2026-11-26T01:00:00+00:00"
    home, away = {"id": teams[0], "abbr": "ABCD"}, {"id": teams[1], "abbr": "EFGH"}
    game = {"espn_id": "1", "sport": "mbb", "home": home, "away": away,
            "start_utc": tip, "season": season, "neutral": False}
    fields = {"outcomes": ["ABCD", "EFGH"], "clobTokenIds": ["1", "2"], "feesEnabled": False}
    fixture = {"start_utc": tip, "college_markets": [
        {**fields, "sportsMarketType": "moneyline", "question": "ABCD vs. EFGH"},
        {**fields, "sportsMarketType": "spreads", "question": "Spread: ABCD (-3.5)", "line": -3.5},
        {**fields, "sportsMarketType": "totals", "question": "ABCD vs. EFGH: O/U 145.5", "line": 145.5,
         "outcomes": ["Over", "Under"]},
    ]}
    assert match_fixture(fixture, [game]) == game
    assert match_fixture(fixture, [game, {**game, "espn_id": "2"}]) is None
    candidates = candidates_for_fixture(fixture, game, bundle)
    assert [market["sportsMarketType"] for market, _, _ in candidates] == ["spreads", "totals"]


def test_nba_refresh_does_not_expire_valid_college_quotes():
    state = ScanState()
    now = datetime.now(timezone.utc)
    state.snapshot["updated_utc"] = (now - timedelta(minutes=3)).isoformat()
    state.college = {
        "updated_utc": now.isoformat(), "games": [{"event": "cbb-game", "priced_sides": 1, "modeled_markets": 1,
                                                    "phase": "pregame", "start_utc": (now + timedelta(hours=1)).isoformat()}],
        "warnings": [], "opportunities": [{"event": "cbb-game", "quote_utc": now.isoformat(), "screened_ev": 0.08,
                                             "token_id": "123", "phase": "pregame", "probability": 0.55}],
    }
    state.sports["games"] = [{"event": "cbb-game", "college_markets": [{"id": "123"}]}]
    snap = state.read()
    assert not snap["stale"]
    assert snap["top_picks"] == {"cbb-game": "123"}
    assert snap["opportunities"][0]["event"] == "cbb-game"
    assert "college_markets" not in snap["sports"]["games"][0]
