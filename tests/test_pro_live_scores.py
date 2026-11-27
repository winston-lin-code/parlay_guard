from datetime import timedelta

from src import pro_live_scores as verifier


def listing(league, away, home, raw_score, period, fields=None):
    slug = f"{league}-{away.lower()}-{home.lower()}"
    sport = verifier.SPORTS[league]
    return {
        "slug": slug, "startTime": "2026-09-24T16:35:00Z", "live": True,
        "markets": [{"marketType": "moneyline", "sportsMarketType": f"{sport}_team_full_game_winner",
                     "marketSides": [{"team": {"ordering": label, "abbreviation": club}}
                                     for label, club in [("home", home), ("away", away)]]}],
        "eventState": {"type": sport, "score": raw_score, "period": period, "live": True,
                       "updatedAt": "2026-09-24T17:00:05Z", **(fields or {})},
    }


def test_nhl_detects_away_home_score_and_explicit_intermission():
    pm = listing("nhl", "WPG", "BOS", "4-2", "END P2")
    game = {
        "awayTeam": {"abbrev": "WPG", "score": 4}, "homeTeam": {"abbrev": "BOS", "score": 2},
        "periodDescriptor": {"number": 2}, "clock": {"inIntermission": True, "timeRemaining": "00:00"},
        "gameState": "LIVE",
    }
    official = verifier.nhl_state(game)
    assert verifier.agreement(official, verifier.market_state("nhl", pm)) == "score_and_phase_agree"
    assert verifier.received_state(official, verifier.now())["state_age_seconds"] is None
    pm["eventState"]["period"] = "P2"
    assert verifier.agreement(official, verifier.market_state("nhl", pm)) == "phase_mismatch"
    pm["eventState"]["score"] = "2-4"
    assert verifier.agreement(official, verifier.market_state("nhl", pm)) == "score_mismatch"


def test_mlb_compares_inning_half_and_outs_and_preserves_source_clocks():
    event = listing("mlb", "STL", "PIT", "1-2", "B5", {"baseballState": {"outs": 1}})
    game = {
        "gameData": {"status": {"abstractGameState": "Live", "detailedState": "In Progress"}},
        "metaData": {"timeStamp": "20260924_170006"},
        "liveData": {"linescore": {"inningState": "Bottom", "currentInning": 5, "outs": 1,
                                  "teams": {"away": {"runs": 1}, "home": {"runs": 2}}},
                     "plays": {"allPlays": [{"about": {"isComplete": True, "endTime": "2026-09-24T17:00:04Z"}}]}},
    }
    official = verifier.mlb_state(game)
    market = verifier.market_state("mlb", event)
    assert verifier.agreement(official, market) == "score_and_phase_agree"
    assert verifier.received_state(official, verifier.utc("2026-09-24T17:00:10Z"))["state_age_seconds"] == 4
    assert official["last_completed_play_at"] == "2026-09-24T17:00:04Z"
    assert verifier.agreement(official, {**market, "period": 4}) == "period_mismatch"
    assert verifier.agreement(official, {**market, "outs": 2}) == "outs_mismatch"
    assert verifier.agreement(official, {**market, "phase": "unavailable"}) == "score_agrees_phase_unverified"


def test_doubleheaders_and_reversed_teams_do_not_guess_a_fixture(monkeypatch):
    event = listing("mlb", "STL", "PIT", "1-2", "B5", {"baseballState": {"outs": 1}})
    start = verifier.utc(event["startTime"])
    received = start + timedelta(minutes=25)
    teams = {"away": {"team": {"abbreviation": "STL"}}, "home": {"team": {"abbreviation": "PIT"}}}
    rows = [(123, start, {"teams": teams}, received),
            (456, start + timedelta(minutes=15), {"teams": teams}, received),
            (789, start, {"teams": {side: teams[key] for side, key in [("away", "home"), ("home", "away")]}}, received)]
    monkeypatch.setattr(verifier, "official_fixtures", lambda *_: rows)
    result = verifier.assess("mlb", {event["slug"]: (event, received)})["games"][0]
    assert result["status"] == "unmatched_fixture"
    assert result["matching_official_games"] == 2
