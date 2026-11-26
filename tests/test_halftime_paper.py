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
    monkeypatch.setattr(paper, "live_inventory", lambda: ({}, []))
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


def test_mens_college_halftime_is_after_first_half():
    fixture = event()
    status = fixture["competitions"][0]["status"]
    status.update(period=1, type={"state": "in", "shortDetail": "End of 1st Half"})
    assert paper.game_state("CBB", fixture)
    assert paper.game_state("WCBB", fixture) is None
    status.update(period=2, type={"state": "in", "name": "STATUS_END_PERIOD"})
    assert paper.game_state("CBB", fixture) is None
    assert paper.game_state("WCBB", fixture)


@pytest.mark.parametrize(("league", "period", "slug", "scope", "rules"), [
    ("CBB", 1, "cbb-pho-hou", "basketball_team_full_game_winner", "Overtime is included if played."),
    ("WCBB", 2, "wcbb-pho-hou", "basketball_team_full_game_winner", "Overtime is included if played."),
    ("CFB", 2, "cfb-pho-hou", "football_team_full_game_winner", "Overtime is included if played."),
    ("WNBA", 2, "wnba-pho-hou", "moneyline", "Official winner of the Phoenix vs Houston WNBA game. Overtime is included."),
])
def test_extended_league_moneylines_use_both_team_identities(monkeypatch, clock, league, period, slug, scope, rules):
    fixture = event()
    fixture["competitions"][0]["status"]["period"] = period
    slug += "-2026-11-25"
    contract = {**market(), "slug": f"aec-{slug}", "sportsMarketType": scope, "description": rules}
    game = paper.game_state(league, fixture)
    game["listed_moneylines"] = [(slug, contract)]
    if league == "WCBB":
        contract["marketSides"][0]["team"] = {"safeName": "Suns (W)"}
    monkeypatch.setattr(paper, "home_probability", lambda *args: .70)
    book = {**order_book(), "marketSlug": f"aec-{slug}"}
    monkeypatch.setattr(paper, "public", lambda path, **kwargs: {"marketData": book})
    result = paper.price_game(game)
    assert result["league"] == league
    assert result["side"] == "Rockets"
    assert f"/{league.lower()}/{slug}?" in result["url"]

    contract["marketSides"][0]["team"] = {"safeName": "Some other team"}
    assert paper.price_game(game) is None


def test_signed_phase_and_catalog_require_a_unique_match():
    fixture = event()
    fixture["competitions"][0]["status"]["period"] = 1
    selected = paper.game_state("CBB", fixture)
    slug = "cbb-different-ticker-2026-11-25"
    contract = {**market(), "slug": f"aec-{slug}"}
    other_slug = "cbb-other-ticker-2026-11-25"
    selected["listed_moneylines"] = [(slug, contract), (other_slug, {**contract, "slug": f"aec-{other_slug}"})]
    assert paper.contract_and_sides(selected) is None


def test_basketball_models_use_held_out_games_and_favor_the_leading_team():
    models = paper.load_models()
    assert set(models) == {"CBB", "WCBB", "WNBA"}
    for model in models.values():
        holdout = model["holdout"]
        assert holdout["games"] >= 200
        assert holdout["brier"] < holdout["baseline_brier"]
        assert holdout["log_loss"] < holdout["baseline_log_loss"]
        assert paper.win_probability(model, 50, 40, True) > .60
        assert paper.win_probability(model, 40, 50, True) < .40


def test_discovery_surfaces_nonmodeled_sports_without_using_their_markets(monkeypatch):
    def listed(slug, title, teams):
        return {"slug": slug, "title": title, "teams": teams, "live": True,
                "markets": [{"slug": f"aec-{slug}"}]}

    league = listed("wcbb-sd-dxst-2026-11-25", "San Diego vs. Utah Tech", [{}, {}])
    other = listed("nhl-vgk-chi-2026-11-25", "Vegas vs. Chicago", [{}, {}])
    unrelated = listed("btc-updown-5m-123", "Bitcoin Up or Down", [])
    monkeypatch.setattr(paper, "read_json", lambda *args, **kwargs: {"events": [league, other, unrelated]})
    listings, games = paper.live_inventory()
    assert listings["WCBB"] == [(league["slug"], league["markets"][0])]
    assert {game["league"] for game in games} == {"WCBB", "NHL"}
    assert "NHL" not in listings


@pytest.mark.parametrize(("league", "quarter_length"), [("NBA", 720), ("NFL", 900)])
def test_five_minute_halftime_window_boundaries(league, quarter_length):
    fixture = event()
    status = fixture["competitions"][0]["status"]
    status["type"] = {"state": "in", "name": "STATUS_IN_PROGRESS"}
    for period, clock_value, expected in (
        (2, 301, None), (2, 300, "Before halftime"), (2, 0, None),
        (3, quarter_length, "After halftime"), (3, quarter_length - 300, "After halftime"),
        (3, quarter_length - 301, None), (1, 15, None), (4, 15, None),
    ):
        status.update(period=period, clock=clock_value)
        game = paper.game_state(league, fixture)
        if expected is None:
            assert game is None
        else:
            assert game["checkpoint"] == expected
            assert game["remaining"] == (4 - period) * quarter_length + clock_value
    status.update(period=2, clock=300)
    assert paper.game_state("WNBA", fixture) is None


def test_nfl_window_model_uses_current_clock(monkeypatch):
    fixture = event()
    status = fixture["competitions"][0]["status"]
    status.update(period=3, clock=840, type={"state": "in", "name": "STATUS_IN_PROGRESS"})
    monkeypatch.setattr(paper, "load_nfl", lambda: {})
    def forecast(state, artifact):
        assert state == (67, 60, 1740)
        return (.7, None, None), None
    monkeypatch.setattr(paper, "nfl_forecast", forecast)
    assert paper.home_probability(paper.game_state("NFL", fixture), NOW) == .7
