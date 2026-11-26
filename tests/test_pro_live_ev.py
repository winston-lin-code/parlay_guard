from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pytest

from src import pro_inplay_model as model, pro_live_ev as ev

NOW = datetime(2026, 7, 1, 18, tzinfo=timezone.utc)
SLUG = "nhl-wpg-bos-2026-07-01"


def test_mlb_forecast_uses_base_out_state_and_rejects_unknown_inputs():
    state = {"score_away_home": [2, 2], "phase": "bottom", "outs": 1, "bases": [False, False, True],
             "period": 5, "scheduled_innings": 9}
    artifact = model.load("mlb")
    with_runner = model.forecast(model.model_state("mlb", state), artifact)
    empty = model.forecast(model.model_state("mlb", {**state, "bases": [False] * 3}), artifact)
    assert 0 < empty < with_runner < 1
    assert model.model_state("mlb", {**state, "bases": None}) is None
    assert model.model_state("mlb", {**state, "period": 10}) is None
    assert model.model_state("mlb", {**state, "phase": "end", "bases": None, "outs": None})


def nhl_scenario():
    official = {"score_away_home": [1, 3], "phase": "in_play", "period": 2,
                "period_type": "REG", "seconds_remaining": 28,
                "received_at": (NOW - timedelta(seconds=2)).isoformat()}
    event = {"slug": SLUG, "markets": [{
        "slug": f"aec-{SLUG}", "active": True, "closed": False, "marketType": "moneyline",
        "sportsMarketType": "hockey_team_full_game_winner",
        "description": "Overtime and any shootout are included if played.", "question": "Match winner",
        "minimumTradeQty": "1", "orderPriceMinTickSize": ".01", "feeCoefficient": ".11",
        "marketSides": [{"long": True, "tradable": True, "team": {"ordering": "away", "name": "Jets"}},
                        {"long": False, "tradable": True, "team": {"ordering": "home", "name": "Bruins"}}],
    }], "eventState": {"type": "hockey", "live": True, "period": "P2", "score": "1-3"}}
    market = {**ev.market_state("nhl", event), "received_at": official["received_at"]}
    verified = {"official_game_id": 1, "status": "score_and_phase_agree", "official": official,
                "polymarket_us": market}
    return official, event, market, verified


def test_live_pricing_reuses_executable_depth_and_holds_on_state_change(monkeypatch):
    official, event, market, verified = nhl_scenario()
    book = {"marketSlug": f"aec-{SLUG}", "state": "MARKET_STATE_OPEN",
            "transactTime": (NOW - timedelta(seconds=1)).isoformat(),
            "bids": [{"px": {"value": ".50", "currency": "USD"}, "qty": "30"}],
            "offers": [{"px": {"value": ".52", "currency": "USD"}, "qty": "30"}]}
    monkeypatch.setattr(ev, "now", lambda: NOW)
    monkeypatch.setattr(ev, "current_official", lambda *_: official)
    monkeypatch.setattr(ev, "current_market", lambda *_: (event, market))
    monkeypatch.setattr(ev, "public", lambda path, **kwargs: {"marketData": book})
    artifact = model.load("nhl")
    result = ev.evaluate("nhl", event, verified, artifact, Decimal(2), Decimal(".01"))
    assert result["pricing_status"] == "priced"
    assert sum(result["win_probability"].values()) == pytest.approx(1)
    for quote in result["quotes"].values():
        cost = Decimal(quote["total_estimate"])
        assert cost <= 2
        assert cost == (Decimal(quote["limit"]) + Decimal(".01")) * quote["quantity"] + Decimal(quote["fee_estimate"])
        assert quote["estimated_net_roi"] == pytest.approx(quote["probability"] * quote["quantity"] / float(cost) - 1)
        assert quote["screened_ev"] <= quote["estimated_net_roi"]

    book["transactTime"] = (NOW - timedelta(seconds=3)).isoformat()
    assert ev.evaluate("nhl", event, verified, artifact, Decimal(2), Decimal(".01"))[
        "pricing_status"
    ] == "no_fresh_post_score_book"
    monkeypatch.setattr(ev, "current_official", lambda *_: {**official, "score_away_home": [2, 3]})
    assert ev.evaluate("nhl", event, verified, artifact, Decimal(2), Decimal(".01"))[
        "pricing_status"
    ] == "game_state_changed_during_quote"


def test_idle_scan_reports_next_market_in_eastern_time(monkeypatch):
    upcoming = {"title": "Jets vs. Bruins", "active": True, "startTime": "2026-07-01T22:00:00Z"}
    monkeypatch.setattr(ev, "league_listings", lambda *_: [
        (SLUG, upcoming, NOW),
        ("nhl-ended", {**upcoming, "ended": True, "startTime": "2026-07-01T21:00:00Z"}, NOW),
        ("nhl-later", {**upcoming, "startTime": "2026-07-01T23:00:00Z"}, NOW),
    ])
    result = ev.scan("nhl", Decimal(2), Decimal(".01"))
    assert result["status"] == "no_live_listings"
    assert result["upcoming_listings"] == 2
    assert result["next_start_et"] == "2026-07-01T18:00:00-04:00"
    assert result["next_games"] == [{"event": SLUG, "title": upcoming["title"]}]
    assert ev.should_report(result, None, 0)
    assert not ev.should_report(result, result, 59)
    assert ev.should_report(result, result, 60)
    assert ev.should_report({"league": "nhl", "games": [SLUG]}, result, 1)
