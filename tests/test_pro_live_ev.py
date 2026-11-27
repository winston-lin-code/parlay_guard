from contextlib import nullcontext
from datetime import datetime, timedelta, timezone
from decimal import Decimal
import sqlite3
from types import SimpleNamespace

import pytest

from src import pro_inplay_model as model, pro_live_ev as ev
from src import pro_halftime_orders as orders

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


def test_halftime_paper_logs_one_qualifying_bet_per_game(tmp_path, monkeypatch):
    monkeypatch.setattr(ev, "now", lambda: NOW)
    path = tmp_path / "paper.sqlite3"
    quote = {"probability": .65, "screened_ev": .03, "total_estimate": "9.75",
             "quote_utc": (NOW - timedelta(seconds=1)).isoformat()}
    official = {"score_away_home": [1, 3], "phase": "intermission", "period": 2}
    game = {"event": SLUG, "league": "nhl", "official_game_id": 1, "pricing_status": "priced",
            "official": official, "quotes": {"home": quote}}
    for invalid in ({"probability": .60}, {"screened_ev": 0}, {"screened_ev": .05},
                    {"total_estimate": "10.01"}):
        result = ev.log_paper_entry(path, {**game, "quotes": {"home": {**quote, **invalid}}}, Decimal(10))
        assert result["status"] == "outside_probability_or_roi_band"
    assert ev.log_paper_entry(path, {**game, "official": {**official, "period": 1}}, Decimal(10))[
        "status"
    ] == "no_verified_halftime_quote"
    stale = {**quote, "quote_utc": (NOW - timedelta(seconds=6)).isoformat()}
    assert ev.log_paper_entry(path, {**game, "quotes": {"home": stale}}, Decimal(10))[
        "status"
    ] == "quote_expired_before_paper_entry"
    assert not path.exists()

    result = ev.log_paper_entry(path, game, Decimal(10))
    assert result["status"] == "paper_entry_logged"
    assert result["entry"]["outcome"] == "home"
    assert ev.log_paper_entry(path, game, Decimal(10))["status"] == "already_logged"
    assert ev.log_paper_entry(path, {**game, "event": f"{SLUG}-rescheduled"}, Decimal(10))[
        "status"
    ] == "already_logged"
    with sqlite3.connect(path) as db:
        assert db.execute("SELECT count(*) FROM pro_halftime_entries").fetchone() == (1,)
    assert ev.recent_paper_entries(path) == [result["entry"]]


def test_mlb_halftime_uses_end_of_fifth():
    assert ev.at_halftime("mlb", {"phase": "end", "period": 5})
    assert not ev.at_halftime("mlb", {"phase": "middle", "period": 5})


def test_any_time_paper_scanner_records_one_high_ev_entry_before_halftime(monkeypatch, tmp_path):
    official, event, market, verified = nhl_scenario()
    event["live"] = True
    verified.update(event=SLUG, league="nhl")
    book = {"marketSlug": f"aec-{SLUG}", "state": "MARKET_STATE_OPEN",
            "transactTime": (NOW - timedelta(seconds=1)).isoformat(),
            "bids": [{"px": {"value": ".32", "currency": "USD"}, "qty": "30"}],
            "offers": [{"px": {"value": ".34", "currency": "USD"}, "qty": "30"}]}
    monkeypatch.setattr(ev, "now", lambda: NOW)
    monkeypatch.setattr(ev.model, "forecast", lambda *_: .70)
    monkeypatch.setattr(ev, "league_listings", lambda *_: [(SLUG, event, NOW)])
    monkeypatch.setattr(ev, "assess", lambda *_: {"games": [verified]})
    monkeypatch.setattr(ev, "current_official", lambda *_: official)
    monkeypatch.setattr(ev, "current_market", lambda *_: (event, market))
    monkeypatch.setattr(ev, "public", lambda *_args, **_kwargs: {"marketData": book})

    path = tmp_path / "any-high.sqlite3"

    def scan():
        return ev.scan("nhl", Decimal(5), Decimal(".01"), paper_db=path, paper_any_high_ev=True)["games"][0]

    first = scan()
    assert first["any_high_ev_paper"]["status"] == "outside_probability_or_roi_band"
    assert not path.exists()
    book["bids"][0]["px"]["value"] = ".45"
    second = scan()
    entry, = ev.recent_paper_entries(path)
    assert second["any_high_ev_paper"]["status"] == "paper_entry_logged"
    assert entry["mode"] == "paper"
    assert entry["strategy"] == "any_high_ev"
    assert entry["checkpoint"] == "in_play"
    assert entry["quote"]["screened_ev"] >= .05
    assert Decimal(entry["quote"]["total_estimate"]) <= 5
    assert scan()["any_high_ev_paper"]["status"] == "already_logged"


@pytest.mark.parametrize("league,phase,period,winning_state,venue_period", [
    ("nhl", "intermission", 2, {"period_type": "REG", "seconds_remaining": 0}, "END P2"),
    ("mlb", "end", 5, {"scheduled_innings": 9, "bases": None, "outs": None}, "END 5"),
])
def test_live_scan_places_only_one_capped_automatic_limit(monkeypatch, tmp_path, league, phase, period,
                                                  winning_state, venue_period):
    official, event, _, verified = nhl_scenario()
    slug = SLUG if league == "nhl" else "mlb-bos-nyy-2026-07-01"
    official.update(winning_state, phase=phase, period=period)
    event.update(slug=slug, live=True)
    event["eventState"].update(type=ev.SPORTS[league], period=venue_period)
    contract = event["markets"][0]
    contract.update(slug=f"aec-{slug}", sportsMarketType=f"{ev.SPORTS[league]}_team_full_game_winner")
    if league == "mlb":
        contract["description"] = "Extra innings are included."
    market = {**ev.market_state(league, event), "received_at": official["received_at"]}
    verified.update(event=slug, league=league, official=official, polymarket_us=market)
    book = {"marketSlug": contract["slug"], "state": "MARKET_STATE_OPEN",
            "transactTime": (NOW - timedelta(seconds=1)).isoformat(),
            "bids": [{"px": {"value": ".45", "currency": "USD"}, "qty": "30"}],
            "offers": [{"px": {"value": ".46", "currency": "USD"}, "qty": "30"}]}
    monkeypatch.setattr(ev, "now", lambda: NOW)
    monkeypatch.setattr(orders, "now", lambda: NOW)
    monkeypatch.setattr(ev.model, "forecast", lambda *_: .65)
    monkeypatch.setattr(ev, "league_listings", lambda *_: [(slug, event, NOW)])
    monkeypatch.setattr(ev, "assess", lambda *_: {"games": [verified]})
    monkeypatch.setattr(ev, "current_official", lambda *_: official)
    monkeypatch.setattr(ev, "current_market", lambda *_: (event, market))
    monkeypatch.setattr(ev, "public", lambda *_args, **_kwargs: {"marketData": book})

    submissions = []

    def submit(payload):
        submissions.append(payload)
        return {"id": "test-order", "executions": [{"order": {"state": "ORDER_STATE_FILLED"}}]}

    def client():
        return nullcontext(SimpleNamespace(orders=SimpleNamespace(create=submit)))

    path = tmp_path / "live.sqlite3"
    ledger = orders.HalftimeOrders(path, client)
    result = ev.scan(league, orders.LIVE_STAKE, Decimal(".01"), live_orders=ledger)["games"][0]
    assert result["halftime_live"]["state"] == "ORDER_STATE_FILLED"
    assert "halftime_paper" not in result
    entry, = ledger.recent()
    assert entry["order_id"] == "test-order"
    assert entry["quote"]["quantity"] == 8
    assert Decimal(entry["quote"]["total_estimate"]) == Decimal("4.70")
    assert 0 < entry["quote"]["screened_ev"] < .05
    assert submissions == [{"marketSlug": contract["slug"], "type": "ORDER_TYPE_LIMIT",
                            "price": {"value": "0.45", "currency": "USD"}, "quantity": 8,
                            "tif": "TIME_IN_FORCE_FILL_OR_KILL", "intent": "ORDER_INTENT_BUY_SHORT",
                            "manualOrderIndicator": "MANUAL_ORDER_INDICATOR_AUTOMATIC", "synchronousExecution": True}]
    restarted = orders.HalftimeOrders(path, client)
    repeat = ev.scan(league, orders.LIVE_STAKE, Decimal(".01"), live_orders=restarted)["games"][0]
    assert repeat["halftime_live"]["status"] == "live_order_already_attempted"
    assert len(submissions) == 1


@pytest.mark.parametrize("retry_expired", [False, True])
def test_ambiguous_real_order_is_reserved_and_never_retried(monkeypatch, tmp_path, retry_expired):
    monkeypatch.setattr(orders, "now", lambda: NOW)
    entry = {"league": "nhl", "event": SLUG, "official_game_id": 1, "recorded_at": NOW.isoformat(),
             "quote": {"quote_utc": NOW.isoformat(), "total_estimate": "4.95"}}
    payload = {"marketSlug": f"aec-{SLUG}"}
    seen = []

    def submit(_):
        seen.append(True)
        raise TimeoutError

    def client():
        return nullcontext(SimpleNamespace(orders=SimpleNamespace(create=submit)))

    path = tmp_path / "live.sqlite3"
    ledger = orders.AutomaticOrders(path, client, retry_expired=retry_expired)
    assert ledger.place(entry, payload)["status"] == "live_order_unknown"
    attempt, = ledger.recent()
    assert attempt["state"] == "unknown" and not ledger.can_retry(attempt)
    assert orders.AutomaticOrders(path, client, retry_expired=retry_expired).place({**entry, "event": f"{SLUG}-new"}, payload)[
        "status"] == "live_order_already_attempted"
    assert seen == [True]
