from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone

import pytest
import requests

from src import cs2_live_ev as cs2, cs2_ratings as elo
from src import env_config

NOW = datetime(2026, 11, 27, 18, 3, 15, tzinfo=timezone.utc)
SLUG = "cs2-navi-m80-2026-11-27"
MARKET = "aec-" + SLUG


def test_reader_loads_project_env_outside_the_project_directory(monkeypatch, tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text('CS2_CITO_API_KEY="test#key"\nUNRELATED_TEST_SETTING=unused\n', encoding="utf-8-sig")
    monkeypatch.setattr(env_config, "PROJECT_ENV", env_file)
    monkeypatch.delenv("CS2_CITO_API_KEY", raising=False)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)

    assert cs2.Reader().cito_key == "test#key"
    monkeypatch.setenv("CS2_CITO_API_KEY", "process-value")
    assert cs2.Reader().cito_key == "process-value"
    monkeypatch.setenv("CS2_CITO_API_KEY", "")
    assert cs2.Reader().cito_key == ""


def event():
    start = NOW - timedelta(minutes=5)
    stamp = NOW - timedelta(seconds=5)
    return {"slug": SLUG, "title": "NAVI vs M80", "startTime": start.isoformat(),
            "live": True, "ended": False, "score": "0-0",
            "teams": [{"id": 10, "name": "NAVI"}, {"id": 20, "name": "M80"}],
            "eventState": {"live": True, "period": "Map 1", "score": "0-0", "updatedAt": stamp.isoformat()},
            "markets": [{"slug": MARKET, "question": "NAVI vs M80", "marketType": "moneyline",
                         "sportsMarketType": "esports_match_winner", "status": "MARKET_STATUS_OPEN",
                         "active": True, "closed": False, "feeCoefficient": ".11", "minimumTradeQty": ".01",
                         "orderPriceMinTickSize": ".01",
                         "marketSides": [{"teamId": 20, "long": True, "tradable": True},
                                         {"teamId": 10, "long": False, "tradable": True}]}]}


def book(age=1):
    def level(price):
        return {"px": {"currency": "USD", "value": price}, "qty": "100"}
    return {"marketSlug": MARKET, "state": "MARKET_STATE_OPEN",
            "transactTime": (NOW - timedelta(seconds=age)).isoformat(),
            "bids": [level(".70")], "offers": [level(".73")]}


def http_error(status, retry_after=None):
    response = requests.Response()
    response.status_code = status
    if retry_after is not None:
        response.headers["Retry-After"] = str(retry_after)
    return requests.HTTPError("fixture HTTP error", response=response)


def test_cs2_catalog_can_recover_from_a_failed_live_filter(monkeypatch):
    monkeypatch.setattr(cs2, "now", lambda: NOW)
    calls = []

    def get(method, url, *, max_age, params):
        assert (method, url) == ("GET", cs2.GATEWAY + "/v1/events")
        assert max_age == cs2.EVENT_CACHE_MAX_AGE > cs2.BOOK_AGE
        assert params["tagSlug"] == "cs2"
        calls.append(params)
        if "live" in params:
            raise http_error(503)
        return {"events": [event(), {**event(), "slug": "cs2-upcoming-2026-11-27", "live": False}]}

    monkeypatch.setattr(cs2, "read_json", get)
    matches = [match for raw in cs2.live_events() if (match := cs2.fixture(raw, NOW))]
    assert [match["event"] for match in matches] == [SLUG]
    assert len(calls) == 2 and "startDateMin" in calls[1] and "startDateMax" in calls[1]


def test_cs2_catalog_reports_and_honors_rate_limit_without_retrying_another_filter(monkeypatch):
    tick = 100.
    calls = []

    def get(method, url, *, max_age, params):
        calls.append(params)
        if len(calls) == 1:
            raise http_error(429, 20)
        return {"events": []}

    monkeypatch.setattr(cs2, "read_json", get)
    monkeypatch.setattr(cs2, "env_value", lambda _: None)
    monkeypatch.setattr(cs2.time, "monotonic", lambda: tick)
    reader = cs2.Reader()
    reader.rating_checked = tick
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = reader.snapshot(pool)
        tick += 15
        still_limited = reader.snapshot(pool)
        assert len(calls) == 1 and calls[0]["live"] == "true"
        assert first["catalog_status"] == still_limited["catalog_status"] == "rate_limited"
        assert first["warnings"] == ["Polymarket US rate-limited CS2 listings."]
        tick += 5
        recovered = reader.snapshot(pool)
        assert recovered["catalog_status"] == "ready" and recovered["games"] == []
        assert recovered["warnings"] == [] and len(calls) == 2


@pytest.mark.parametrize(("code", "status", "cooldown"), [(401, "access_denied", 300), (403, "access_denied", 300),
                                                       (429, "rate_limited", 37)])
def test_cito_reports_auth_or_rate_limit_without_exposing_key(monkeypatch, code, status, cooldown):
    tick = 100.
    monkeypatch.setattr(cs2.time, "monotonic", lambda: tick)
    monkeypatch.setattr(cs2, "env_value", lambda _: "fixture-secret")
    calls = []

    def live_list(key):
        calls.append(key)
        if len(calls) == 1:
            raise http_error(code, 37)
        return []

    monkeypatch.setattr(cs2, "cito_list", live_list)
    reader = cs2.Reader()
    reader.refresh_cito(tick, has_matches=True)
    assert reader.cito_status == status
    assert str(code) in reader.cito_warning and "fixture-secret" not in reader.cito_warning
    tick += cooldown - 1
    reader.refresh_cito(tick, has_matches=True)
    assert len(calls) == 1
    tick += 1
    reader.refresh_cito(tick, has_matches=True)
    assert reader.cito_status == "no_live_matches" and reader.cito_warning is None
    assert len(calls) == 2


def test_series_and_mr12_progression():
    assert cs2.series_probability((1, 0), 3, .5) == .75
    assert cs2.series_probability((0, 1), 3, .5) == .25
    assert cs2.map_probability((12, 12), .5) == .5
    assert cs2.map_probability((13, 12), .5) == .65625
    assert cs2.map_probability((15, 15), .5) == .5
    assert cs2.map_probability((16, 14), .5) == 1
    assert cs2.map_probability((13, 11), .5) == 1
    assert cs2.projections((1600, 1500), (0, 0), (3,)) == pytest.approx([.640065], abs=1e-6)
    leading = cs2.projections((1500, 1500), (0, 0), (3,), (11, 5))[0]
    trailing = cs2.projections((1500, 1500), (0, 0), (3,), (5, 11))[0]
    assert leading > .5 > trailing
    assert leading + trailing == pytest.approx(1.)


def test_live_market_bounds_and_expiry(monkeypatch):
    monkeypatch.setattr(cs2, "now", lambda: NOW)
    monkeypatch.setattr(cs2, "public", lambda *args, **kwargs: {"marketData": book()})
    match = cs2.fixture(event(), NOW)
    ranks = [elo.Ratings("example Elo", NOW - timedelta(hours=1), {"natusvincere": 1600, "m80": 1500})]
    row = cs2.evaluate(match, ranks, [], None)
    assert row["status"] == "priced"
    assert row["formats"] == (1, 3, 5)
    navi, m80 = row["sides"]
    assert navi["ask"] == .30 and m80["ask"] == .73
    assert navi["probability_high"] > navi["probability_low"] > .5
    assert navi["ev_high"] >= navi["ev_low"] > 0
    assert navi["total_cost"] <= 5
    display = cs2.read_for_display({"games": [row]}, NOW + timedelta(seconds=13))
    assert "ask" not in display["games"][0]["sides"][0]
    assert "ev_low" not in display["games"][0]["sides"][0]

    def old_book(*args, **kwargs):
        return {"marketData": book(age=8)}

    monkeypatch.setattr(cs2, "public", old_book)
    row = cs2.evaluate(match, ranks, [], None)
    assert row["status"] == "waiting_for_book"
    assert "ask" in row["sides"][0] and "ev_low" not in row["sides"][0]
    row = cs2.evaluate(match, [], [], None)
    assert row["status"] == "rating_missing"
    assert "probability_low" not in row["sides"][0]


def test_round_book_alignment_and_orientation(monkeypatch):
    monkeypatch.setattr(cs2, "now", lambda: NOW)
    monkeypatch.setattr(cs2, "public", lambda *args, **kwargs: {"marketData": book()})
    match = cs2.fixture(event(), NOW)
    ranks = [elo.Ratings("example Elo", NOW, {"natusvincere": 1500, "m80": 1500})]
    provider = {"status": "live", "matchId": "cs2-match-101", "bestOf": 3,
                "score": {"team1": 0, "team2": 0},
                "team1Name": "M80", "team2Name": "Natus Vincere", "team1Id": "B", "team2Id": "A"}
    board = {"matchId": "cs2-match-101", "status": "live", "currentGame": 1, "bestOf": 3,
             "teams": [{"id": "B", "score": 0, "mapScore": 5}, {"id": "A", "score": 0, "mapScore": 10}],
             "meta": {"updatedAt": (NOW - timedelta(seconds=2)).isoformat()}, "game": {"map": "de_mirage"}}
    monkeypatch.setattr(cs2, "read_json", lambda *args, **kwargs: {"data": board})
    row = cs2.evaluate(match, ranks, [provider], "fixture-key")
    assert row["rounds"] == (10, 5)
    assert row["formats"] == (3,)
    assert row["model_scope"] == "round_score"
    assert row["sides"][0]["probability_low"] > .5
    assert "ev_low" in row["sides"][0]
    old_venue = {**match, "score_at": NOW - timedelta(minutes=3)}
    refreshed = cs2.evaluate(old_venue, ranks, [provider], "fixture-key")
    assert cs2.read_for_display({"games": [refreshed]}, NOW)["games"][0]["status"] == "priced"
    board["teams"][1]["score"] = 1
    board["currentGame"] = 2
    refreshed = cs2.evaluate(old_venue, ranks, [provider], "fixture-key")
    assert refreshed["maps"] == (1, 0)
    assert refreshed["period"] == "Map 2"
    assert refreshed["sides"][0]["probability_low"] > .75
    board["meta"]["updatedAt"] = (NOW - timedelta(seconds=60)).isoformat()
    row = cs2.evaluate(match, ranks, [provider], "fixture-key")
    assert row["model_scope"] == "series_score"
    assert row["sides"][0]["probability_low"] == pytest.approx(.5)
    assert "rounds" not in row


def test_venue_rounds_use_competitor_ids():
    raw = event()
    raw["eventState"]["periodScores"] = [{"number": 1, "label": "Map 1",
                                          "scores": [{"competitorId": "20", "score": 2},
                                                     {"competitorId": "10", "score": 7}]}]
    assert cs2.fixture(raw, NOW)["rounds"] == (7, 2)
    raw["eventState"]["periodScores"][0]["scores"][0]["competitorId"] = "999"
    assert cs2.fixture(raw, NOW)["rounds"] is None
    raw["markets"][0]["sportsMarketType"] = "esports_map_winner_1"
    assert cs2.fixture(raw, NOW) is None


def table(rows):
    return '<table><tr><th>#</th><th>Team</th><th>Elo</th></tr>' + "".join(
        f"<tr><td>1</td><td><a>{team}</a></td><td>{rating}</td></tr>" for team, rating in rows
    ) + "</table>"


def test_loads_paginated_elo_and_rejects_outdated_ranking(monkeypatch):
    table1 = table([("Natus Vincere", 1577), ("M80", 1504)])
    table2 = table([("MOUZ", 1600), ("Team Liquid", 1532)])
    pages = iter(["<p>Data is up to date: 19 minutes ago</p>", table1 + "<a>Next</a>", table2])
    monkeypatch.setattr(elo, "get_page", lambda *args, **kwargs: next(pages))
    source = elo.analyzer_ratings(NOW)
    assert source.as_of == NOW - timedelta(minutes=19)
    assert source.teams == {"natusvincere": 1577, "m80": 1504, "mouz": 1600, "liquid": 1532}
    monkeypatch.setattr(elo, "get_page", lambda *args, **kwargs: table1 + "<p>Last updated 2026-08-28 13:07 UTC</p>")
    with pytest.raises(ValueError, match="outdated"):
        elo.predict_ratings(NOW)


def test_loads_dated_fragcharts_map_elo(monkeypatch):
    entries = [("100 Thieves", "1,725"), ("HAVU", "1,415"), ("BOJONG", "1,312")]
    entries.extend((f"Example club {i}", "1,400") for i in range(18))
    cards = "".join(f"<div><span>{i}</span><a>{name}</a><span>{rating}</span><span>—</span>"
                    "<span>no Valve file</span></div>" for i, (name, rating) in enumerate(entries, 1))
    monkeypatch.setattr(elo, "get_page", lambda *args: "<p>Fragcharts rating · 27 November 2026</p>" + cards)
    source = elo.fragcharts_ratings(NOW)
    assert source.per_map is True
    assert source.teams["100thieves"] == 1725
    assert source.teams["bojong"] == 1312
    assert cs2.projections((1600, 1500), (0, 0), (1,), per_map=True) == pytest.approx([.640065], abs=1e-6)


def test_reader_uses_dated_cache_or_snapshot_until_expiry(monkeypatch, tmp_path):
    cache = tmp_path / "ratings.json"
    monkeypatch.setattr(elo, "CACHE_PATH", cache)
    monkeypatch.setattr(cs2, "now", lambda: NOW)

    seed = cs2.Reader().ratings[0]
    assert seed.origin == "bundled"
    assert seed.as_of.date().isoformat() == "2026-11-27"
    assert seed.teams["bojong"] == 1312
    assert seed.teams[elo.team_key("FUT Esports")] == 1798

    update = elo.Ratings("Fragcharts", NOW, {"natusvincere": 1831, "m80": 1661}, per_map=True)
    elo.save_ratings(cache, [update])
    saved = cs2.Reader().ratings[0]
    assert saved.origin == "cache" and saved.per_map and saved.teams == update.teams

    match = {**cs2.fixture(event(), NOW), "score_at": NOW - timedelta(minutes=3)}
    monkeypatch.setattr(cs2, "market_book", lambda _: book())
    row = cs2.evaluate(match, [saved], [], None)
    assert row["sides"][0]["ratings"] == [{"source": "Fragcharts", "rating": 1831}]
    assert row["status"] == "waiting_for_score"
    assert "ev_low" not in row["sides"][0]

    monkeypatch.setattr(cs2, "now", lambda: NOW + timedelta(days=3, seconds=1))
    assert cs2.Reader().ratings == []
