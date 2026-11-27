from datetime import timedelta
from decimal import Decimal
import json
from threading import Event
from types import SimpleNamespace

import pytest

from src import tennis_wager as wager


def test_verified_itf_match_and_capped_fill_or_kill(monkeypatch):
    slug = "itfme-strombachs-yadav"
    players = [{"name": "Robert Strombachs", "abbreviation": "robstr"},
               {"name": "Yash Yadav", "abbreviation": "yasyad"}]
    contract = {
        "slug": "aec-" + slug, "marketType": "moneyline", "sportsMarketType": "tennis_match_winner",
        "active": True, "closed": False, "question": "Match winner", "minimumTradeQty": ".01",
        "orderPriceMinTickSize": ".01",
        "marketSides": [{"team": team, "long": index == 0, "tradable": True}
                        for index, team in enumerate(players)],
    }
    event = {"slug": slug, "title": "Robert Strombachs vs. Yash Yadav", "live": True, "active": True,
             "closed": False, "teams": players, "markets": [contract]}
    timestamp = wager.now().isoformat()
    score = {"event_slug": "", "score_league": "ITFME", "score_columns": ["yasyad", "robstr"],
             "phase": "live", "score_utc": timestamp, "score_text": "4-5:0-40", "period_text": "S1"}
    assert wager.catalog_event({slug: event}, score) is event
    market, long, pick = wager.match_selection(event, score)
    assert pick["side"] == "Robert Strombachs" and long
    book = {"marketSlug": market["slug"], "state": "MARKET_STATE_OPEN", "transactTime": timestamp,
            "offers": [{"px": {"value": ".50", "currency": "USD"}, "qty": "10"}]}
    payload, summary = wager.order_for_stake(pick, market, long, book, Decimal("2"))
    assert payload["quantity"] == 3
    assert payload["tif"] == "TIME_IN_FORCE_FILL_OR_KILL"
    assert payload["manualOrderIndicator"] == "MANUAL_ORDER_INDICATOR_AUTOMATIC"
    assert Decimal(summary["total_estimate"]) <= 2
    assert wager.match_selection(event, {**score, "score_columns": ["unknown", "robstr"]}) is None

    def project(match, state, now=None):
        first = {"1-0:15-0": .25, "1-0:30-0": .32, "1-0:15-15": .20}[state["score_text"]]
        return {"first_probability": first, "second_probability": 1 - first,
                "checkpoint": None, "elo_prior": True}

    monkeypatch.setattr(wager, "tennis_projection", project)
    early = {**score, "score_text": "1-0:15-0"}
    assert wager.match_selection(event, early) is None
    _, _, stressed = wager.match_selection(event, early, score_based=True)
    assert stressed["displayed_probability"] == .75
    assert stressed["probability"] == pytest.approx(.68)
    stream = wager.BookStream(None, Event())
    stream.set_markets([market["slug"]])
    stream.snapshots[market["slug"]] = book, wager.now()
    monkeypatch.setattr(wager, "public", lambda *args, **kwargs: pytest.fail("unexpected HTTP price request"))
    candidate = ("game", early, event)
    assert wager.price_candidate(candidate, Decimal("2"), stream)[4]["assumed_unseen_points"] == 1
    stream.snapshots[market["slug"]] = book, wager.now() - timedelta(seconds=1)
    diagnostic = {}
    assert wager.price_candidate(candidate, Decimal("2"), stream, diagnostic=diagnostic) is None
    assert diagnostic["reason"] == "book_received_before_score"
    stream.snapshots[market["slug"]] = (
        {**book, "transactTime": (wager.now() - timedelta(seconds=4)).isoformat()}, wager.now(),
    )
    diagnostic = {}
    assert wager.price_candidate(candidate, Decimal("2"), stream, diagnostic=diagnostic) is None
    assert diagnostic["reason"] == "stale_or_invalid_book_timestamp"

    diagnostic = {}
    assert wager.order_for_stake({**pick, "probability": .55}, market, long, book, Decimal("2"),
                                 diagnostic=diagnostic) is None
    assert diagnostic["reason"] == "roi_below_floor"
    assert Decimal(diagnostic["limit"]) == Decimal(".50")
    assert diagnostic["screened_roi"] < diagnostic["minimum_roi"]

    diagnostic = {}
    assert wager.order_for_stake(pick, market, long, {**book, "offers": []}, Decimal("2"),
                                 diagnostic=diagnostic) is None
    assert diagnostic["reason"] == "no_executable_quote"

    stream.snapshots.clear()
    request_paths = []
    def fresh_book(path, **kwargs):
        request_paths.append((path, kwargs))
        return {"marketData": {**book, "bids": [], "transactTime": wager.now().isoformat()}}

    monkeypatch.setattr(wager, "public", fresh_book)
    refreshed = wager.price_candidate(candidate, Decimal("2"), stream, refresh_book=True)
    assert refreshed[4]["book_source"] == "rest_refresh"
    assert wager.price_candidate(candidate, Decimal("2"), stream, refresh_book=True)
    assert request_paths == [(f"/v1/markets/{market['slug']}/book", {"fresh_book": True})]


def test_book_refresh_rejects_stale_exchange_timestamp_and_limits_retries(monkeypatch):
    clock = wager.now()
    monkeypatch.setattr(wager, "now", lambda: clock)
    market = "aec-itfme-example"
    score = {"score_utc": (clock - timedelta(seconds=4)).isoformat(), "score_received_at": clock.isoformat()}
    stream = wager.BookStream(None, Event())
    stream.set_markets([market])
    calls = []
    def public(path, *, fresh_book):
        calls.append(path)
        return {"marketData": {
            "marketSlug": market, "state": "MARKET_STATE_OPEN", "bids": [], "offers": [],
            "transactTime": (clock - timedelta(seconds=5)).isoformat(),
        }}

    monkeypatch.setattr(wager, "public", public)
    diagnostic = {}
    assert stream.refresh_after(market, score, diagnostic=diagnostic) is None
    assert diagnostic == {"reason": "stale_or_invalid_book_timestamp", "book_age_seconds": 5.0}
    assert stream.refresh_after(market, score) is None
    assert stream.refresh_after(market, {**score, "score_utc": (clock - timedelta(seconds=21)).isoformat()}) is None
    assert len(calls) == 1


def test_status_diagnostics_distinguish_stale_missing_and_pricing_rejections(monkeypatch, capsys):
    clock = wager.now()
    monkeypatch.setattr(wager, "now", lambda: clock)
    events = {
        slug: {"slug": slug, "live": True} for slug in ("itfme-stale", "itfme-missing", "itfme-priced")
    }
    scores = wager.Scores()
    scores.games["itfme-stale"] = {"score_utc": (clock - timedelta(seconds=25)).isoformat()}
    wager.report_diagnostics(events, scores, [("itfme-priced", {})],
                             [("itfme-priced", {}, events["itfme-priced"])],
                             {"itfme-priced": {"reason": "missing_elo_rating"}}, {"missing_elo_rating": 2})
    output = json.loads(capsys.readouterr().out)
    assert output["matches"] == {
        "itfme-stale": {"reason": "stale_score", "score_age_seconds": 25.0},
        "itfme-missing": {"reason": "no_usable_us_score"},
        "itfme-priced": {"reason": "missing_elo_rating"},
    }
    assert output["pricing_skips_since_last_status"] == {"missing_elo_rating": 2}


def test_stream_catalog_discovers_live_and_near_start_singles_in_both_leagues(monkeypatch):
    clock = wager.now()

    def event(slug, minutes, live=False, title="Player A vs. Player B", **details):
        return {
            "slug": slug, "title": title, "live": live, "active": True, "closed": False,
            "startTime": (clock + timedelta(minutes=minutes)).isoformat(),
            "markets": [{"slug": f"aec-{slug}", "marketType": "moneyline",
                         "sportsMarketType": "tennis_match_winner", "active": True, "closed": False}],
            **details,
        }

    a = event("itfme-live", -60, live=True)
    b = event("itfwo-upcoming", 10)
    pages = {
        "itfme": [a, event("itfme-finished", -5, ended=True), event("itfme-distant", 90)],
        "itfwo": [b, event("itfwo-doubles", 5, title="A - B vs. C - D")],
    }
    monkeypatch.setattr(wager, "now", lambda: clock)
    def public(path, *, fresh_score):
        assert fresh_score
        return {"events": pages[path.split("/")[3]]}

    monkeypatch.setattr(wager, "public", public)
    assert wager.stream_catalog() == {
        a["slug"]: {**a, "score_received_at": clock}, b["slug"]: {**b, "score_received_at": clock},
    }


def test_us_score_uses_provider_timestamp_and_explicit_side_order(monkeypatch):
    clock = wager.now()
    monkeypatch.setattr(wager, "now", lambda: clock)
    source_time = (clock - timedelta(seconds=4)).isoformat()
    slug = "itfme-one-two"
    event = {
        "slug": slug, "title": "One vs. Two", "live": True, "active": True, "closed": False,
        "eventState": {"type": "tennis", "live": True, "score": "3-1:30-15", "period": "S1",
                       "updatedAt": source_time},
        "markets": [{"slug": f"aec-{slug}", "marketType": "moneyline",
                     "sportsMarketType": "tennis_match_winner", "active": True, "closed": False,
                     "marketSides": [{"team": {"ordering": "away", "abbreviation": "two"}},
                                     {"team": {"ordering": "home", "abbreviation": "one"}}]}],
    }
    first = wager.us_score(event, clock - timedelta(seconds=2))
    assert first["score_columns"] == ["one", "two"]
    assert first["score_utc"] == source_time
    scores = wager.Scores()
    scores.publish(slug, first)
    scores.publish(slug, wager.us_score(event, clock))
    assert scores.get(slug) == first | {"first_set_break_utc": None}
    assert len(scores.active_games()) == 1

    market = f"aec-{slug}"
    book = {"marketSlug": market, "state": "MARKET_STATE_OPEN", "bids": [], "offers": [],
            "transactTime": (clock - timedelta(seconds=1)).isoformat()}
    stream = wager.BookStream(None, Event())
    stream.set_markets([market])
    stream.snapshots[market] = (book, clock - timedelta(seconds=3))
    assert stream.usable_after(market, first) is None
    stream.publish(stream.markets, book)
    assert stream.usable_after(market, first)

    monkeypatch.setattr(wager, "now", lambda: clock + timedelta(seconds=30))
    scores.publish(slug, wager.us_score(event, wager.now()))
    assert scores.active_games() == []
    event["eventState"].pop("updatedAt")
    assert wager.us_score(event, wager.now()) is None


def test_stream_keeps_separate_books_and_rejects_retired_subscriptions():
    first, second, third = "aec-itfme-a", "aec-itfwo-b", "aec-itfwo-c"
    stream = wager.BookStream(None, Event())
    score = {"score_utc": wager.now().isoformat()}
    book = {"state": "MARKET_STATE_OPEN", "bids": [], "offers": [], "transactTime": score["score_utc"]}
    initial = frozenset((first, second))
    stream.set_markets(initial)
    for slug in initial:
        stream.publish(initial, {**book, "marketSlug": slug})
        assert stream.usable_after(slug, score)[0]["marketSlug"] == slug
    stream.publish(initial, {**book, "marketSlug": first, "state": "MARKET_STATE_CLOSED"})
    assert stream.usable_after(first, score) is None
    assert stream.usable_after(second, score)
    updated = frozenset((second, third))
    stream.set_markets(updated)
    stream.publish(initial, {**book, "marketSlug": second})
    assert stream.usable_after(second, score) is None
    stream.publish(updated, {**book, "marketSlug": third})
    assert stream.usable_after(third, score)


@pytest.mark.parametrize(("score", "period", "expected"), [
    ("2-2:AD-40", "S1", [("3-2:0-0", "S1"), ("2-2:40-40", "S1")]),
    ("5-4:40-30", "S1", [("6-4, 0-0:0-0", "S2"), ("5-4:40-40", "S1")]),
    ("6-6:6-5", "TB1", [("7-6, 0-0:0-0", "S2"), ("6-6:6-6", "TB1")]),
    ("6-4, 5-4:40-30", "S2", ()),
    ("2-2", "S1", ()),
])
def test_next_point_scores(score, period, expected):
    assert wager.next_point_scores(score, period) == expected


def test_ledger_blocks_duplicates_and_unresolved_orders(tmp_path):
    store = wager.ledger(tmp_path / "orders.sqlite3")
    summary = {"total_estimate": "1.70"}
    assert wager.reserve(store, "live", "itfme-one", summary, Decimal("3"))
    assert not wager.reserve(store, "live", "itfme-two", summary, Decimal("5"))
    result = SimpleNamespace(orders=SimpleNamespace(create=lambda payload: {"id": "order-1"}))
    assert wager.submit(result, store, "itfme-one", {})["order_id"] == "order-1"
    assert not wager.reserve(store, "live", "itfme-one", summary, Decimal("5"))
    assert not wager.reserve(store, "live", "itfme-two", summary, Decimal("3"))
    assert wager.reserve(store, "live", "itfme-two", summary, Decimal("5"))
    unavailable = SimpleNamespace(orders=SimpleNamespace(create=lambda payload: {}))
    with pytest.raises(RuntimeError, match="Order state unknown"):
        wager.submit(unavailable, store, "itfme-two", {})
    assert not wager.reserve(store, "live", "itfme-three", summary, Decimal("10"))
    store.close()
