from decimal import Decimal
from types import SimpleNamespace

import pytest

from src import tennis_wager as wager


def test_verified_itf_match_and_capped_fill_or_kill():
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


def test_paper_catalog_scores_check_title_order_and_track_set_break():
    from datetime import timedelta

    slug = "itfwo-player1-player2"
    start = wager.now() - timedelta(seconds=10)
    teams = [{"name": "Player One"}, {"name": "Player Two"}]
    first = {"slug": slug, "title": "Player One vs. Player Two", "teams": teams, "live": True,
             "score": "5-4:40-0", "period": "S1", "updatedAt": start.isoformat()}
    baseline = wager.catalog_scores({slug: first}, {})
    second = {**first, "score": "6-4, 0-0:0-0", "period": "S2", "updatedAt": wager.now().isoformat()}
    result = wager.catalog_scores({slug: second}, baseline)[slug]
    assert result["source"] == "catalog"
    assert result["first_set_break_utc"] == second["updatedAt"]
    assert not wager.catalog_scores({slug: {**first, "teams": teams[::-1]}}, {})
    assert not wager.catalog_scores({slug: {**first, "updatedAt": (start - timedelta(minutes=1)).isoformat()}}, {})


def test_pricing_uses_newer_catalog_score_and_preserves_transition(monkeypatch):
    from datetime import timedelta

    slug = "itfme-strombachs-yadav"
    now = wager.now()
    teams = [{"name": "Robert Strombachs"}, {"name": "Yash Yadav"}]
    first = {"slug": slug, "title": "Robert Strombachs vs. Yash Yadav", "teams": teams, "live": True,
             "score": "5-4:40-0", "period": "S1", "updatedAt": (now - timedelta(seconds=5)).isoformat()}
    listed = wager.catalog_scores({slug: first}, {})[slug]
    second = {**first, "score": "6-4, 0-0:0-0", "period": "S2", "updatedAt": now.isoformat()}
    newest, reason = wager.resolve_catalog_score(slug, listed, second)
    assert reason is None
    assert newest["first_set_break_utc"] == second["updatedAt"]
    assert wager.resolve_catalog_score(slug, newest, first) == (newest, None)
    assert wager.resolve_catalog_score(slug, listed, {**second, "updatedAt": first["updatedAt"]})[0] is None

    monkeypatch.setattr(wager, "public", lambda path: {"event": second})

    def capture(event, state, report):
        assert state == newest
        report("observed newer score")

    monkeypatch.setattr(wager, "match_selection", capture)
    assert wager.price_candidate((slug, listed, first), Decimal("2")) == (None, "observed newer score")


def test_catalog_recognizes_missed_opening_only_with_recent_set_one_score():
    from datetime import timedelta

    slug = "itfwo-player1-player2"
    now = wager.now()
    first = {"slug": slug, "title": "Player One vs. Player Two",
             "teams": [{"name": "Player One"}, {"name": "Player Two"}], "live": True,
             "score": "5-4:40-0", "period": "S1", "updatedAt": (now - timedelta(seconds=10)).isoformat()}
    previous = wager.catalog_scores({slug: first}, {})
    second = {**first, "score": "6-4, 1-0:15-0", "period": "S2", "updatedAt": now.isoformat()}
    assert wager.catalog_scores({slug: second}, previous)[slug]["first_set_break_utc"] == first["updatedAt"]
    assert wager.catalog_scores({slug: second}, {})[slug]["first_set_break_utc"] is None
    late = {**second, "score": "6-4, 2-0:0-0"}
    assert wager.catalog_scores({slug: late}, previous)[slug]["first_set_break_utc"] is None
    old = {slug: {**previous[slug], "score_utc": (now - timedelta(seconds=61)).isoformat()}}
    assert wager.catalog_scores({slug: second}, old)[slug]["first_set_break_utc"] is None
