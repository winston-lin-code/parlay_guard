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
