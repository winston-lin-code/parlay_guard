from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
import json

from src import pro_halftime_backtest as replay

AT = datetime(2026, 11, 1, tzinfo=timezone.utc)
EVENT = "nhl-one-two"
GAME = {"league": "nhl", "official_game_id": "1", "tick": ".01", "fee_coefficient": ".11",
        "minimum_quantity": "1", "winner": "home", "home_long": "0",
        "settled_at": (AT + timedelta(hours=2)).isoformat()}


def state(at, phase="intermission"):
    official = {"phase": phase, "period": 2, "period_type": "REG", "seconds_remaining": 0,
                "score_away_home": [1, 2], "received_at": at.isoformat()}
    return {"official": official, "polymarket_us": dict(official)}


def book(at, status="MARKET_STATE_OPEN"):
    return {"marketSlug": f"aec-{EVENT}", "transactTime": at.isoformat(), "state": status,
            "offers": [{"px": {"value": ".42", "currency": "USD"}, "qty": "20"}],
            "bids": [{"px": {"value": ".40", "currency": "USD"}, "qty": "20"}]}


def test_replay_only_counts_a_book_known_before_the_post_quote_confirmation(monkeypatch):
    monkeypatch.setattr(replay.model, "forecast", lambda *_: .70)
    coverage = defaultdict(Counter)
    entry = AT + timedelta(seconds=3)
    states = {EVENT: {AT: state(AT), entry: state(entry)}}
    breaks = replay.confirmed_breaks(states, {EVENT: GAME}, coverage)
    assert len(breaks[EVENT]) == 1
    newer = AT + timedelta(seconds=4)
    possible = replay.available_books(breaks, [(newer, book(newer))])
    assert replay.simulate({EVENT: GAME}, breaks, possible, coverage) == []

    received = AT + timedelta(seconds=2)
    available = replay.available_books(breaks, [(received, book(received)), (newer, book(newer))])
    (trade,) = replay.simulate({EVENT: GAME}, breaks, available, coverage)
    assert trade["entry_at"] == entry.isoformat()
    assert trade["selected_side"] == "home"
    assert 0 < trade["screened_roi"] < .05
    assert trade["cost"] <= 10
    assert trade["won"] == 1

    halted = AT + timedelta(seconds=2, milliseconds=500)
    available = replay.available_books(breaks, [(received, book(received)), (halted, book(halted, "MARKET_STATE_HALTED"))])
    assert replay.simulate({EVENT: GAME}, breaks, available, coverage) == []


def test_native_epoch_is_decoded_and_new_period_disqualifies_entry(tmp_path):
    ns = int(AT.timestamp()) * 10**9 + 2000
    path = tmp_path / "books.jsonl"
    path.write_text(json.dumps({"kind": "book_update", "received_at_ns": ns, "book": book(AT)}))
    ((received, _),) = replay.snapshots([path])
    assert received == AT + timedelta(microseconds=2)
    later = AT + timedelta(seconds=5)
    assert replay.confirmed_breaks({EVENT: {AT: state(AT), later: state(later, "in_play")}},
                                   {EVENT: GAME}, defaultdict(Counter)) == {}
