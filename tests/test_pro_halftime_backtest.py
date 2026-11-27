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


def test_any_window_uses_first_qualifying_quote_and_same_five_dollar_cap(monkeypatch):
    monkeypatch.setattr(replay.model, "forecast", lambda *_: .70)
    early_entry = AT + timedelta(seconds=3)
    halftime = AT + timedelta(minutes=6)
    halftime_entry = halftime + timedelta(seconds=3)
    games = {EVENT: GAME}
    states = {EVENT: {AT: state(AT, "in_play"), early_entry: state(early_entry, "in_play"),
                      halftime: state(halftime), halftime_entry: state(halftime_entry)}}
    book_times = (AT + timedelta(seconds=2), halftime + timedelta(seconds=2))
    records = [(t, book(t)) for t in book_times]
    stake = replay.Decimal("5")

    for window, expected in (("any", early_entry), ("halftime", halftime_entry)):
        coverage = defaultdict(Counter)
        checks = replay.confirmed_entries(states, games, coverage, window)
        books = replay.available_books(checks, records)
        trade, = replay.simulate(games, checks, books, coverage, stake=stake)
        assert trade["entry_at"] == expected.isoformat()
        assert trade["cost"] <= stake
        assert trade["checkpoint"] == int(window == "halftime")

    coverage = defaultdict(Counter)
    checks = replay.confirmed_entries(states, games, coverage, "any")
    halted = [(book_times[0], book(book_times[0], "MARKET_STATE_HALTED")), records[1]]
    trade, = replay.simulate(games, checks, replay.available_books(checks, halted), coverage, stake=stake)
    assert trade["entry_at"] == halftime_entry.isoformat()


def test_high_ev_replay_passes_higher_edge_and_skips_low_ev(monkeypatch):
    monkeypatch.setattr(replay.model, "forecast", lambda *_: .70)
    games = {EVENT: GAME}
    early_entry = AT + timedelta(seconds=3)
    later = AT + timedelta(minutes=1)
    later_entry = later + timedelta(seconds=3)
    states = {EVENT: {t: state(t, "in_play") for t in (AT, early_entry, later, later_entry)}}
    coverage = defaultdict(Counter)
    checks = replay.confirmed_entries(states, games, coverage, "any")
    early_book = AT + timedelta(seconds=2)
    later_book = later + timedelta(seconds=2)
    cheaper = book(later_book)
    cheaper["bids"][0]["px"]["value"] = ".45"
    books = replay.available_books(checks, [(early_book, book(early_book)), (later_book, cheaper)])
    stake = replay.Decimal("5")

    low, = replay.simulate(games, checks, books, coverage, stake=stake)
    high, = replay.simulate(games, checks, books, coverage, stake=stake, high_ev=True)
    assert low["entry_at"] == early_entry.isoformat()
    assert 0 < low["screened_roi"] < .05
    assert high["entry_at"] == later_entry.isoformat()
    assert high["screened_roi"] >= .05
    assert high["cost"] <= stake
