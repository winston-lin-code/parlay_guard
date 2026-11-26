from datetime import datetime, timedelta, timezone

import pytest

from src import watch_halftime as watcher


NOW = datetime(2026, 11, 26, 2, tzinfo=timezone.utc)


def test_caches_fixture_and_score_but_prices_again_on_every_scan(monkeypatch):
    game = {"id": "1", "league": "NBA", "tip": NOW, "scores": {"away": 50, "home": 54}}
    selection = object()
    prepared, priced = [], []
    tick = [0]
    monkeypatch.setattr(watcher.time, "monotonic", lambda: tick[0])
    def prepare(row):
        prepared.append(row["scores"]["home"])
        return selection
    def price(row, picked):
        priced.append(row["scores"]["home"])
        assert picked == selection
        return {}
    monkeypatch.setattr(watcher, "prepare_selection", prepare)
    monkeypatch.setattr(watcher, "price_selection", price)

    prices = watcher.WatchPrices()
    prices.price(game)
    tick[0] = 2
    prices.price(game)
    prices.price({**game, "scores": {"away": 50, "home": 55}})
    tick[0] = 32
    prices.discard_expired()
    prices.price(game)
    assert prepared == [54, 55, 54]
    assert priced == [54, 54, 55, 54]


def test_alert_rejects_expired_future_and_invalid_quotes():
    def quote(offset):
        return {"quote_utc": (NOW + timedelta(seconds=offset)).isoformat()}
    assert list(watcher.alertable_picks([quote(-6), quote(-5), quote(6), {"quote_utc": ""}], NOW)) == [quote(-5)]


def test_new_and_reopened_quotes_alert_once_and_removed_quotes_are_reported(monkeypatch, capsys):
    class FrozenClock(datetime):
        @staticmethod
        def now(tz):
            return NOW

    pick = {"event": "nba-test", "side": "Home", "title": "Away @ Home",
            "quote_utc": NOW.isoformat(), "cost": .50, "score_text": "50–55"}
    snapshots = iter([[pick], [pick], [dict(pick, cost=.52)], [], [pick]])

    def screen(**kwargs):
        assert kwargs["leagues"] == ("NBA",)
        try:
            return {"games_at_halftime": 1, "warnings": [], "picks": next(snapshots)}
        except StopIteration:
            raise KeyboardInterrupt from None

    emitted = []
    def emit(row, *, bell):
        emitted.append((row["cost"], bell))
        return True

    monkeypatch.setattr(watcher, "screen_halftime", screen)
    monkeypatch.setattr(watcher, "emit", emit)
    monkeypatch.setattr(watcher, "datetime", FrozenClock)
    monkeypatch.setattr(watcher.time, "sleep", lambda seconds: None)
    with pytest.raises(KeyboardInterrupt):
        watcher.watch(leagues=("NBA",), interval=2, once=False, bell=True)

    assert emitted == [(.50, True), (.52, False), (.50, True)]
    assert capsys.readouterr().out.count("Quote no longer verified") == 1
