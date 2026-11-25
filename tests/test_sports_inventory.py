from datetime import datetime, timezone
from threading import Event

from src.sports_inventory import add_fixtures, scan_sports_events

NOW = datetime(2026, 11, 25, 19, tzinfo=timezone.utc)


def event(slug, title, *markets):
    return {
        "slug": slug, "title": title, "startTime": "2026-11-25T18:00:00Z",
        "seriesSlug": "nhl-2026", "series": [{"id": "99", "title": "NHL 2026"}],
        "tags": [{"slug": "hockey", "label": "Hockey"}, {"slug": "games", "label": "Games"}],
        "markets": [
            {"id": market, "active": True, "acceptingOrders": True, "enableOrderBook": True}
            for market in markets
        ],
    }


def test_game_groups_child_events_and_deduplicates_markets():
    key = "nhl-bos-nyr-2026-11-25"
    main = event(key, "Bruins vs. Rangers", "1", "2")
    child = {**event(key + "-totals", "Bruins vs. Rangers: totals", "2", "3"),
             "live": True, "score": "2–0", "period": "P2"}
    unrelated = event("season-awards", "Player awards", "4")
    unrelated["tags"] = [{"slug": "sports"}]
    rows = {}
    add_fixtures(rows, [child, unrelated, main], NOW, {"99": "NHL"})
    assert set(rows) == {key}
    assert rows[key]["title"] == "Bruins vs. Rangers"
    assert rows[key]["open_market_ids"] == {"1", "2", "3"}
    assert rows[key]["league"] == "NHL"
    assert rows[key]["phase"] == "live"
    assert rows[key]["score_text"] == "2–0"


def test_catalog_follows_cursor_and_includes_unmodeled_sports(monkeypatch):
    class Clock(datetime):
        @staticmethod
        def now(tz):
            return NOW

    def response(method, url, params=None):
        if url.endswith("/sports"):
            return [{"series": "99", "name": "NHL"}]
        assert params["tag_id"] == 1
        if "after_cursor" not in params:
            return {"events": [event("nhl-bos-nyr-2026-11-25", "Bruins vs. Rangers", "1")],
                    "next_cursor": "page2"}
        assert params["after_cursor"] == "page2"
        tennis = {**event("atp-player-opponent-2026-11-25", "Player vs. Opponent", "2"), "series": [],
                  "seriesSlug": "", "tags": [{"slug": "games"}, {"slug": "tennis"}]}
        return {"events": [tennis]}

    monkeypatch.setattr("src.sports_inventory.read_json", response)
    monkeypatch.setattr("src.sports_inventory.datetime", Clock)
    result = scan_sports_events(Event())
    assert result["status"] == "complete"
    assert [row["sport"] for row in result["games"]] == ["Hockey", "Tennis"]
    assert all(row["open_markets"] == 1 for row in result["games"])
