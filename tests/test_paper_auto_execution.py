"""Offline tests: no network and no actual Polymarket orders."""
import sqlite3
from datetime import datetime, timedelta, timezone
from src.paper_auto_execution import on_live_refresh

def fixture(now, *, ask=.4, p=.75, size=100, lag_seconds=10):
    ts = lambda dt: dt.isoformat()
    return {"updated_at_utc": ts(now - timedelta(seconds=5)), "games": [{
        "slug": "nba-nyk-bos-2026-11-21", "away": "NYK", "home": "BOS",
        "tipoff_utc": ts(now + timedelta(hours=1)),
        "winner_forecast": {
            "predicted_winner": "BOS", "home_probability": p,
            "forecast_generated_at_utc": ts(now - timedelta(minutes=15)),
        },
        "moneylines": {"BOS": {"ask": ask, "shares": size,
                               "token_id": "123456789",
                               "observed_at_utc": ts(now - timedelta(seconds=lag_seconds))}},
    }]}

def test_paper_fill_is_idempotent(tmp_path):
    now = datetime(2026, 11, 21, 21, tzinfo=timezone.utc)
    data = fixture(now)
    a = on_live_refresh(data, tmp_path, at=now)
    b = on_live_refresh(data, tmp_path, at=now)
    assert len(a["assumed_fills"]) == 1
    assert a["paper_only"] and a["daily_paper_exposure_dollars"] == 5
    assert b["assumed_fills"] == []
    assert b["skipped"]["nba-nyk-bos-2026-11-21"] == "already_paper_traded"
    with sqlite3.connect(tmp_path / "data/paper_trades/auto_execution.sqlite3") as db:
        assert db.execute("SELECT count(*) FROM paper_entries").fetchone()[0] == 1

def test_rejects_stale_quote_and_weak_edge(tmp_path):
    now = datetime(2026, 11, 21, 21, tzinfo=timezone.utc)
    result = on_live_refresh(fixture(now, lag_seconds=121), tmp_path, at=now)
    assert result["skipped"]["nba-nyk-bos-2026-11-21"] == "missing_or_stale_quote"
    result = on_live_refresh(fixture(now, ask=.70), tmp_path, at=now)
    assert result["skipped"]["nba-nyk-bos-2026-11-21"] == "insufficient_stressed_ev"

def test_missing_token_or_stale_snapshot_fail_closed(tmp_path):
    now = datetime(2026, 11, 21, 21, tzinfo=timezone.utc)
    data = fixture(now)
    data["games"][0]["moneylines"]["BOS"].pop("token_id")
    assert not on_live_refresh(data, tmp_path, at=now)["assumed_fills"]
    data = fixture(now)
    data["updated_at_utc"] = (now - timedelta(minutes=20)).isoformat()
    assert on_live_refresh(data, tmp_path, at=now)["reason"] == "invalid_or_stale_snapshot"

def test_daily_paper_cap(tmp_path):
    now = datetime(2026, 11, 21, 21, tzinfo=timezone.utc)
    data = fixture(now)
    game = data["games"][0]
    data["games"] = [{**game, "slug": f"nba-{team.lower()}-bos-2026-11-21",
                      "away": team} for team in ("NYK","CHI","LAL","MIA","DET")]
    result = on_live_refresh(data, tmp_path, at=now)
    assert len(result["assumed_fills"]) == 4
    assert result["daily_paper_exposure_dollars"] == 20
    assert list(result["skipped"].values()) == ["daily_paper_cap"]


def test_reject_event_identity_mismatch(tmp_path):
    now = datetime(2026, 11, 21, 21, tzinfo=timezone.utc)
    data = fixture(now)
    data["games"][0]["away"] = "LAL"
    assert on_live_refresh(data, tmp_path, at=now)["skipped"]["nba-nyk-bos-2026-11-21"] == "event_identity_or_date_mismatch"
