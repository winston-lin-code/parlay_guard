import csv
from datetime import datetime, timedelta, timezone
from pathlib import Path

from src import live_dashboard_api as d


def test_conservative_roi_and_risk_gate():
    game = {"game_id": "0026000001", "selected_team": "BOS",
            "selected_probability": 0.70, "roster_risk": "CLEAR"}
    quote = {"status": "ok", "sides": {"BOS": {"ask": 0.50}}}
    result = d._market_result(game, quote)
    assert result["status"] == "REVIEW"
    assert round(result["estimated_roi"], 6) == round(0.65/0.52-1, 6)
    game["roster_risk"] = "UNVERIFIED"
    assert d._market_result(game, quote)["status"] == "SKIP"
    assert d._market_result(game, {"status": "unavailable"})["status"] == "NO_QUOTE"


def test_reads_only_future_immutable_predictions_and_requires_roster(tmp_path, monkeypatch):
    monkeypatch.setattr(d, "ROOT", tmp_path)
    (tmp_path / "models").mkdir()
    now = datetime.now(timezone.utc)
    today = now.astimezone(d.ET).date().isoformat()
    forecast = tmp_path / "models" / f"live_predictions_{today}_20261118T120000Z.csv"
    with forecast.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["GAME_DATE", "GAME_ID", "AWAY_TEAM", "HOME_TEAM",
                                         "TIPOFF_UTC", "GENERATED_AT_UTC", "P_HOME_WIN_ELO"])
        w.writeheader()
        w.writerow({"GAME_DATE": today, "GAME_ID": "2600001", "AWAY_TEAM": "NYK", "HOME_TEAM": "BOS",
                    "TIPOFF_UTC": (now+timedelta(hours=2)).isoformat(),
                    "GENERATED_AT_UTC": (now-timedelta(hours=1)).isoformat(),
                    "P_HOME_WIN_ELO": "0.70"})
        w.writerow({"GAME_DATE": today, "GAME_ID": "2600002", "AWAY_TEAM": "MIA", "HOME_TEAM": "MIL",
                    "TIPOFF_UTC": (now-timedelta(hours=1)).isoformat(),
                    "GENERATED_AT_UTC": (now-timedelta(hours=2)).isoformat(),
                    "P_HOME_WIN_ELO": "0.61"})
    data = d._load_games()
    assert len(data["games"]) == 1
    assert data["games"][0]["selected_team"] == "BOS"
    assert data["games"][0]["roster_risk"] == "UNVERIFIED"


def test_api_endpoints_use_real_snapshot_and_mocked_market(tmp_path, monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    monkeypatch.setattr(d, "ROOT", tmp_path)
    (tmp_path / "models").mkdir()
    (tmp_path / "static").mkdir()
    (tmp_path / "static" / "live.html").write_text("<h1>Live</h1>")
    now = datetime.now(timezone.utc)
    today = now.astimezone(d.ET).date().isoformat()
    with (tmp_path / "models" / f"live_predictions_{today}_20261118T120000Z.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["GAME_DATE", "GAME_ID", "AWAY_TEAM", "HOME_TEAM",
                                         "TIPOFF_UTC", "GENERATED_AT_UTC", "P_HOME_WIN_ELO"])
        w.writeheader()
        w.writerow({"GAME_DATE": today, "GAME_ID": "2600003", "AWAY_TEAM": "NYK", "HOME_TEAM": "BOS",
                    "TIPOFF_UTC": (now+timedelta(hours=2)).isoformat(),
                    "GENERATED_AT_UTC": (now-timedelta(hours=1)).isoformat(),
                    "P_HOME_WIN_ELO": "0.70"})
    (tmp_path / "data").mkdir()
    with (tmp_path / "data" / "roster_checks.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["game_date", "team", "status", "reason", "checked_at_utc"])
        w.writeheader()
        for team in ("NYK", "BOS"):
            w.writerow({"game_date": today, "team": team, "status": "CLEAR", "reason": "Verified",
                        "checked_at_utc": (now-timedelta(minutes=10)).isoformat()})
    monkeypatch.setattr(d, "get_live_quotes", lambda games: {"games": {
        f"NYK|BOS|{today}": {"status": "ok", "sides": {"BOS": {"ask": 0.50}}}
    }})
    monkeypatch.setitem(d._quote_cache, "key", None)
    monkeypatch.setitem(d._quote_cache, "data", None)
    app = FastAPI()
    app.include_router(d.router)
    client = TestClient(app)
    assert client.get("/live").status_code == 200
    games = client.get("/api/live/games").json()["games"]
    assert len(games) == 1 and games[0]["roster_risk"] == "CLEAR"
    quotes = client.get("/api/live/quotes").json()["results"]
    assert quotes[0]["status"] == "REVIEW"
    assert quotes[0]["estimated_roi"] > 0.05
