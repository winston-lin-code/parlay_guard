"""Read-only paper portfolio endpoint coverage. Auto-refresh is never started."""
from fastapi.testclient import TestClient

import src.api as api


def test_portfolio_api_defaults_off_and_never_mutates(tmp_path, monkeypatch):
    monkeypatch.setattr(api, "ROOT", tmp_path)
    monkeypatch.delenv("PG_ENABLE_PAPER_AUTO", raising=False)
    client = TestClient(api.app)
    res = client.get("/api/paper-auto/summary")
    assert res.status_code == 200
    data = res.json()
    assert data["paper_only"] is True
    assert data["live_orders_submitted"] == 0
    assert data["capture_enabled"] is False
    assert data["status"] == "NOT_STARTED"
    assert data["summary"]["starting_bankroll"] == 100
    assert not (tmp_path / "data/paper_trades/auto_execution.sqlite3").exists()
    assert "no-store" in res.headers["cache-control"]


def test_paper_dashboard_route_is_read_only(tmp_path, monkeypatch):
    monkeypatch.setattr(api, "ROOT", tmp_path)
    client = TestClient(api.app)
    assert client.get("/paper-auto").status_code == 503
    page = tmp_path / "static/paper_auto_dashboard.html"
    page.parent.mkdir(parents=True)
    page.write_text("<html><body>Paper Portfolio</body></html>", encoding="utf-8")
    response = client.get("/paper-auto")
    assert response.status_code == 200
    assert "Paper Portfolio" in response.text
    assert "no-store" in response.headers["cache-control"]
    monkeypatch.setenv("PG_ENABLE_PAPER_AUTO", "1")
    assert client.get("/api/paper-auto/summary").json()["capture_enabled"] is True
