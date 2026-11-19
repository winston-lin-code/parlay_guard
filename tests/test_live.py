"""Tests for the current dynamic FastAPI live dashboard (not the archived UI)."""
from fastapi.testclient import TestClient
import src.api as api


def test_dashboard_live_reload(tmp_path, monkeypatch):
    page = tmp_path / 'live.html'
    page.write_text('<!doctype html><html><body><div id="new-live">Live V1</div></body></html>', encoding='utf-8')
    monkeypatch.setattr(api, 'LIVE', page)
    client = TestClient(api.app)

    result = client.get('/')
    assert result.status_code == 200
    assert 'no-store' in result.headers['cache-control']
    assert 'Live V1' in result.text
    assert 'new-live' in client.get('/strategy').text

    page.write_text('<html><body>Live V2</body></html>', encoding='utf-8')
    assert 'Live V2' in client.get('/').text
    assert client.get('/api/health').json()['live_dashboard_available'] is True
    assert client.get('/api/games').status_code == 200
    assert isinstance(client.get('/api/games').json()['games'], list)
    assert client.get('/api/live/data').status_code == 200
    assert 'refresh_status' in client.get('/api/live/data').json()
    # The read-only dashboard router should not be lost during app construction.
    assert '/api/live/quotes' in client.get('/openapi.json').json()['paths']

def test_missing_report_recovers(tmp_path, monkeypatch):
    page = tmp_path / 'missing_live.html'
    monkeypatch.setattr(api, 'LIVE', page)
    client = TestClient(api.app)

    assert client.get('/').status_code == 503
    assert client.get('/api/health').json()['live_dashboard_available'] is False

    page.write_text('<html><body>Available!</body></html>', encoding='utf-8')
    assert client.get('/').status_code == 200
    assert 'Available!' in client.get('/').text
    assert client.get('/api/health').json()['live_dashboard_available'] is True
