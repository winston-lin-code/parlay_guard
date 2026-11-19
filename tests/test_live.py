import importlib.util
from pathlib import Path
from fastapi.testclient import TestClient

SOURCE = Path(__file__).resolve().parents[1] / 'src' / 'api.py'
spec = importlib.util.spec_from_file_location('parlay_guard_dev_api', SOURCE)
api = importlib.util.module_from_spec(spec)
spec.loader.exec_module(api)


def test_dashboard_live_reload(tmp_path):
    api.REPORT = tmp_path / 'models' / 'parlay_guard_v1_vs_v2.html'
    api.STRATEGY_REPORT = tmp_path / 'models' / 'parlay_comparison.html'
    api.REPORT.parent.mkdir()
    api.REPORT.write_text('<!doctype html><html><body><div id="legacy">hi</div><script>window.value=1</script></body></html>', encoding='utf-8')
    api.STRATEGY_REPORT.write_text('<html><body><div id="strategy">Lab</div></body></html>', encoding='utf-8')
    client = TestClient(api.app)

    page = client.get('/')
    assert page.status_code == 200
    assert 'no-store' in page.headers['cache-control']
    assert 'id="legacy"' in page.text
    assert 'window.value=1' in page.text
    assert 'parlay-guard-dev-live-reload' in page.text
    assert '/api/dashboard-version?view=dashboard' in page.text

    v1 = client.get('/api/dashboard-version').json()['version']
    api.REPORT.write_text('<html><body><div>updated HTML, now longer!</div></body></html>', encoding='utf-8')
    v2 = client.get('/api/dashboard-version').json()['version']
    assert v1 != v2
    assert 'updated HTML' in client.get('/').text
    assert 'id="strategy"' in client.get('/strategy').text
    assert 'no-store' in client.get('/api/dashboard-version').headers['cache-control']
    assert client.get('/api/health').json()['strategy_available']
    assert client.get('/api/games').status_code == 501


def test_missing_report_recovers(tmp_path):
    api.REPORT = tmp_path / 'missing_main.html'
    api.STRATEGY_REPORT = tmp_path / 'missing_strategy.html'
    client = TestClient(api.app)
    before = client.get('/api/dashboard-version').json()['version']
    assert 'Parlay Guard is running' in client.get('/').text
    api.STRATEGY_REPORT.write_text('<html><body>Available!</body></html>', encoding='utf-8')
    after = client.get('/api/dashboard-version').json()['version']
    assert before != after
    assert 'Available!' in client.get('/').text
