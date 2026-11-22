"""Update legacy Parlay Guard tests for the current live API/comparison interface.

Run from the repository root:
    python path/to/fix_parlay_guard_tests.py

Backups are created in a timestamped folder alongside the repository.
This script never modifies or removes the data/ or models/ directories.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
import shutil

ROOT = Path.cwd()
if not (ROOT / '.git').exists() or not (ROOT / 'src' / 'api.py').is_file():
    raise SystemExit('Run from the parlay_guard Git repository root.')

COMPARISON_TEST = '''"""Tests for the current two-distinct-players-per-game comparison engine."""
import tempfile
import unittest
from datetime import datetime, timezone, timedelta
from pathlib import Path
import pandas as pd
import src.add_parlay_comparison as impl


class FakeResponse:
    def __init__(self, body):
        self.body = body
    def raise_for_status(self):
        pass
    def json(self):
        return self.body


class FakeSession:
    def get(self, url, timeout=None):
        slug = url.split('/')[-1]
        a, b = 'Team ' + slug[-1], 'Other ' + slug[-1]
        return FakeResponse({'slug': slug, 'title': slug, 'markets': [
            {'sportsMarketType': 'moneyline',
             'outcomes': f'["{a}","{b}"]',
             'clobTokenIds': f'["{slug}1","{slug}2"]',
             'active': True}
        ]})
    def post(self, url, json=None, timeout=None):
        return FakeResponse([
            {'asset_id': obj['token_id'], 'asks': [
                {'price': '.63' if obj['token_id'].endswith('1') else '.42', 'size': '300'}
            ]} for obj in json
        ])


class ComparisonTests(unittest.TestCase):
    def sample(self, path):
        now = datetime.now(timezone.utc)
        items = []
        # Two distinct players in each of two games: required by the current algorithm.
        for game in range(2):
            for index in range(2):
                i = game * 2 + index
                items.append({
                    'event': 'Game ' + str(game),
                    'event_slug': 'nba-match-' + str(game),
                    'player_name': 'Player ' + str(i),
                    'stat': 'REB', 'side': 'UNDER', 'line': 5.5 + i,
                    'ask_price': [.49, .50, .40, .46][i],
                    'stressed_ev_roi': .4 - .05 * i,
                    'observed_utc': (now - timedelta(minutes=5)).isoformat(),
                    'game_start_utc': (now + timedelta(hours=3)).isoformat(),
                    'outcome_mapping': 'description_yes_over',
                    'book_status': 'priced'
                })
        pd.DataFrame(items).to_csv(path, index=False)

    def test_expected_moneyline_and_parlay_odds(self):
        with tempfile.TemporaryDirectory() as d:
            csv = Path(d) / 'shortlist.csv'
            self.sample(csv)
            legs, stats = impl.load_legs(csv, 90)
            self.assertEqual(stats['eligible_rows'], 4)
            self.assertEqual(stats['selected_games'], 2)
            self.assertEqual(len(legs), 4)
            quotes = impl.fetch_moneylines(
                [x['event_slug'] for x in legs], session=FakeSession()
            )
            self.assertEqual(quotes['nba-match-0']['sides'][0]['ask'], .63)
            self.assertEqual(impl.american(.63), '-170')
            tickets = impl.build_tickets(legs)
            self.assertEqual(len(tickets['singles']), 4)
            self.assertEqual(len(tickets['same_game']), 2)
            self.assertEqual(len(tickets['cross_game']), 2)
            self.assertAlmostEqual(tickets['same_game'][0]['cost'], .49 * .50)
            moneylines = impl.benchmark_moneylines(legs, quotes)
            fragment = impl.html_fragment(legs, tickets, moneylines, 20, stats, 'example')
            self.assertIn('Separate same-game parlays', fragment)
            self.assertIn('Separate cross-game parlays', fragment)
            self.assertIn('not executable Polymarket Combo quotes', fragment)

    def test_stale_snapshot_and_started_games_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            csv = Path(d) / 'shortlist.csv'
            self.sample(csv)
            df = pd.read_csv(csv)
            # Both players from the first game are ineligible; game two survives.
            df.loc[0, 'observed_utc'] = (
                datetime.now(timezone.utc) - timedelta(hours=4)
            ).isoformat()
            df.loc[1, 'game_start_utc'] = (
                datetime.now(timezone.utc) - timedelta(minutes=1)
            ).isoformat()
            df.to_csv(csv, index=False)
            legs, stats = impl.load_legs(csv, 90)
            self.assertEqual(len(legs), 2)
            self.assertEqual(stats['selected_games'], 1)
            self.assertEqual({x['event_slug'] for x in legs}, {'nba-match-1'})

    def test_patch_existing_report_idempotently(self):
        original = '<!doctype html><html><head><style>p{color:#fff}</style></head><body><h1>Existing dashboard</h1><div class="live-toolbar">toolbar</div></body></html>'
        piece = f'{impl.BEGIN}<section id="pg-parlay-comparison">Hi</section>{impl.END}'
        once = impl.patch_dashboard(original, piece)
        twice = impl.patch_dashboard(once, piece)
        self.assertEqual(once, twice)
        self.assertEqual(twice.count(impl.BEGIN), 1)
        self.assertEqual(twice.count(impl.JS_BEGIN), 1)
        self.assertIn('Existing dashboard', twice)

    def test_missing_moneyline_is_explicit(self):
        stats = {'eligible_rows': 0, 'eligible_games_with_pair': 0,
                 'selected_games': 0, 'excluded_single_pick_games': 0}
        tickets = impl.build_tickets([])
        fragment = impl.html_fragment([], tickets, [], 20, stats, 'example')
        self.assertIn('Moneyline CLOB asks unavailable', fragment)
        self.assertIn('No games have two distinct', fragment)


if __name__ == '__main__':
    unittest.main()
'''

LIVE_TEST = '''"""Tests for the current dynamic FastAPI live dashboard (not the archived UI)."""
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
    assert '/api/live/quotes' in {route.path for route in api.app.routes}


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
'''

backup = ROOT.parent / ('parlay_guard_test_fix_backup_' + datetime.now().strftime('%Y%m%d_%H%M%S'))
backup.mkdir(parents=True, exist_ok=False)
for rel in ('src/api.py', 'tests/test_add_parlay_comparison.py', 'tests/test_live.py', 'requirements.txt'):
    path = ROOT / rel
    if not path.is_file():
        raise SystemExit(f'Missing expected file: {path}')
    target = backup / rel
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, target)

api_file = ROOT / 'src/api.py'
api_text = api_file.read_text(encoding='utf-8')
old_first = "app = FastAPI(title='Parlay Guard', version='0.1.0')\nfrom .live_dashboard_api import router as live_dashboard_router\napp.include_router(live_dashboard_router)"
new_first = "from .live_dashboard_api import router as live_dashboard_router"
if old_first in api_text:
    api_text = api_text.replace(old_first, new_first, 1)
base_init = "app=FastAPI(title='Parlay Guard Auto Live',version='0.4',lifespan=lifespan)"
if base_init not in api_text:
    raise SystemExit('API app initialization changed; did not modify src/api.py.')
if base_init + '\napp.include_router(live_dashboard_router)' not in api_text:
    api_text = api_text.replace(base_init, base_init + '\napp.include_router(live_dashboard_router)', 1)
api_file.write_text(api_text, encoding='utf-8')

(ROOT / 'tests/test_add_parlay_comparison.py').write_text(COMPARISON_TEST, encoding='utf-8')
(ROOT / 'tests/test_live.py').write_text(LIVE_TEST, encoding='utf-8')
req_file = ROOT / 'requirements.txt'
req = req_file.read_text(encoding='utf-8')
if not any(line.strip().lower().startswith('httpx') for line in req.splitlines()):
    req_file.write_text(req.rstrip() + '\nhttpx>=0.27,<1\n', encoding='utf-8')

print('Updated: src/api.py, tests/test_add_parlay_comparison.py, tests/test_live.py')
print('Ensured httpx test dependency in requirements.txt')
print('Backups:', backup)
print('Next: python -m pytest tests/ -v --tb=short')
