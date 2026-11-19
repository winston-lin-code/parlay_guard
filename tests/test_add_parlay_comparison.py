"""Tests for the current two-distinct-players-per-game comparison engine."""
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
