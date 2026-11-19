import tempfile
import unittest
from datetime import datetime, timezone, timedelta
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
import parlay_guard.src.add_parlay_comparison as impl


class FakeResponse:
    def __init__(self, body): self.body = body
    def raise_for_status(self): pass
    def json(self): return self.body


class FakeSession:
    def get(self, url, timeout=None):
        slug = url.split('/')[-1]
        a,b = ('Team '+slug[-1], 'Other '+slug[-1])
        return FakeResponse({'slug':slug, 'title':slug, 'markets':[
            {'sportsMarketType':'moneyline','outcomes':f'["{a}","{b}"]',
             'clobTokenIds':f'["{slug}1","{slug}2"]', 'active':True}
        ]})
    def post(self, url, json=None, timeout=None):
        return FakeResponse([{'asset_id':obj['token_id'],
                              'asks':[{'price':'.63' if obj['token_id'].endswith('1') else '.42', 'size':'300'}]}
                             for obj in json])


class ComparisonTests(unittest.TestCase):
    def sample(self, path):
        now=datetime.now(timezone.utc)
        items=[]
        for i in range(4):
            items.append({'event':'Game '+str(i),'event_slug':'nba-match-'+str(i),
                          'player_name':'Player '+str(i),'stat':'REB','side':'UNDER','line':5.5+i,
                          'ask_price':[.49,.50,.40,.46][i], 'stressed_ev_roi':.4-.05*i,
                          'observed_utc':(now-timedelta(minutes=5)).isoformat(),
                          'game_start_utc':(now+timedelta(hours=3)).isoformat(),
                          'outcome_mapping':'description_yes_over'})
        pd.DataFrame(items).to_csv(path,index=False)

    def test_expected_moneyline_and_parlay_odds(self):
        with tempfile.TemporaryDirectory() as d:
            csv=Path(d)/'shortlist.csv';self.sample(csv)
            legs,count=impl.load_legs(csv,90)
            self.assertEqual(count,4);self.assertEqual(len(legs),4)
            prices=impl.fetch_moneylines([x['event_slug'] for x in legs],session=FakeSession())
            self.assertEqual(prices['nba-match-0']['sides'][0]['ask'],.63)
            self.assertEqual(impl.amer(.63),'-170')
            html=impl.html_fragment(legs,prices,20,count_valid=count,fetch_time='example')
            self.assertIn('2-leg cross-game prop parlay',html)
            self.assertIn('3-leg cross-game prop parlay',html)
            self.assertIn('4-leg cross-game prop parlay',html)
            self.assertIn('data-pg-cost="0.24500000"',html)
            self.assertIn('NOT Polymarket Combo quotes',html)

    def test_stale_snapshot_and_started_games_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            csv=Path(d)/'shortlist.csv';self.sample(csv)
            df=pd.read_csv(csv)
            df.loc[0,'observed_utc']=(datetime.now(timezone.utc)-timedelta(hours=4)).isoformat()
            df.loc[1,'game_start_utc']=(datetime.now(timezone.utc)-timedelta(minutes=1)).isoformat()
            df.to_csv(csv,index=False)
            legs,count=impl.load_legs(csv,90)
            self.assertEqual(len(legs),2)

    def test_patch_existing_report_idempotently(self):
        original='<!doctype html><html><head><style>p{color:#fff}</style></head><body><h1>Existing dashboard</h1><div class="live-toolbar">toolbar</div></body></html>'
        piece=f'{impl.BEGIN}<section id="pg-parlay-comparison">Hi</section>{impl.END}'
        once=impl.patch_dashboard(original,piece)
        twice=impl.patch_dashboard(once,piece)
        self.assertEqual(once,twice)
        self.assertEqual(twice.count(impl.BEGIN),1)
        self.assertEqual(twice.count(impl.JS_BEGIN),1)
        self.assertIn('Existing dashboard',twice)

    def test_missing_moneyline_is_explicit(self):
        fragment=impl.html_fragment([{'event':'G','event_slug':'slug','player':'A','side':'OVER','line':9.5,'stat':'PTS','ask':.5,'stressed_roi':.2}],{},20,count_valid=1,fetch_time='example')
        self.assertIn('Moneyline quote unavailable',fragment)
        self.assertIn('Fewer than two qualifying picks',fragment)

if __name__ == '__main__':unittest.main()
