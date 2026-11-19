import sys
import tempfile
import unittest
import json
from datetime import datetime, timezone, timedelta
from pathlib import Path
import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import parlay_guard.src.add_parlay_comparison as ui
import parlay_guard.src.paper_strategy_results as paper

class FakeResponse:
    def __init__(self,value):self.value=value
    def raise_for_status(self):pass
    def json(self):return self.value

class FakeSession:
    def get(self,url,timeout=None):
        slug=url.split('/')[-1]
        return FakeResponse({'slug':slug,'title':slug,'markets':[
            {'sportsMarketType':'moneyline','outcomes':'["Warriors","Grizzlies"]',
             'clobTokenIds':f'["{slug}_1","{slug}_2"]','active':True}
        ]})
    def post(self,url,json=None,timeout=None):
        return FakeResponse([{'asset_id':x['token_id'],
                              'asks':[{'price':'.63' if x['token_id'].endswith('1') else '.42', 'size':'100'}]}
                             for x in json])

class StrategyTests(unittest.TestCase):
    def make_csv(self,path,now):
        games=[('nba-gsw-mem-2026-11-19','Warriors vs. Grizzlies'),
               ('nba-por-mil-2026-11-19','Blazers vs. Bucks'),
               ('nba-lac-sas-2026-11-19','Clippers vs. Spurs'),
               ('nba-nyk-min-2026-11-19','Knicks vs. Wolves')]
        rows=[]
        for n,(slug,event) in enumerate(games):
            for j in range(2 if n<3 else 1):
                rows.append({'event':event,'event_slug':slug,'player_name':f'Example Player {n*2+j}',
                             'stat':'PTS','side':'OVER' if j else 'UNDER',
                             'line':5.5+j,'ask_price':.4+.02*n+.07*j,
                             'stressed_ev_roi':.4-.04*n-.03*j,
                             'game_start_utc':(now+timedelta(hours=6)).isoformat(),
                             'observed_utc':(now-timedelta(minutes=5)).isoformat(),
                             'ask_size_shares':200,'model_probability':.7,
                             'outcome_mapping':'description_yes_over','book_status':'priced'})
        pd.DataFrame(rows).to_csv(path,index=False)
        return games

    def test_equal_exposure_and_pairing(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'shortlist.csv';now=datetime.now(timezone.utc)
            self.make_csv(p,now)
            legs,stats=ui.load_legs(p,90,now=now)
            self.assertEqual(len(legs),6)
            self.assertEqual(stats['eligible_games_with_pair'],3)
            t=ui.build_tickets(legs)
            self.assertEqual([len(t[x]) for x in ('singles','same_game','cross_game')],[6,3,3])
            for name in ('singles','same_game','cross_game'):
                collected=[x for ticket in t[name] for x in ticket['leg_ids']]
                self.assertEqual(set(collected),set(leg['id'] for leg in legs))
                self.assertEqual(len(collected),len(legs))
            for ticket in t['same_game']:
                self.assertEqual(len(ticket['events']),1)
            for ticket in t['cross_game']:
                self.assertEqual(len(ticket['events']),2)
            self.assertAlmostEqual(t['same_game'][0]['cost'],legs[0]['ask']*legs[1]['ask'])

    def test_freshness_and_no_pair(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'shortlist.csv';now=datetime.now(timezone.utc)
            self.make_csv(p,now)
            df=pd.read_csv(p)
            df.loc[1,'observed_utc']=(now-timedelta(hours=10)).isoformat()
            df.loc[2,'game_start_utc']=(now-timedelta(hours=1)).isoformat()
            df.to_csv(p,index=False)
            legs,stats=ui.load_legs(p,90,now=now)
            self.assertEqual(len(legs),2)
            self.assertEqual(stats['selected_games'],1)
            self.assertFalse(ui.build_tickets(legs)['cross_game'])

    def test_moneyline_and_dashboard_patch(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'shortlist.csv';now=datetime.now(timezone.utc)
            self.make_csv(p,now)
            legs,stats=ui.load_legs(p,90,now=now)
            t=ui.build_tickets(legs)
            ml=ui.benchmark_moneylines(legs,ui.fetch_moneylines(list({l['event_slug'] for l in legs}),session=FakeSession()))
            self.assertTrue(all(m['sides'][0]['ask']==.63 for m in ml))
            self.assertEqual(ui.american(.63),'-170')
            self.assertEqual(ui.american(.2),'+400')
            f=ui.html_fragment(legs,t,ml,20,stats,'example',90)
            self.assertIn('Separate same-game parlays',f)
            self.assertIn('Separate cross-game parlays',f)
            self.assertIn('Moneyline benchmark',f)
            self.assertIn('NOT',f.upper())
            self.assertIn('data-pg-leg-result',f)
            self.assertIn('data-pg-ticket-panel="cross_game"',f)
            original='<!doctype html><html><head><style>body{color:blue}</style></head><body><h1>Original</h1><div class="live-toolbar">bar</div></body></html>'
            first=ui.patch_dashboard(original,f)
            second=ui.patch_dashboard(first,f)
            self.assertEqual(first,second)
            self.assertEqual(first.count(ui.BEGIN),1)
            self.assertEqual(first.count(ui.JS_BEGIN),1)
            self.assertIn('Original',first)

    def test_paper_settlement_only_when_verified_data(self):
        snap_time=pd.Timestamp('2026-11-19T17:00:00Z')
        a={'id':'p1','event':'Warriors vs. Grizzlies','event_slug':'nba-gsw-mem-2026-11-19',
           'player':'Zach Edey','stat':'REB','side':'OVER','line':5.5,'ask':.4,'tipoff_utc':'2026-11-20T01:00:00Z'}
        b={'id':'p2','event':'Warriors vs. Grizzlies','event_slug':'nba-gsw-mem-2026-11-19',
           'player':'Jaylen Wells','stat':'PTS','side':'UNDER','line':9.5,'ask':.5,'tipoff_utc':'2026-11-20T01:00:00Z'}
        row=lambda p,team,stat,val: {'GAME_DATE':'2026-11-19','PLAYER_NAME':p,'TEAM_ID':paper.TEAM_IDS[team],
                                     'MIN':20,'PTS':val if stat=='PTS' else 7,'REB':val if stat=='REB' else 3,'AST':1,'FG3M':1}
        games=pd.DataFrame([row('Zach Edey','MEM','REB',7),row('Jaylen Wells','MEM','PTS',10)])
        games['_NAME']=games['PLAYER_NAME'].map(paper.clean)
        self.assertTrue(paper.settle_player(a,games,snap_time))
        self.assertFalse(paper.settle_player(b,games,snap_time))
        self.assertIsNone(paper.settle_player(a,games.iloc[0:0],snap_time))
        teams=pd.DataFrame([{'GAME_DATE':'2026-11-19','TEAM_ABBREVIATION':'GSW','GAME_ID':'2001','PTS':99,'WL':'L'},
                            {'GAME_DATE':'2026-11-19','TEAM_ABBREVIATION':'MEM','GAME_ID':'2001','PTS':106,'WL':'W'}])
        self.assertEqual(paper.game_winner(a['event_slug'],teams),'MEM')
        self.assertEqual(paper.map_team('Grizzlies'),'MEM')
        ticket={'id':'same_game_1','strategy':'same_game','leg_ids':['p1','p2'],'cost':.2}
        snapshot={'observed_utc':str(snap_time),'budget':20,'legs':[a,b],
                  'tickets':{'singles':[{'leg_ids':['p1'],'cost':.4},{'leg_ids':['p2'],'cost':.5}],
                             'same_game':[ticket],'cross_game':[]},
                  'moneylines':[{'slug':a['event_slug'],'chosen':1,'sides':[{'team':'Warriors','ask':.6}, {'team':'Grizzlies','ask':.45}]}]}
        rows,pending=paper.paper_session(snapshot,games,teams)
        lookup={x['strategy']:x for x in rows}
        self.assertEqual(len(pending),0)
        self.assertTrue(lookup['singles']['complete'])
        self.assertAlmostEqual(lookup['singles']['gross_payout'],25)
        self.assertEqual(lookup['same_game']['gross_payout'],0)
        self.assertEqual(lookup['moneyline']['won_tickets'],1)
        self.assertFalse(lookup['cross_game']['complete'])
        rows2,pending2=paper.paper_session(snapshot,games.iloc[0:0],None)
        self.assertFalse(rows2[0]['complete'])
        self.assertEqual(len(pending2),2)

if __name__=='__main__':unittest.main()
