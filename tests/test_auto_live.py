from __future__ import annotations
import os
import sys
from pathlib import Path
from datetime import datetime,timedelta,timezone
import numpy as np
import pandas as pd

sys.path.insert(0,str(Path(__file__).parent))
from game_lines_live import feature_training,fit_line_models,predict_line
import types
from src.auto_live import match_team,choose_market_line,moneyline_quotes,assemble_data,eligible_props

def test_names():
    assert match_team('Golden State Warriors','gsw','mem')=='gsw'
    assert match_team('Memphis Grizzlies','gsw','mem')=='mem'
    assert match_team('Yes','gsw','mem') is None

def test_market():
    rows=[{'sportsMarketType':'spreads','question':'Spread: Grizzlies (-4.5)','outcomePrices':'["0.54","0.46"]'},
          {'sportsMarketType':'spreads','question':'Spread: Grizzlies (-5.5)','outcomePrices':'["0.80","0.20"]'},
          {'sportsMarketType':'totals','question':'Warriors vs. Grizzlies: O/U 222.5','outcomePrices':'["0.48","0.52"]'}]
    assert choose_market_line(rows,'spreads')['line']==-4.5
    assert choose_market_line(rows,'totals')['line']==222.5

class Response:
    def __init__(self,output):self.output=output
    def raise_for_status(self):pass
    def json(self):return self.output
class Session:
    def post(self,url,json=None,timeout=None):
        assert '/books' in url
        return Response([{'asset_id':x['token_id'],'asks':[{'price':'0.56','size':'100'}]} for x in json])

def test_moneyline():
    events=[{'slug':'nba-gsw-mem-2026-11-19','home':'MEM','away':'GSW',
      'markets':[{'sportsMarketType':'moneyline','outcomes':'["Warriors","Grizzlies"]',
                  'clobTokenIds':'["w","m"]'}]}]
    ret=moneyline_quotes(Session(),events)
    assert ret[events[0]['slug']]['GSW']['ask']==.56
    assert ret[events[0]['slug']]['MEM']['shares']==100


def test_winner_filter(tmp_path):
    date=pd.Timestamp.now(tz='UTC')
    timestamp=(date+pd.Timedelta(hours=3)).isoformat()
    df=pd.DataFrame([{'event_slug':'nba-gsw-mem-2026-11-19','player_name':'Player G','side':'OVER',
        'line':8.5,'stat':'PTS','ask_price':.45,'stressed_ev_roi':.25,'game_start_utc':timestamp,
        'observed_utc':date.isoformat(),'model_probability':.70},
        {'event_slug':'nba-gsw-mem-2026-11-19','player_name':'Player M','side':'UNDER',
        'line':5.5,'stat':'REB','ask_price':.47,'stressed_ev_roi':.30,'game_start_utc':timestamp,
        'observed_utc':date.isoformat(),'model_probability':.75}])
    shortlist=tmp_path/'shortlist.csv';df.to_csv(shortlist,index=False)
    features=tmp_path/'features.csv'
    pd.DataFrame([{'event_slug':'nba-gsw-mem-2026-11-19','player_name':'Player G','team':'GSW'},
                  {'event_slug':'nba-gsw-mem-2026-11-19','player_name':'Player M','team':'MEM'}]).to_csv(features,index=False)
    picks=eligible_props(shortlist,features,[{'slug':'nba-gsw-mem-2026-11-19'}],date.to_pydatetime())
    assert len(picks['nba-gsw-mem-2026-11-19'])==2
    event={'slug':'nba-gsw-mem-2026-11-19','home':'MEM','away':'GSW','game_date':'2026-11-19',
        'tipoff_utc':timestamp,'event':'Warriors vs. Grizzlies','markets':[]}
    winner={('2026-11-19','GSW','MEM'):{'predicted_winner':'MEM','predicted_winner_probability':.65}}
    data=assemble_data([event],winner,None,None,{},picks,{},date.to_pydatetime(),[])
    assert len(data['games'][0]['selected_props'])==1
    assert data['games'][0]['selected_props'][0]['player']=='Player M'
    data2=assemble_data([event],{},None,None,{},picks,{},date.to_pydatetime(),[])
    assert data2['games'][0]['selected_props']==[]


def test_line_model():
    teams=['BOS','NYK','GSW','LAL','MEM','DEN']
    rng=np.random.default_rng(44)
    rows=[]
    start=pd.Timestamp('2023-10-01')
    for i in range(1100):
        d=start+pd.Timedelta(days=i//4)
        h=teams[i%6];a=teams[(i*3+1)%6]
        if h==a:a=teams[(i*3+2)%6]
        hp=float(112+rng.normal(0,10));ap=float(108+rng.normal(0,10))
        rows.append({'GAME_ID':str(i),'GAME_DATE':d,'HOME_TEAM':h,'AWAY_TEAM':a,'HOME_PTS':hp,'AWAY_PTS':ap})
    data=pd.DataFrame(rows)
    model,history,meta=fit_line_models(data)
    assert meta['trained_games']>=400
    pred=predict_line(model,history,home='BOS',away='NYK',game_day='2026-12-01')
    assert pred['projected_total']>0
    assert pred['home_spread']==round(-pred['home_margin'],1)

if __name__=='__main__':
    import tempfile
    test_names();test_market();test_moneyline()
    with tempfile.TemporaryDirectory() as tmp:test_winner_filter(Path(tmp))
    test_line_model();print('all five offline tests passed')
