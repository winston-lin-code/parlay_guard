from __future__ import annotations
import json
from datetime import datetime,timedelta,timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd
import pytest

from src.college_train import parse_schedule,training_rows,snapshot_features,initial_team,train,load_model,predict,FEATURES
from src.college_sources import canonical,outcome_team,parse_espn,match_markets,check_team,_injury_items,_roster_athletes,manual_report
from src.college_api import score_game
from install_college_guard import update_api,update_live

T=datetime(2026,11,20,20,tzinfo=timezone.utc)


def school(id,name,abbr=None):
    loc='North Carolina' if name.startswith('North Carolina') else 'Duke' if name.startswith('Duke') else name.rsplit(' ',1)[0]
    nick=name[len(loc):].strip()
    return {'id':str(id),'display_name':name,'short_name':loc,
            'location':loc,'name':nick,
            'abbr':abbr or '', 'logo':''}


def game():
    return {'sport':'mbb','espn_id':'123456','home':school(1,'Duke Blue Devils'),
            'away':school(2,'North Carolina Tar Heels'),
            'start_utc':(T+timedelta(hours=3)).isoformat(),'neutral':False,
            'season':2027,'name':'North Carolina @ Duke'}


def rec(i,home,away,w,season=2026):
    return {'game_id':str(i),'date':(T-timedelta(days=120-i)).isoformat(),
            'season':season,'home_id':str(home),'away_id':str(away),'home_win':w,'neutral':False}


def test_parse_mbb_real_public_schema_no_season_field():
    df=pd.DataFrame([{'id':101,'status_type_name':'STATUS_FINAL','status_type_completed':True,
        'game_date_time':'2025-12-15T01:30:00+00:00','home_id':1,'away_id':2,'home_score':81,
        'away_score':65,'neutral_site':False,'home_non_div1_team':None,'away_non_div1_team':None},
        {'id':102,'status_type_name':'STATUS_SCHEDULED','status_type_completed':False,
         'game_date_time':'2025-12-16T01:30:00+00:00','home_id':1,'away_id':2,
         'home_score':None,'away_score':None}])
    games=parse_schedule(df,'mbb',now=T,season_hint=2026)
    assert len(games)==1 and games[0]['season']==2026 and games[0]['home_win']==1


def test_mbb_season_fallback_derived_from_fall():
    df=pd.DataFrame([{'id':11,'status_type_name':'STATUS_FINAL','status_type_completed':True,
        'start_date':'2025-11-20T00:30:00+00:00','home_id':'1','away_id':'2','home_score':81,'away_score':65}])
    g=parse_schedule(df,'mbb',now=T)
    assert g[0]['season']==2026


def test_parse_cfb_excludes_fcs_and_future():
    df=pd.DataFrame([{'game_id':i,'season':2025,'start_date':day,'home_id':1,'away_id':2,
        'home_points':30,'away_points':17,'completed':True,
        'home_division':'fbs','away_division':div,'neutral_site':False}
        for i,day,div in [(1,'2025-09-20T19:00:00Z','fbs'),(2,'2025-09-21T19:00:00Z','fcs'),(3,'2028-01-01T19:00:00Z','fbs')]])
    assert [g['game_id'] for g in parse_schedule(df,'cfb',now=T)]==['1']


def test_pregame_features_precede_result_update():
    first,second=rec(1,1,2,1),rec(2,1,2,0)
    x,y,states,dates=training_rows([first,second],'mbb')
    assert x[0][0]==0.0 and x[1][0]>0.0
    assert y==[1,0] and states['1']['games']==2


def test_season_reset_and_no_inference_mutation():
    state={'1':{**initial_team(),'elo':1700,'season':2026,'games':30,'wins':[1]*10},'2':initial_team()}
    original=state['1']['elo']
    f,_,_=snapshot_features(state,'1','2',T.isoformat(),2027,False)
    assert f[0]==(1630-1500)/400 and state['1']['elo']==original


def test_fit_chronological_realized_model_and_load(tmp_path):
    records=[rec(i,i%6,(i+1)%6,int(i%5!=0)) for i in range(1,82)]
    metrics=train('cfb',records,model_path=tmp_path/'cfb.joblib',min_games=35)
    bundle=load_model('cfb',path=tmp_path/'cfb.joblib')
    assert metrics['holdout_games']>0 and metrics['train_games']>0
    assert bundle['metrics']==metrics
    assert 0<predict(bundle,'1','2',T.isoformat(),2026)<1


def test_ambiguous_team_name_never_assigned():
    g=game()
    assert outcome_team('Duke Blue Devils',g['home'],g['away'])=='1'
    assert outcome_team('North Carolina',g['home'],g['away'])=='2'
    assert outcome_team('Blue Devils of a different school',g['home'],g['away']) is None
    assert outcome_team('UNC',g['home'],g['away']) is None


def test_parse_future_scoreboard_and_logos():
    raw={'events':[{'id':'123456','date':(T+timedelta(hours=3)).isoformat(),
        'status':{'type':{'state':'pre'}},'season':{'year':2027},
        'competitions':[{'neutralSite':False,'competitors':[
          {'homeAway':'home','team':{'id':'1','displayName':'Duke Blue Devils','location':'Duke','name':'Blue Devils','logos':[{'href':'https://a.espncdn.com/combiner/i?x'}]}},
          {'homeAway':'away','team':{'id':'2','displayName':'North Carolina Tar Heels','location':'North Carolina','name':'Tar Heels','logos':[{'href':'https://bad.com/track'}]}}]}]}]}
    gs=parse_espn(raw,'mbb','2026-11-20',T)
    assert gs[0]['home']['logo'].startswith('https://a.espncdn.com/')
    assert not gs[0]['away']['logo']
    assert parse_espn(raw,'mbb','2026-11-21',T)==[]


class Response:
    def __init__(self,obj):self.obj=obj
    def raise_for_status(self):pass
    def json(self):return self.obj


class Session:
    def __init__(self,when, reversed_team=False):
        outcomes=['Duke Blue Devils','North Carolina Tar Heels']
        if reversed_team:outcomes=['Duke','UNC']
        self.when=when
        self.book=[{'asset_id':'A','asks':[{'price':'.65','size':'60'},{'price':'.62','size':'20'}]},
                   {'asset_id':'B','asks':[{'price':'.39','size':'90'}]}]
        self.events=[{'gameStartTime':when,'markets':[{'sportsMarketType':'moneyline',
                'outcomes':json.dumps(outcomes),'clobTokenIds':json.dumps(['A','B'])}]}]
    def get(self,url,**kw):return Response(self.events)
    def post(self,url,**kw):return Response(self.book)


def test_book_matching_best_ask_and_team_ids():
    g=game();q=match_markets(Session(g['start_utc']),'mbb',[g],T,pages=1)
    assert q[g['espn_id']]['1']['ask']==.62
    assert q[g['espn_id']]['2']['ask']==.39


def test_no_fuzzy_or_ambiguous_market_mapping():
    g=game();q=match_markets(Session(g['start_utc'],reversed_team=True),'mbb',[g],T,pages=1)
    assert not q[g['espn_id']]


def stub_model():
    return {'sport':'mbb','team_states':{'1':initial_team(),'2':initial_team()},
            'metrics':{'vs_constant_baseline':'better'},'trained_utc':T.isoformat(),
            'last_game_utc':(T-timedelta(days=2)).isoformat()}


def quotes():
    return {'1':{'ask':.55,'shares':100,'token_id':'A','observed_utc':T.isoformat()},
            '2':{'ask':.6,'shares':100,'token_id':'B','observed_utc':T.isoformat()}}


def test_ev_filters_negative_side_and_marks_unverified():
    with patch('src.college_api.predict',return_value=.70):
        r=score_game(game(),stub_model(),quotes(),{},T)
    assert r['best_pick']['team_name']=='Duke Blue Devils'
    assert r['best_pick']['estimated_ev']>0.2
    assert r['market']['away']['estimated_ev']<0
    assert r['status']=='POSITIVE_ESTIMATED_EV_UNVERIFIED_INJURIES'


def test_injury_report_requires_review():
    reports={'2':{'injuries':[{'player':'Example','status':'out'}],'roster_changes':[],
                   'injury_status':'reported','roster_status':'unchanged_listing'}}
    with patch('src.college_api.predict',return_value=.70):
        r=score_game(game(),stub_model(),quotes(),reports,T)
    assert r['best_pick'] and r['status']=='ROSTER_INJURY_REVIEW'


def test_missing_model_wont_invent_probabilities():
    r=score_game(game(),None,quotes(),{},T)
    assert r['predictions'] is None and r['best_pick'] is None


def test_invalid_quote_or_stale_quote_zero_ev():
    q=quotes();q['1']['observed_utc']=(T-timedelta(minutes=30)).isoformat()
    q['2']['ask']=1.01
    with patch('src.college_api.predict',return_value=.70):
        r=score_game(game(),stub_model(),q,{},T)
    assert r['best_pick'] is None and r['status']=='NO_MATCHED_LIVE_ASK'


def test_regression_bad_metrics_stops_positive_label():
    model=stub_model();model['metrics']['vs_constant_baseline']='worse'
    with patch('src.college_api.predict',return_value=.70):
        r=score_game(game(),model,quotes(),{},T)
    assert r['status']=='UNVALIDATED_MODEL'


def test_parse_roster_and_injury_entries():
    roster={'athletes':[{'position':'G','items':[{'id':'100','fullName':'Player A'}]},{'id':'200','displayName':'Player B'}]}
    assert len(_roster_athletes(roster))==2
    assert _injury_items({'injuries':[{'athlete':{'displayName':'Player A'},'status':{'description':'Questionable'}}]})[0]['status']=='Questionable'


def test_roster_listing_differences_not_called_injuries(tmp_path):
    class Client:
        def get(self,url,**kw):
            if url.endswith('injuries'):raise __import__('requests').HTTPError('No injury API')
            return Response({'athletes':[{'id':'100','displayName':'Player A'}]})
    first=check_team('mbb','1',client=Client(),root=tmp_path,now=T)
    second=check_team('mbb','1',client=Client(),root=tmp_path,now=T+timedelta(minutes=1))
    assert first['roster_status']=='baseline_recorded'
    assert second['roster_status']=='unchanged_listing'
    assert second['injury_status']=='unverified' and not second['injuries']


def test_roster_change_is_reported_as_listing_not_injury(tmp_path):
    class Client:
        def __init__(self):self.i=0
        def get(self,url,**kw):
            if url.endswith('/injuries'):raise __import__('requests').HTTPError('N/A')
            self.i+=1
            return Response({'athletes':[{'id':str(self.i),'displayName':'Player '+str(self.i)}]})
    client=Client()
    check_team('cfb','1',client=client,root=tmp_path,now=T)
    after=check_team('cfb','1',client=client,root=tmp_path,now=T)
    assert after['roster_status']=='changed' and len(after['roster_changes'])==2
    assert not after['injuries']


def test_manual_reports_require_recent_source_url(tmp_path):
    folder=tmp_path/'data'/'college_guard';folder.mkdir(parents=True)
    (folder/'roster_reports.csv').write_text('sport,team_id,player,status,checked_at_utc,source_url\n'
      f'mbb,1,A,Out,{T.isoformat()},https://example.edu/report\n'
      f'mbb,1,B,Out,{T.isoformat()},http://not-secure\n',encoding='utf8')
    assert len(manual_report('mbb','1',root=tmp_path,now=T+timedelta(minutes=10)))==1
    assert not manual_report('mbb','1',root=tmp_path,now=T+timedelta(hours=37))


def test_installer_idempotent_and_compatible_with_extra_routes():
    api='from fastapi import FastAPI\napp=FastAPI()\n@app.get("/")\ndef x():return 1\n'
    patched,yes=update_api(api)
    assert yes and '/api/college' not in patched and 'college_guard_router' in patched
    assert update_api(patched)==(patched,False)
    html='<html><body><header>test</header><section>other section</section><script>test();</script></body></html>'
    modified,yes=update_live(html)
    assert yes and '/college-preview.js' in modified
    assert update_live(modified)==(modified,False)


def test_installer_missing_anchor_fails_closed():
    with pytest.raises(RuntimeError):update_api('app = 1')
    with pytest.raises(RuntimeError):update_live('<html><body>No header</body></html>')


def test_unknown_college_team_never_gets_untrained_elo_pick():
    missing=game();missing['away']['id']='999999'
    with patch('src.college_api.predict') as mock_predict:
        result=score_game(missing,stub_model(),quotes(),{},T)
    mock_predict.assert_not_called()
    assert result['status']=='UNKNOWN_TEAM_MODEL'
    assert result['best_pick'] is None


def test_book_tipoff_mismatch_no_cross_game_join():
    g=game()
    far_start=(T+timedelta(days=1)).isoformat()
    got=match_markets(Session(far_start),'mbb',[g],T,pages=1)
    assert not got[g['espn_id']]


def test_incomplete_injury_coverage_explicit_not_checked():
    with patch('src.college_api.predict',return_value=.70):
        r=score_game(game(),stub_model(),quotes(),{},T)
    assert all(s['injury_status']=='not_checked' for s in r['injuries'].values())
    assert r['status']=='POSITIVE_ESTIMATED_EV_UNVERIFIED_INJURIES'


def test_fee_allowance_reduces_estimated_ev():
    with patch('src.college_api.predict',return_value=.70):
        low=score_game(game(),stub_model(),quotes(),{},T,fee_buffer=0)
        high=score_game(game(),stub_model(),quotes(),{},T,fee_buffer=.04)
    assert high['market']['home']['estimated_ev'] < low['market']['home']['estimated_ev']


def test_roster_changes_persist_after_next_refresh(tmp_path):
    class Client:
        def __init__(self):self.i=0
        def get(self,url,**kw):
            if url.endswith('/injuries'):raise __import__('requests').HTTPError('n/a')
            self.i+=1
            value='1' if self.i==1 else '2'
            return Response({'athletes':[{'id':value,'displayName':'Player '+value}]})
    client=Client()
    check_team('cfb','1',client=client,root=tmp_path,now=T)
    check_team('cfb','1',client=client,root=tmp_path,now=T+timedelta(minutes=10))
    later=check_team('cfb','1',client=client,root=tmp_path,now=T+timedelta(hours=1))
    assert later['roster_status']=='changed' and len(later['roster_changes'])==2
    expired=check_team('cfb','1',client=client,root=tmp_path,now=T+timedelta(days=8))
    assert expired['roster_status']=='unchanged_listing' and not expired['roster_changes']


def test_team_abbreviations_only_when_unique_in_matchup():
    h=school(1,'Ohio State Buckeyes',abbr='OSU')
    a=school(2,'Oregon State Beavers',abbr='OSU')
    assert outcome_team('OSU',h,a) is None
    a['abbr']='ORST'
    assert outcome_team('OSU',h,a)=='1'
