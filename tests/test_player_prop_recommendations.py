from datetime import datetime, timedelta
import json

import pandas as pd

from src import player_prop_recommendations as r


def test_points_line_parser():
    assert r.parse_points_line('Will Jordan Sample score 25+ points?', {}) == 24.5
    assert r.parse_points_line('Will Jordan Sample score over 24.5 points?',{}) == 24.5
    assert r.parse_points_line('Jordan Sample 1st quarter 5+ points?',{}) is None


def test_exact_player_matching():
    assert r.match_name_in_question('Jordan Sample','Will Jordan Sample score 25+ points?')
    assert not r.match_name_in_question('Jordan Sam','Will Jordan Sample score 25+ points?')


def test_candidate_discovery_and_skip_ambiguous_under():
    event={'markets':[
        {'question':'Will Jordan Sample score 25+ points?',
         'outcomes':json.dumps(['Yes','No']),
         'clobTokenIds':json.dumps(['yes1','no1']),
         'outcomePrices':json.dumps(['0.55','0.45']),
         'sportsMarketType':'player_points','id':'1','active':True},
        {'question':'Will Jordan Sample score under 24.5 points?',
         'outcomes':['Yes','No'],'clobTokenIds':['yes2','no2'],
         'sportsMarketType':'player_points','id':'2','active':True},
    ]}
    found=r.extract_prop_candidates(event,{'Jordan Sample':object()})
    assert len(found)==1
    assert found[0]['yes_token']=='yes1'
    assert found[0]['line']==24.5


def test_game_history_uses_only_prior_games(tmp_path):
    rows=[]
    start=datetime(2026,10,12)
    for i in range(27):
        rows.append({'GAME_ID':str(i), 'GAME_DATE':(start+timedelta(days=i)).date(),
        'PLAYER_NAME':'Jordan Sample', 'TEAM_ABBREVIATION':'BOS',
        'PTS':20+i%10,'WL':'W' if i%2 else 'L','MIN':30})
    rows.append({'GAME_ID':'999','GAME_DATE':'2026-11-18','PLAYER_NAME':'Jordan Sample',
                 'TEAM_ABBREVIATION':'BOS','PTS':70,'WL':'W','MIN':30})
    csv=tmp_path/'sample.csv'
    pd.DataFrame(rows).to_csv(csv,index=False)
    hist,reason=r.game_player_history({'away':'BOS','home':'DEN','date':'2026-11-18'},csv)
    assert reason is None
    assert len(hist['Jordan Sample'])==27
    pred=r.estimate(hist['Jordan Sample'],24.5,True)
    assert 0<=pred['over']<=1
    assert 0<=pred['over_given_result']<=1


def test_moneyline_value_filter_does_not_force_pick(monkeypatch):
    event={
        'title':'Celtics vs Nuggets','slug':'nba-bos-den-2026-11-18',
        'markets':[{'question':'Celtics vs Nuggets','outcomes':['Celtics','Nuggets'],
                    'clobTokenIds':['tokenBOS','tokenDEN']}]
    }
    monkeypatch.setattr(r,'book_quote',lambda token:{'ask':.8 if token=='tokenBOS' else .8})
    best,detail=r.moneyline_recommendation({'away':'BOS','home':'DEN',
        'date':'2026-11-18','home_v1':.55,'home_v2':.60},event)
    assert best is None
    assert detail['predicted_winner']=='DEN'


def test_full_analysis_combined_is_modeled_not_executable(monkeypatch):
    dates=pd.date_range('2026-09-25',periods=40,freq='D')
    rows=pd.DataFrame({
        'GAME_DATE':dates,'PLAYER_NAME':['Jordan Sample']*40,
        'TEAM_ABBREVIATION':['BOS']*40,
        'PTS':[28]*28+[19]*12,'WL':['W' if i%2==0 else 'L' for i in range(40)]})
    event={
        'title':'Celtics vs Nuggets','slug':'nba-bos-den-2026-11-18',
        'markets':[
            {'id':'ml','question':'Celtics vs Nuggets',
             'outcomes':['Celtics','Nuggets'],'clobTokenIds':['bos','den']},
            {'id':'points','question':'Will Jordan Sample score 25+ points?',
             'outcomes':['Yes','No'],'clobTokenIds':['yespoints','nopoints'],
             'outcomePrices':['0.50','0.50'],'sportsMarketType':'player_points'}]
    }
    monkeypatch.setattr(r,'_event',lambda g:(event,None))
    monkeypatch.setattr(r,'game_player_history',lambda g:({'Jordan Sample':rows},None))
    prices={'bos':(.35,.34,80), 'den':(.70,.69,80),
        'yespoints':(.40,.38,200), 'nopoints':(.68,.67,200)}
    def book(token):
        ask,bid,depth=prices[token]
        return {'ask':ask,'bid':bid,'ask_dollars':depth,'spread':round(ask-bid,4)}
    monkeypatch.setattr(r,'book_quote',book)
    result=r.analyze_game({'away':'BOS','home':'DEN','date':'2026-11-18',
                            'home_v1':.40,'home_v2':.42})
    assert result['status']=='ok'
    assert result['best_moneyline']['team']=='BOS'
    assert result['best_prop']['player']=='Jordan Sample'
    assert 0<result['combined']['estimated_joint_probability']<1
    assert result['combined']['actual_combo_quote'] is None
    assert result['combined']['combined_estimated_roi'] is None
    assert result['combined']['combined_payout_multiplier'] is None
