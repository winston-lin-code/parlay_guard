"""Offline safeguards for REAL pregame historical market-line evaluation."""
from datetime import datetime, timedelta, timezone
from pathlib import Path
import json

import numpy as np
import pandas as pd
import pytest

from src.market_line_validation import (
    extract_snapshot, import_snapshots, _read_quotes, match_quotes,
    evaluate_market_history, _probability,
)


def example_envelope(*, snapshot='2025-11-19T22:00:00Z', start='2025-11-20T01:00:00Z'):
    return {
      'timestamp':snapshot, 'data':[{'sport_key':'basketball_nba', 'id':'game1',
      'home_team':'Boston Celtics', 'away_team':'New York Knicks',
      'commence_time':start, 'bookmakers':[{'key':'testbook','markets':[
        {'key':'spreads','last_update':snapshot,'outcomes':[
          {'name':'Boston Celtics','point':-5.5,'price':1.91},
          {'name':'New York Knicks','point':5.5,'price':1.91}]},
        {'key':'totals','last_update':snapshot,'outcomes':[
          {'name':'Over','point':220.5,'price':1.94},
          {'name':'Under','point':220.5,'price':1.91}]}]}]}]
    }


def test_extract_pregame_quotes_only_home_over_and_decimal():
    rows=extract_snapshot(example_envelope())
    assert len(rows)==2
    assert {r['market'] for r in rows}=={'spreads','totals'}
    assert {r['side'] for r in rows}=={'HOME','OVER'}
    assert rows[0]['game_date']=='2025-11-19' # game starts 8 PM ET
    assert rows[0]['point']==-5.5


def test_source_time_is_from_envelope_not_import_time():
    rows=extract_snapshot(example_envelope())
    assert rows[0]['snapshot_utc'].startswith('2025-11-19T22:00:00')
    assert extract_snapshot(example_envelope(snapshot='2025-11-20T02:00:00Z'))==[]


def test_reject_fake_snapshot_and_naive_timestamp():
    with pytest.raises(ValueError, match='historical envelope'):
        extract_snapshot({'data':[]})
    with pytest.raises(ValueError, match='timezone-aware'):
        extract_snapshot(example_envelope(snapshot='2025-11-19 22:00:00'))


def test_import_idempotent_duplicates(tmp_path):
    root=tmp_path/'odds';root.mkdir()
    for name in ('1.json','2.json'):
        (root/name).write_text(json.dumps(example_envelope()),encoding='utf-8')
    output=tmp_path/'out.csv'
    result=import_snapshots(root,output)
    assert result['quotes']==2 and result['duplicates']==2
    assert len(pd.read_csv(output))==2


def test_naive_csv_times_are_rejected(tmp_path):
    csv=tmp_path/'quotes.csv'
    row=extract_snapshot(example_envelope())[0]
    row['snapshot_utc']='2025-11-19 22:00:00'
    pd.DataFrame([row]).to_csv(csv,index=False)
    with pytest.raises(ValueError, match='explicit timezone'):
        _read_quotes(csv)


def test_select_pregame_match_by_date_and_teams_not_by_name_substring(tmp_path):
    game=pd.DataFrame([{'GAME_ID':'0001','GAME_DATE':'2025-11-19','HOME_TEAM':'BOS',
                        'AWAY_TEAM':'NYK','HOME_PTS':108,'AWAY_PTS':101}])
    before=example_envelope()
    rows=extract_snapshot(before)
    rows+=extract_snapshot(example_envelope(snapshot='2025-11-19T23:30:00Z'))
    # 90 minutes before start is eligible; but 30 minutes isn't.
    rows+=extract_snapshot(example_envelope(snapshot='2025-11-20T00:30:00Z'))
    quotes=pd.DataFrame(rows)
    for col in ('snapshot_utc','game_start_utc','market_update_utc'):
        quotes[col]=pd.to_datetime(quotes[col],utc=True)
    selected,metrics=match_quotes(game,quotes,bookmaker='testbook',lead_minutes=60)
    assert len(selected)==2
    assert all(selected.snapshot_utc.dt.strftime('%H:%M:%S')=='23:30:00')
    assert metrics['joined_quotes']==2


def test_wrong_bookmaker_and_other_team_never_join():
    game=pd.DataFrame([{'GAME_ID':'0001','GAME_DATE':'2025-11-19',
                        'HOME_TEAM':'LAL','AWAY_TEAM':'NYK'}])
    q=pd.DataFrame(extract_snapshot(example_envelope()))
    for col in ('snapshot_utc','game_start_utc','market_update_utc'):
        q[col]=pd.to_datetime(q[col],utc=True)
    selected,_=match_quotes(game,q,bookmaker='testbook')
    assert selected.empty
    selected,_=match_quotes(game,q,bookmaker='otherbook')
    assert selected.empty


def make_completed_and_quotes():
    rng=np.random.default_rng(120)
    teams=['BOS','NYK','MIL','ATL','LAL','GSW']
    games=[];quotes=[]
    day=pd.Timestamp('2022-09-15')
    for i in range(1350):
        d=day+pd.Timedelta(days=i)
        home=teams[i%6]; away=teams[(i*2+1)%6]
        if away==home: away=teams[(i+3)%6]
        if away==home: away=teams[(i+1)%6]
        hp=round(113+rng.normal(0,13)); ap=round(109+rng.normal(0,13))
        gid=f'{i:010d}'
        games.append({'GAME_ID':gid,'GAME_DATE':d,'HOME_TEAM':home,
                      'AWAY_TEAM':away,'HOME_PTS':hp,'AWAY_PTS':ap})
        if d>=pd.Timestamp('2024-10-01'):
            start=d.tz_localize('America/New_York')+pd.Timedelta(hours=20)
            snapshot=start-pd.Timedelta(hours=2)
            for kind,side,point in [('spreads','HOME',float(rng.choice([-6.5,-3.5,-0.5,3.5,6.5]))),
                                    ('totals','OVER',float(rng.choice([207.5,216.5,225.5,234.5,244.5])) )]:
                quotes.append({'game_date':str(d.date()),'home_team':home,'away_team':away,
                    'game_start_utc':start.tz_convert('UTC').isoformat(),
                    'snapshot_utc':snapshot.tz_convert('UTC').isoformat(),
                    'bookmaker':'testbook','market':kind,'side':side,'point':point,
                    'decimal_odds':1.91})
    completed=pd.DataFrame(games)
    q=pd.DataFrame(quotes)
    for col in ['game_start_utc','snapshot_utc']:
        q[col]=pd.to_datetime(q[col],utc=True)
    return completed,q


def test_chronological_market_validation_on_synthetic_test_fixture():
    completed,quotes=make_completed_and_quotes()
    artifact,audit=evaluate_market_history(completed,quotes,bookmaker='testbook',
         min_market_val=30,min_market_test=30)
    assert artifact['validation_scope']=='timestamped_historical_bookmaker_reference_lines'
    assert artifact['automatic_live_use'] is False
    assert artifact['polymarket_market_line_validated'] is False
    assert artifact['targets']['home_margin']['test_quotes_resolved']>30
    assert artifact['targets']['total']['test_quotes_resolved']>30
    assert set(audit.market)=={'spreads','totals'}
    assert audit.loc[audit.period=='test','game_date'].min()>='2025-10-01'
    assert 0<=audit.model_p.min()<=audit.model_p.max()<=1


def test_without_real_quotes_cannot_validate():
    completed,quotes=make_completed_and_quotes()
    only_test=quotes[quotes.game_date>='2025-10-01']
    artifact,audit=evaluate_market_history(completed,only_test,bookmaker='testbook',
                                           min_market_val=30,min_market_test=30)
    assert artifact['targets']['home_margin']['status']=='insufficient_market_quotes'
    assert artifact['targets']['total']['research_gate_passed'] is False
    assert artifact['targets']['total']['isotonic_knots'] is None


def test_wrong_spread_direction_is_not_a_win():
    # Home is expected to win by 5, but covers -5.5 only if actual margin >5.5.
    assert _probability(5,-(-5.5),12)<.5


def test_real_timestamped_points_without_prices_can_calibrate_but_no_roi():
    completed,quotes=make_completed_and_quotes()
    quotes=quotes.drop(columns=['decimal_odds'])
    # The direct evaluator expects the normalized DataFrame schema; CSV loader
    # explicitly creates NaN decimal_odds when absent.
    quotes['decimal_odds']=float('nan')
    artifact,audit=evaluate_market_history(completed,quotes,bookmaker='testbook',
                                           min_market_val=30,min_market_test=30)
    assert len(audit)>100
    assert artifact['targets']['total']['baseline_reference_roi']['n_all']==0
    assert artifact['targets']['total']['test_baseline']['brier'] is not None
