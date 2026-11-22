"""Prospective paper-only moneyline rule freeze and accounting invariants."""
import copy
import json
from pathlib import Path

import pytest
from src.moneyline_paper_track import capture, init, settle, report, _load_results

AT='2026-11-20T23:00:00+00:00'


def snapshot(*, p=.66,ask=.62,tip='2026-11-21T01:00:00+00:00', generated='2026-11-20T22:53:00+00:00',slug='nba-bkn-orl-2026-11-20'):
    side='ORL' if p>=.5 else 'BKN'
    return {
        'updated_at_utc': '2026-11-20T22:55:00+00:00',
        'games': [{'slug':slug, 'game_date':'2026-11-20', 'home':'ORL', 'away':'BKN',
                   'tipoff_utc':tip,
                   'winner_forecast':{'predicted_winner':side,'home_probability':p,
                                     'forecast_generated_at_utc':generated,
                                     'nba_game_id':'ESPN-401909306'},
                   'moneylines':{side:{'ask':ask,'shares':100.0,'token_id':'42'}}}]
    }


def first_record(tmp_path):
    picks=list((tmp_path/'data/paper_trades/moneyline_tracker/picks').glob('*.json'))
    assert len(picks)==1
    return picks[0],json.loads(picks[0].read_text())


def results(tmp_path,*,away_pts=95,home_pts=103, day='2026-11-20',home='ORL',away='BKN'):
    csv_path=tmp_path/'data/nba_2026_27_latest_raw.csv'
    csv_path.parent.mkdir(parents=True,exist_ok=True)
    home_won=home_pts>away_pts
    csv_path.write_text(
        'GAME_ID,GAME_DATE,TEAM_ID,TEAM_ABBREVIATION,PTS,WL,MATCHUP\n'
        f'0022600123,{day},2,{home},{home_pts},{"W" if home_won else "L"},{home} vs. {away}\n'
        f'0022600123,{day},1,{away},{away_pts},{"L" if home_won else "W"},{away} @ {home}\n')
    return csv_path


def test_config_freeze_is_idempotent(tmp_path):
    a=init(tmp_path,at=AT);b=init(tmp_path,at='2026-11-21T23:00:00Z')
    assert a==b
    assert set(a['strategies'])=={'every_game','ev_5pct','ev_10pct','win_prob_70pct'}


def test_freezes_one_selection_per_game_and_cannot_overwrite(tmp_path):
    a=capture(snapshot(),tmp_path,at=AT)
    assert len(a['created'])==1
    path,rec=first_record(tmp_path)
    assert rec['strategy_selected']=={'every_game':True,'ev_5pct':True,'ev_10pct':False,'win_prob_70pct':False}
    assert rec['price_basis']=='OBSERVED_BEST_ASK_NOT_CONFIRMED_FILL'
    raw=path.read_bytes()
    assert capture(snapshot(p=.9,ask=.2),tmp_path,at=AT)['created']==[]
    assert path.read_bytes()==raw


def test_every_game_includes_negative_estimated_ev(tmp_path):
    capture(snapshot(p=.72,ask=.81),tmp_path,at=AT)
    rec=first_record(tmp_path)[1]
    assert rec['strategy_selected']['every_game']
    assert rec['strategy_selected']['win_prob_70pct']
    assert not rec['strategy_selected']['ev_5pct']
    assert rec['estimated_ev_uncalibrated']<0


def test_no_late_or_too_early_or_old_or_absent_quotes(tmp_path):
    assert not capture(snapshot(tip='2026-11-20T22:59:00Z'),tmp_path,at=AT)['created']
    assert not capture(snapshot(tip='2026-11-21T10:00:00Z'),tmp_path,at=AT)['created']
    assert not capture(snapshot(generated='2026-11-19T12:00:00Z'),tmp_path,at=AT)['created']
    assert not capture(snapshot(ask=None),tmp_path,at=AT)['created']
    assert not list((tmp_path/'data/paper_trades/moneyline_tracker/picks').glob('*.json'))


def test_refuses_snapshots_before_start_or_stale(tmp_path):
    init(tmp_path,at=AT)
    with pytest.raises(ValueError,match='backfill'):
        capture(snapshot(),tmp_path,at='2026-11-20T22:59:59Z')
    s=snapshot();s['refresh_status']={'stale':True}
    assert capture(s,tmp_path,at=AT)['created']==[]
    s=snapshot();s['updated_at_utc']='2026-11-20T20:00:00Z'
    assert capture(s,tmp_path,at=AT)['created']==[]


def test_espn_source_id_settles_from_exact_nba_game_team_pairs(tmp_path):
    capture(snapshot(),tmp_path,at=AT)
    assert settle(tmp_path)=={'updated':0,'pending':1}
    results(tmp_path)
    assert settle(tmp_path)=={'updated':1,'pending':0}
    assert settle(tmp_path)=={'updated':0,'pending':0}
    r=report(tmp_path)
    assert r['captured_games']==1
    base=r['strategies'][0]
    assert base['strategy']=='every_game' and base['wins']==1 and base['selected']==1
    assert base['paper_profit_units']==pytest.approx(1/.62-1,abs=1e-5)
    assert base['paper_roi']==pytest.approx(1/.62-1,abs=1e-5)
    assert next(x for x in r['strategies'] if x['strategy']=='ev_10pct')['selected']==0


def test_completed_game_no_score_or_team_mismatch(tmp_path):
    capture(snapshot(),tmp_path,at=AT)
    results(tmp_path,day='2026-11-19')
    assert settle(tmp_path)['pending']==1
    results(tmp_path,home='BOS')
    assert settle(tmp_path)['pending']==1


def test_official_nba_id_must_match_exactly(tmp_path):
    s=snapshot();s['games'][0]['winner_forecast']['nba_game_id']='0022609999'
    capture(s,tmp_path,at=AT)
    results(tmp_path)
    assert settle(tmp_path)['pending']==1


def test_existing_settlement_conflicts_are_not_overwritten(tmp_path):
    capture(snapshot(),tmp_path,at=AT)
    csv=results(tmp_path)
    assert settle(tmp_path)['updated']==1
    before=list((tmp_path/'data/paper_trades/moneyline_tracker/settlements').glob('*.json'))[0]
    original=before.read_bytes()
    results(tmp_path,home_pts=90,away_pts=103)
    with pytest.raises(ValueError,match='Conflict'):
        settle(tmp_path)
    assert before.read_bytes()==original


def test_loss_has_minus_one_unit(tmp_path):
    capture(snapshot(),tmp_path,at=AT)
    results(tmp_path,home_pts=92,away_pts=107)
    settle(tmp_path)
    d=report(tmp_path)['strategies'][0]
    assert d['losses']==1 and d['paper_profit_units']==-1 and d['paper_roi']==-1


def test_no_bet_for_duplicate_game_in_same_snapshot(tmp_path):
    s=snapshot();s['games'].append(copy.deepcopy(s['games'][0]))
    r=capture(s,tmp_path,at=AT)
    assert len(r['created'])==1
    assert 'duplicate_live_game' in r['unavailable'].values()


def test_inconsistent_predicted_side_not_frozen(tmp_path):
    s=snapshot();s['games'][0]['winner_forecast']['predicted_winner']='BKN'
    assert not capture(s,tmp_path,at=AT)['created']


def test_report_before_init_is_explicit(tmp_path):
    assert report(tmp_path)['status']=='NOT_INITIALIZED'
