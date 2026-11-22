"""No network, no wagering; exercises chronological validation and safeguards."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.calibrate_game_lines import (
    probability_over, candidate_grid, reliability, evaluate_calibration,
    load_calibrator, calibrated_probability,
)
from src.joint_ev import _calibrate_line


def synthetic_games():
    """Fictional fully observed game history with a four-season time span."""
    rng=np.random.default_rng(33)
    rows=[]
    clubs=['ATL','BOS','CLE','DEN','MEM','NYK']
    for i,date in enumerate(pd.date_range('2023-01-01','2026-05-31',freq='D')):
        home=clubs[i%6]
        away=clubs[(i+1+(i//6)%4)%6]
        if home==away:away=clubs[(i+3)%6]
        margin=3+rng.normal(0,13)
        total=224+rng.normal(0,19)
        rows.append({'GAME_ID':str(i+1),'GAME_DATE':date,'HOME_TEAM':home,'AWAY_TEAM':away,
                     'HOME_PTS':(total+margin)/2,'AWAY_PTS':(total-margin)/2})
    return pd.DataFrame(rows)


def test_tail_probabilities_and_reliability():
    assert probability_over(10,10,3)==pytest.approx(.5)
    assert probability_over(10,8,3)>.5
    assert probability_over(10,12,3)<.5
    base, outcome=candidate_grid(np.array([100.,105.]),np.array([110.,99.]),10.)
    assert len(base)==len(outcome)==16
    report=reliability(np.array([.8,.2]),np.array([1,0]))
    assert report['brier']==pytest.approx(.04)
    assert report['threshold_events']==2


def test_chronological_fit_and_untouched_test():
    completed=synthetic_games()
    artifact=evaluate_calibration(completed,min_train=200,min_val=100,min_test=100)
    assert artifact['market_line_validated'] is False
    assert artifact['validation_scope']=='synthetic_forecast_centered_thresholds_only'
    assert artifact['split']['train_latest']<'2024-10-01'
    assert artifact['split']['validation_latest']<'2025-10-01'
    assert artifact['split']['test_latest']<'2026-07-01'
    for model in artifact['targets'].values():
        assert model['test_games']>=100
        assert len(model['knots_x'])>=2
        assert model['test_calibrated']['threshold_events']==model['test_games']*8
    # Change ONLY the untouched test labels and verify calibrator knots are
    # identical. This checks validation/test leakage protection in the fit.
    altered=completed.copy()
    mask=altered.GAME_DATE>=pd.Timestamp('2025-10-01')
    altered.loc[mask,'HOME_PTS']+=23
    second=evaluate_calibration(altered,min_train=200,min_val=100,min_test=100)
    for key in ('home_margin','total'):
        assert artifact['targets'][key]['knots_x']==second['targets'][key]['knots_x']
        assert artifact['targets'][key]['knots_y']==second['targets'][key]['knots_y']
        assert artifact['targets'][key]['sigma_validation']==second['targets'][key]['sigma_validation']
    assert artifact['targets']['home_margin']['test_calibrated'] != second['targets']['home_margin']['test_calibrated']


def test_invalid_calibration_is_rejected(tmp_path):
    path=tmp_path/'broken.json'
    assert load_calibrator(path) is None
    path.write_text('not json',encoding='utf-8')
    assert load_calibrator(path) is None
    assert calibrated_probability(None,'total',.8)==(.8,'uncalibrated_normal_approximation')
    assert calibrated_probability({'targets':{}},'spread',.8)==(.8,'uncalibrated_normal_approximation')
    data={'schema':1,'validation_scope':'synthetic_forecast_centered_thresholds_only',
          'market_line_validated':False,'probability_method':'isotonic_normal_cdf',
          'targets':{t:{'knots_x':[0,.5,1],'knots_y':[0,.4,1],
                        'use_for_research':True,'sigma_validation':14.0} for t in ('home_margin','total')}}
    path.write_text(json.dumps(data),encoding='utf-8')
    loaded=load_calibrator(path)
    assert loaded is not None
    assert calibrated_probability(loaded,'total',.5)==(.4,'test_evaluated_synthetic_thresholds')
    assert calibrated_probability(loaded,'total',0)==(.005,'test_evaluated_synthetic_thresholds')
    clipped=copy.deepcopy(data)
    clipped['targets']['total']['knots_x']=[.1,.5,.9]
    path.write_text(json.dumps(clipped),encoding='utf-8')
    clipped_artifact=load_calibrator(path)
    assert calibrated_probability(clipped_artifact,'total',.02)==(.02,'outside_calibration_support')
    data['targets']['total']['knots_y']=[1,.5,0]
    path.write_text(json.dumps(data),encoding='utf-8')
    assert load_calibrator(path) is None


def test_spread_away_complement_and_total_flip():
    artifact={'targets':{t:{'knots_x':[0,.5,1],'knots_y':[0,.35,1],
                            'use_for_research':True,'sigma_validation':14.0} for t in ('home_margin','total')}}
    home={'kind':'spread','side':'BOS -4.5','team_side':'HOME',
          'model_probability':.6,'direction':1,'market_line':-4.5}
    away={**home,'team_side':'AWAY','side':'NYK +4.5'}
    h=_calibrate_line(home,'spread',artifact)
    a=_calibrate_line(away,'spread',artifact)
    assert h['market_line_validated'] is False
    assert h['model_probability']==pytest.approx(.48)
    assert a['model_probability']==pytest.approx(.72) # 1 - iso(1-.6), linear interpolation
    over={'kind':'total','side':'OVER 224.5','market_line':224.5,'direction':1,
          'model_probability':.6}
    t=_calibrate_line(over,'total',artifact)
    assert t['side'].startswith('UNDER ')
    assert t['direction']==-1
    assert t['model_probability']==pytest.approx(.52)


def test_invalid_cutoffs_or_sparse_training_fail_closed():
    with pytest.raises(ValueError, match='Chronological'):
        evaluate_calibration(synthetic_games(),train_end='2026-01-01',valid_end='2025-01-01')
    with pytest.raises(ValueError, match='Insufficient chronological'):
        evaluate_calibration(synthetic_games(),min_train=10000)
