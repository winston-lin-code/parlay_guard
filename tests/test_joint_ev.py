"""Offline regression tests for read-only joint probability research."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from src.joint_ev import (
    _line_candidate, _sigma, build_game_residuals, build_joint_research,
    estimate_correlation, joint_gaussian,
)


def test_joint_gaussian_independent_and_bounded():
    assert joint_gaussian(0.6, 0.7, 0) == pytest.approx(0.42)
    assert joint_gaussian(0.6, 0.7, 0.5) > 0.42
    assert joint_gaussian(0.6, 0.7, -0.5) < 0.42
    assert joint_gaussian(1, 0.7, 0.5) == pytest.approx(0.7)
    assert joint_gaussian(0, 0.7, 0.5) == pytest.approx(0)
    for rho in (-0.9, -0.3, 0, 0.3, 0.9):
        val = joint_gaussian(0.25, 0.8, rho)
        assert max(0, .25+.8-1) <= val <= min(.25, .8)
    with pytest.raises(ValueError):
        joint_gaussian(.5, .5, 1.01)


def test_historical_correlation_requires_sample_and_shrinks():
    rng = np.random.default_rng(5)
    x = rng.normal(size=90)
    y = .6*x + rng.normal(size=90)*.4
    result = estimate_correlation(pd.DataFrame({'a':x, 'b':y}), 'a', 'b')
    assert result is not None
    assert result['n'] == 90
    assert 0 < result['shrunk_rho'] < result['raw_rho'] < 1
    assert result['rho_low'] <= result['shrunk_rho'] <= result['rho_high']
    assert estimate_correlation(pd.DataFrame({'a':x[:10], 'b':y[:10]}), 'a', 'b') is None


def test_rolling_game_residuals_do_not_use_current_or_same_day_games():
    dates = pd.date_range('2026-01-01', periods=12, freq='D')
    games = []
    for i, day in enumerate(dates):
        games.append({'GAME_ID': str(i), 'GAME_DATE': day, 'HOME_TEAM':'A',
                      'AWAY_TEAM':'B', 'HOME_PTS':100., 'AWAY_PTS':90.})
    df = pd.DataFrame(games)
    result = build_game_residuals(df)
    assert len(result) == 2
    assert set(result['GAME_ID']) == {'0000000010', '0000000011'}
    assert abs(result.iloc[0]['margin_residual']) < 1e-9
    # A shock in a current game must not change that SAME game's prediction.
    shocked = df.copy()
    shocked.loc[10, 'HOME_PTS'] = 220.
    shifted = build_game_residuals(shocked)
    assert shifted.iloc[0]['margin_residual'] == pytest.approx(120.)
    assert shifted.iloc[1]['margin_residual'] != pytest.approx(0.)


def test_sigma_requires_holdout_and_sensible_mae():
    assert _sigma({'home_margin_holdout_mae': 12, 'holdout_games': 200}, 'spread') > 12
    assert _sigma({'home_margin_holdout_mae': 12, 'holdout_games': 20}, 'spread') is None
    assert _sigma({'holdout_games': 200}, 'total') is None


def test_no_props_refuses_to_invent_parlays(tmp_path):
    data = build_joint_research({
        'updated_at_utc': '2026-11-19T23:00:00Z',
        'games': [{'slug': 'nba-gsw-mem-2026-11-19', 'selected_props': []}]
    }, tmp_path)
    assert data['singles'] == []
    assert data['cross_game'] == []
    assert data['same_game'] == []
    assert data['warnings']


def test_cross_game_requires_distinct_games_and_never_claims_executable_ev(tmp_path):
    games = []
    for slug, player in [('g1','A'), ('g1','B'), ('g2','C')]:
        games.append({'slug': slug, 'selected_props': [{
            'team':'MEM','player':player,'stat':'PTS','side':'OVER','line':10.5,
            'model_probability':.70, 'ask':.50}]})
    data = build_joint_research({'updated_at_utc':'2026-11-19T23:00:00Z', 'games':games},tmp_path)
    assert len(data['singles']) == 3
    assert len(data['cross_game']) == 2
    assert all(p['legs'][0]['game'] != p['legs'][1]['game'] for p in data['cross_game'])
    assert data['cross_game'][0]['joint_probability'] == pytest.approx(.65*.65)
    assert data['cross_game'][0]['hypothetical_product_ask'] == pytest.approx(.25)
    assert data['cross_game'][0]['executable_combo_ask'] is None
    assert data['cross_game'][0]['executable_ev_roi'] is None
    # Never infer same-game independence for missing historical correlation.
    assert data['same_game'] == []
    assert data['cross_game_line'] == []


def test_missing_history_prevents_same_game_price_claim(tmp_path):
    games = [{'slug':'nba-gsw-mem-2026-11-19','home':'MEM','away':'GSW',
              'game_date':'2026-11-19',
              'game_line_prediction':{'home_margin':5.0,'projected_total':220.0},
              'market_total':{'line':222.5},
              'selected_props':[{'team':'MEM','player':'P','stat':'PTS','side':'OVER',
                                 'line':15.5,'model_probability':.7,'ask':.45}]}]
    result = build_joint_research({'updated_at_utc':'2026-11-19T23:00:00Z',
                                  'games':games,
                                  'line_model_metrics':{'total_holdout_mae':15., 'holdout_games':100}},tmp_path)
    assert len(result['singles']) == 1
    assert result['same_game'] == []
    assert result['cross_game_line'] == []
    assert 'missing' in ' '.join(result['warnings']).lower()


def test_same_game_uses_paired_pregame_residuals_and_refuses_combo_ev(tmp_path, monkeypatch):
    import sys
    import types
    from pathlib import Path

    day0 = pd.Timestamp('2026-08-01')
    game_rows = []
    player_rows = []
    for i in range(70):
        day = day0 + pd.Timedelta(days=i)
        hp = 112 + (i % 11)*1.5
        ap = 100 + (i % 7)
        gid = str(i+1).zfill(10)
        game_rows.append({'GAME_ID':gid, 'GAME_DATE':day, 'HOME_TEAM':'MEM',
                          'AWAY_TEAM':'GSW', 'HOME_PTS':hp, 'AWAY_PTS':ap})
        if i >= 15:
            pts = 20 + (i % 11)*1.1
            player_rows.append({'GAME_ID':gid, 'GAME_DATE':day, 'PLAYER_ID':101,
                                'PTS':pts, 'PTS_LAST_10':21.,
                                'REB':5., 'REB_LAST_10':5., 'AST':3., 'AST_LAST_10':3.,
                                'FG3M':2., 'FG3M_LAST_10':2.})
    completed = pd.DataFrame(game_rows)
    raw = pd.DataFrame(player_rows)
    (tmp_path/'data').mkdir()
    raw.to_csv(tmp_path/'data/live_player_training_data_v2.csv',index=False)
    (tmp_path/'data/nba_2026_27_latest_raw.csv').write_text('fake raw',encoding='utf-8')
    pd.DataFrame([{'event_slug':'nba-gsw-mem-2026-11-19',
                   'player_name':'A Player','PLAYER_ID':101}]).to_csv(
        tmp_path/'data/current_player_features.csv',index=False)
    module = types.ModuleType('src.predict_today')
    module.load_history = lambda *args, **kwargs: completed
    module.fetch_current_season = lambda *args, **kwargs: completed
    module.clean_team_history = lambda *args, **kwargs: completed
    module.make_completed_games = lambda *args, **kwargs: completed
    monkeypatch.setitem(sys.modules,'src.predict_today',module)
    team = types.ModuleType('src.auto_live')
    team.NICKNAMES = {'gsw':'warriors','mem':'grizzlies'}
    monkeypatch.setitem(sys.modules,'src.auto_live',team)
    game = {'slug':'nba-gsw-mem-2026-11-19','home':'MEM','away':'GSW',
            'game_date':'2026-11-19','game_line_prediction':{
                'home_margin':6.0,'projected_total':225.0},
            'market_spread':{'line':-3.5,'team_label':'Grizzlies'},
            'market_total':{'line':226.5},
            'selected_props':[{'team':'MEM','player':'A Player','stat':'PTS',
                               'side':'OVER','line':22.5,'model_probability':.68,'ask':.48}]}
    second_game = {'slug':'nba-cha-atl-2026-11-19','home':'ATL','away':'CHA',
                   'game_date':'2026-11-19',
                   'game_line_prediction':{'home_margin':5.0,'projected_total':230.0},
                   'market_total':{'line':228.5},'selected_props':[]}
    result = build_joint_research({
        'updated_at_utc':'2026-11-19T22:00:00Z','games':[game, second_game],
        'line_model_metrics':{'home_margin_holdout_mae':11.7,
                              'total_holdout_mae':15.3,'holdout_games':1230}
    }, tmp_path)
    assert result['warnings'] == []
    assert len(result['same_game']) == 2
    assert len(result['cross_game_line']) == 1
    assert result['cross_game_line'][0]['executable_ev_roi'] is None
    assert all(row['correlation']['n'] >= 25 for row in result['same_game'])
    assert all(0 < row['joint_probability'] < 1 for row in result['same_game'])
    assert all(row['executable_ev_roi'] is None for row in result['same_game'])
    assert all(row['status'] == 'ESTIMATED_UNCALIBRATED' for row in result['same_game'])
