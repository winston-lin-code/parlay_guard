"""Offline tests for immutable pregame tracking and verified-only settlement."""
from __future__ import annotations

import pandas as pd
import pytest

from src.paper_track_joint import (
    _line_leg, capture_pregame_snapshots,
    create_pregame_slate, performance_summary, settle_pending, settle_slate,
)

DATE = '2027-01-16'
START = '2027-01-17T01:00:00Z'
CAPTURE = '2027-01-16T22:05:00Z'
SOURCE = '2027-01-16T22:00:00Z'


def sample(root):
    (root/'data').mkdir(exist_ok=True)
    pd.DataFrame([
        {'event_slug': f'nba-por-mil-{DATE}', 'player_name': 'Player A', 'PLAYER_ID': 11},
        {'event_slug': f'nba-gsw-mem-{DATE}', 'player_name': 'Player B', 'PLAYER_ID': 12},
    ]).to_csv(root/'data/current_player_features.csv', index=False)
    g1 = {'slug': f'nba-por-mil-{DATE}', 'game_date': DATE, 'home': 'MIL', 'away': 'POR',
          'tipoff_utc': START, 'winner_forecast': {'predicted_winner': 'POR', 'nba_game_id': '0022602001'},
          'selected_props': [{'player': 'Player A', 'team': 'POR', 'stat': 'PTS', 'side': 'OVER',
                              'line': 15.5, 'ask': .4, 'model_probability': .7}]}
    g2 = {'slug': f'nba-gsw-mem-{DATE}', 'game_date': DATE, 'home': 'MEM', 'away': 'GSW',
          'tipoff_utc': START, 'winner_forecast': {'predicted_winner': 'MEM', 'nba_game_id': '0022602002'},
          'selected_props': [{'player': 'Player B', 'team': 'MEM', 'stat': 'REB', 'side': 'UNDER',
                              'line': 8.5, 'ask': .5, 'model_probability': .8}]}
    a = {'game': g1['slug'], 'player': 'Player A', 'team': 'POR', 'stat': 'PTS',
         'side': 'OVER', 'line': 15.5}
    b = {'game': g2['slug'], 'player': 'Player B', 'team': 'MEM', 'stat': 'REB',
         'side': 'UNDER', 'line': 8.5}
    joint = {'cross_game': [{'legs': [a, b], 'joint_probability': .49}],
             'same_game': [{'kind': 'prop_plus_spread', 'game': g1['slug'],
                            'player': 'Player A', 'prop': 'PTS OVER 15.5',
                            'game_line': 'POR +3.5', 'joint_probability': .5}],
             'cross_game_line': [{'kind': 'prop_plus_total', 'prop_game': g1['slug'],
                                  'line_game': g2['slug'], 'player': 'Player A',
                                  'prop': 'PTS OVER 15.5', 'game_line': 'OVER 225.5',
                                  'joint_probability': .36}]}
    return {'updated_at_utc': SOURCE, 'games': [g1, g2], 'joint_research': joint}


def team_results(valid=True):
    rows = [
        {'GAME_ID': '0022602001', 'GAME_DATE': DATE, 'TEAM_ABBREVIATION': 'POR', 'PTS': 115, 'WL': 'W'},
        {'GAME_ID': '0022602001', 'GAME_DATE': DATE, 'TEAM_ABBREVIATION': 'MIL', 'PTS': 110, 'WL': 'L'},
        {'GAME_ID': '0022602002', 'GAME_DATE': DATE, 'TEAM_ABBREVIATION': 'GSW', 'PTS': 100, 'WL': 'L'},
        {'GAME_ID': '0022602002', 'GAME_DATE': DATE, 'TEAM_ABBREVIATION': 'MEM', 'PTS': 130, 'WL': 'W'},
    ]
    if not valid:
        rows[1]['WL'] = 'W'
    return pd.DataFrame(rows)


def player_results():
    return pd.DataFrame([
        {'GAME_ID': '0022602001', 'PLAYER_ID': '11', 'TEAM_ID': '1610612757',
         'MIN': 30., 'PTS': 20., 'REB': 4., 'AST': 5., 'FG3M': 2.},
        {'GAME_ID': '0022602002', 'PLAYER_ID': '12', 'TEAM_ID': '1610612763',
         'MIN': 28., 'PTS': 18., 'REB': 6., 'AST': 5., 'FG3M': 1.},
    ])


def test_snapshot_contains_four_strategies_and_clear_price_limitations(tmp_path):
    snapshot = sample(tmp_path)
    slate = create_pregame_slate(snapshot, DATE, tmp_path, at=CAPTURE)
    assert slate is not None
    assert len(slate['tickets']) == 5
    assert {x['strategy'] for x in slate['tickets']} == {
        'single_prop', 'cross_game_props', 'same_game_prop_line', 'cross_game_prop_line'}
    cross = next(x for x in slate['tickets'] if x['strategy'] == 'cross_game_props')
    assert cross['reference_price'] == pytest.approx(.2)
    assert cross['price_basis'] == 'synthetic_product_not_executable'
    assert cross['is_executable_quote'] is False
    assert all(x['reference_price'] is None for x in slate['tickets'] if 'line' in x['strategy'])


def test_capture_is_immutable_and_idempotent(tmp_path):
    snap = sample(tmp_path)
    assert capture_pregame_snapshots(snap, tmp_path, at=CAPTURE) == [DATE]
    frozen = tmp_path/'data/paper_trades/joint_tracker/snapshots'/f'{DATE}.json'
    original = frozen.read_bytes()
    snap['games'][0]['selected_props'][0]['model_probability'] = .99
    assert capture_pregame_snapshots(snap, tmp_path, at=CAPTURE) == []
    assert frozen.read_bytes() == original


def test_pre_tipoff_and_freshness_are_mandatory(tmp_path):
    snap = sample(tmp_path)
    assert create_pregame_slate(snap, DATE, tmp_path, at='2027-01-16T23:00:00Z') is None
    assert create_pregame_slate(snap, DATE, tmp_path, at='2027-01-17T01:00:00Z') is None
    snap['refresh_status'] = {'stale': True}
    assert create_pregame_slate(snap, DATE, tmp_path, at=CAPTURE) is None


def test_does_not_capture_players_without_unique_verified_feature_ids(tmp_path):
    snap = sample(tmp_path)
    df = pd.read_csv(tmp_path/'data/current_player_features.csv')
    df = pd.concat([df, pd.DataFrame([{'event_slug': f'nba-por-mil-{DATE}',
                                       'player_name': 'Player A', 'PLAYER_ID': 999}])], ignore_index=True)
    df.to_csv(tmp_path/'data/current_player_features.csv', index=False)
    slate = create_pregame_slate(snap, DATE, tmp_path, at=CAPTURE)
    assert slate is not None
    assert len([t for t in slate['tickets'] if t['strategy'] == 'single_prop']) == 1
    assert all(t['legs'][0].get('player') != 'Player A' for t in slate['tickets'])


def test_settle_verified_props_spreads_totals_and_cross_game(tmp_path):
    slate = create_pregame_slate(sample(tmp_path), DATE, tmp_path, at=CAPTURE)
    result = settle_slate(slate, team_results(), player_results())
    assert all(r['result'] == 'WIN' for r in result['results'])
    assert len(result['results']) == 5


def test_no_show_missing_or_ambiguous_team_data_remains_pending(tmp_path):
    slate = create_pregame_slate(sample(tmp_path), DATE, tmp_path, at=CAPTURE)
    missing_player = player_results().iloc[:1].copy()
    out = settle_slate(slate, team_results(), missing_player)
    outcomes = {x['ticket_id']: x['result'] for x in out['results']}
    second_single = next(t for t in slate['tickets'] if t['strategy']=='single_prop' and t['legs'][0]['player']=='Player B')
    assert outcomes[second_single['id']] == 'PENDING'
    assert any(r['result']=='PENDING' for r in out['results'])
    assert all(r['result']=='PENDING' for r in settle_slate(slate, team_results(False), player_results())['results'][:1])


def test_push_is_not_scored_as_win_or_loss(tmp_path):
    slate = create_pregame_slate(sample(tmp_path), DATE, tmp_path, at=CAPTURE)
    # POR+3.5 cannot push at integer score, but integer totals can.
    total = _line_leg(slate['games'][f'nba-gsw-mem-{DATE}'], 'total', 'OVER 230')
    ticket = {'id': 'total', 'strategy': 'same_game_prop_line', 'legs': [total], 'reference_price': None,
              'model_probability_uncalibrated': .5}
    slate['tickets'] = [ticket]
    scores = team_results().copy()
    scores.loc[scores['GAME_ID'] == '0022602002', 'PTS'] = [100, 130]
    out = settle_slate(slate, scores, player_results())
    assert out['results'][0]['result'] == 'PUSH'


def test_settle_from_csv_and_only_complete_slates_count_in_roi(tmp_path):
    snap = sample(tmp_path)
    capture_pregame_snapshots(snap, tmp_path, at=CAPTURE)
    path = tmp_path/'data/nba_2026_27_latest_raw.csv'
    team_results().to_csv(path, index=False)
    player_path = tmp_path/'data/live_player_training_data_v2.csv'
    player_results().to_csv(player_path, index=False)
    assert settle_pending(tmp_path) == 1
    assert settle_pending(tmp_path) == 0
    summary = performance_summary(tmp_path)
    rows = {x['strategy']: x for x in summary['strategies']}
    assert summary['recorded_slates'] == 1
    assert rows['single_prop']['wins'] == 2
    assert rows['single_prop']['fully_settled_slates'] == 1
    assert rows['single_prop']['mean_paper_reference_roi'] == pytest.approx((1/.4-1 + 1/.5-1)/2)
    assert rows['cross_game_props']['mean_paper_reference_roi'] == pytest.approx(1/.2-1)
    assert rows['cross_game_props']['price_basis'].startswith('NONEXECUTABLE')
    assert rows['same_game_prop_line']['mean_paper_reference_roi'] is None
    assert rows['same_game_prop_line']['wins'] == 1


def test_pending_slate_has_no_reference_roi(tmp_path):
    capture_pregame_snapshots(sample(tmp_path), tmp_path, at=CAPTURE)
    # No settlement records yet, even though the model predicts high win P.
    rows = {x['strategy']: x for x in performance_summary(tmp_path)['strategies']}
    assert rows['single_prop']['pending'] == 2
    assert rows['single_prop']['mean_paper_reference_roi'] is None
    assert rows['cross_game_props']['mean_paper_reference_roi'] is None


def test_settlement_never_rewrites_immutable_capture(tmp_path):
    capture_pregame_snapshots(sample(tmp_path), tmp_path, at=CAPTURE)
    frozen_path = tmp_path/'data/paper_trades/joint_tracker/snapshots'/f'{DATE}.json'
    original = frozen_path.read_bytes()
    path = tmp_path/'data/nba_2026_27_latest_raw.csv'
    path.write_text('GAME_ID,GAME_DATE,TEAM_ABBREVIATION,PTS,WL\n', encoding='utf-8')
    assert settle_pending(tmp_path) == 1
    team_results().to_csv(path, index=False)
    player_results().to_csv(tmp_path/'data/live_player_training_data_v2.csv', index=False)
    assert settle_pending(tmp_path) == 1
    assert frozen_path.read_bytes() == original
    assert not performance_summary(tmp_path)['pending_dates']
