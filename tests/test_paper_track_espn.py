"""ESPN fallback schedule IDs: safe immutable capture and verified-only results."""
from __future__ import annotations

import copy
import pandas as pd

from src.paper_track_joint import create_pregame_slate, settle_slate, capture_pregame_snapshots

DAY = '2027-01-16'
SOURCE_TIME = '2027-01-16T21:55:00Z'
CAPTURE = '2027-01-16T22:00:00Z'
START = '2027-01-17T01:00:00Z'
SLUG = f'nba-orl-bkn-{DAY}'
NBA_ID = '0022602001'


def sample(root, game_id='ESPN-401909306'):
    (root/'data').mkdir(exist_ok=True)
    pd.DataFrame([{'event_slug': SLUG, 'player_name': 'Player B', 'PLAYER_ID': 22}]).to_csv(
        root/'data/current_player_features.csv', index=False)
    return {
        'updated_at_utc': SOURCE_TIME,
        'games': [{
            'slug': SLUG, 'game_date': DAY, 'home': 'BKN', 'away': 'ORL',
            'tipoff_utc': START,
            'winner_forecast': {'predicted_winner': 'ORL', 'nba_game_id': game_id},
            'selected_props': [{'player': 'Player B', 'team': 'ORL', 'stat': 'PTS',
                                'side': 'OVER', 'line': 15.5, 'ask': .44,
                                'model_probability': .70}],
        }],
        'joint_research': {},
    }


def teams(game_id=NBA_ID, *, bad=False):
    return pd.DataFrame([
        {'GAME_ID': game_id, 'GAME_DATE': DAY, 'TEAM_ABBREVIATION': 'ORL', 'PTS': 115, 'WL': 'W'},
        {'GAME_ID': game_id, 'GAME_DATE': DAY, 'TEAM_ABBREVIATION': 'BKN', 'PTS': 105, 'WL': 'W' if bad else 'L'},
    ])


def players(game_id=NBA_ID):
    return pd.DataFrame([{'GAME_ID': game_id, 'PLAYER_ID': '22', 'TEAM_ID': '1610612753',
                          'MIN': 31, 'PTS': 23, 'REB': 3, 'AST': 2, 'FG3M': 2}])


def test_espn_capture_preserves_provenance_and_player_id(tmp_path):
    slate = create_pregame_slate(sample(tmp_path), DAY, tmp_path, at=CAPTURE)
    assert slate is not None
    game = slate['games'][SLUG]
    assert game['nba_game_id'] is None
    assert game['source_game_id'] == 'ESPN-401909306'
    assert game['source_provider'] == 'ESPN'
    assert slate['tickets'][0]['legs'][0]['player_id'] == '22'
    assert capture_pregame_snapshots(sample(tmp_path), tmp_path, at=CAPTURE) == [DAY]
    assert capture_pregame_snapshots(sample(tmp_path), tmp_path, at=CAPTURE) == []


def test_espn_id_resolves_to_unique_finished_nba_game_for_player_settlement(tmp_path):
    slate = create_pregame_slate(sample(tmp_path), DAY, tmp_path, at=CAPTURE)
    assert settle_slate(slate, teams(), players())['results'][0]['result'] == 'WIN'
    assert settle_slate(slate, teams(), players('0022609999'))['results'][0]['result'] == 'PENDING'


def test_espn_pending_if_missing_incomplete_or_ambiguous_game(tmp_path):
    slate = create_pregame_slate(sample(tmp_path), DAY, tmp_path, at=CAPTURE)
    assert settle_slate(slate, None, players())['results'][0]['result'] == 'PENDING'
    assert settle_slate(slate, teams(bad=True), players())['results'][0]['result'] == 'PENDING'
    duplicate = pd.concat([teams(), teams('0022608888')], ignore_index=True)
    assert settle_slate(slate, duplicate, players())['results'][0]['result'] == 'PENDING'


def test_espn_rejects_future_and_invalid_schedule_ids(tmp_path):
    snap = sample(tmp_path)
    assert create_pregame_slate(snap, DAY, tmp_path, at=START) is None
    snap['games'][0]['winner_forecast']['nba_game_id'] = 'UNKNOWN-001'
    assert create_pregame_slate(snap, DAY, tmp_path, at=CAPTURE) is None


def test_numeric_nba_id_still_requires_exact_verified_id(tmp_path):
    slate = create_pregame_slate(sample(tmp_path, '0022602001'), DAY, tmp_path, at=CAPTURE)
    assert slate['games'][SLUG]['nba_game_id'] == NBA_ID
    assert settle_slate(slate, teams('0022608888'), players('0022608888'))['results'][0]['result'] == 'PENDING'
    assert settle_slate(slate, teams(), players())['results'][0]['result'] == 'WIN'
