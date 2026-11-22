"""Parlay Guard paper tracker fix for ESPN fallback schedule IDs.

Run from the project root: python fix_espn_paper_tracking.py

Only changes src/paper_track_joint.py and adds tests/test_paper_track_espn.py.
Never edits models, data, historical snapshots, or settlement records.
"""
from __future__ import annotations

import ast
from datetime import datetime
from pathlib import Path
import shutil

ROOT = Path.cwd()
TARGET = ROOT / 'src' / 'paper_track_joint.py'
TEST = ROOT / 'tests' / 'test_paper_track_espn.py'
MARKER = '# ESPN_FALLBACK_GAME_IDS_V1'

OLD_CAPTURE = '''        if pd.isna(tip) or not captured < tip or not gameid or not re.fullmatch(r'\\d+', gameid):
            continue
        if not win.get('predicted_winner'):
            continue
        away, home = str(g.get('away', '')).upper(), str(g.get('home', '')).upper()
        if away == home or not away or not home:
            continue
        all_games[g['slug']] = {'slug': str(g['slug']), 'game_date': nba_date,
                                'nba_game_id': _gameid(gameid), 'home': home, 'away': away,
                                'tipoff_utc': tip.isoformat()}'''
NEW_CAPTURE = '''        # ESPN_FALLBACK_GAME_IDS_V1
        # The ESPN schedule ID is *not* an NBA GAME_ID. Freeze it as source
        # metadata and resolve the NBA ID only from completed, paired team logs.
        if pd.isna(tip) or not captured < tip or not re.fullmatch(r'(?:\\d+|ESPN-\\d+)', gameid):
            continue
        if not win.get('predicted_winner'):
            continue
        away, home = str(g.get('away', '')).upper(), str(g.get('home', '')).upper()
        if away == home or not away or not home:
            continue
        official_nba_id = _gameid(gameid) if gameid.isdecimal() else None
        all_games[g['slug']] = {'slug': str(g['slug']), 'game_date': nba_date,
                                'nba_game_id': official_nba_id,
                                'source_game_id': gameid,
                                'source_provider': 'NBA' if official_nba_id else 'ESPN',
                                'home': home, 'away': away,
                                'tipoff_utc': tip.isoformat()}'''
OLD_COMPLETE = '''def _complete_game(info, team_games):
    if team_games is None:
        return None
    rows = team_games[team_games['GAME_ID'] == info['nba_game_id']]
    if len(rows) != 2 or set(rows['TEAM_ABBREVIATION']) != {info['home'], info['away']}:
        return None
    if set(rows['GAME_DATE']) != {info['game_date']} or set(rows['WL']) != {'W', 'L'} or not rows['PTS'].notna().all():
        return None
    scores = {str(r.TEAM_ABBREVIATION): float(r.PTS) for r in rows.itertuples(index=False)}
    if scores[info['home']] == scores[info['away']]:
        return None
    winner = max(scores, key=scores.get)
    recorded_winner = rows.loc[rows['WL'] == 'W', 'TEAM_ABBREVIATION'].iloc[0]
    if winner != recorded_winner:
        return None
    return scores
'''
NEW_COMPLETE = '''def _complete_game(info, team_games):
    """Return (scores, verified NBA GAME_ID), or None if missing/ambiguous.

    An ESPN source identifier is never converted into an NBA ID by stripping
    digits. For ESPN schedules we resolve an NBA ID via a unique, completed
    two-team record on the exact NBA date, then use that ID for player logs.
    """
    if team_games is None:
        return None
    teams = {info['home'], info['away']}
    official_id = info.get('nba_game_id')
    if official_id:
        candidates = [(official_id, team_games[team_games['GAME_ID'] == official_id])]
    elif re.fullmatch(r'ESPN-\\d+', str(info.get('source_game_id') or '')):
        same_date_teams = team_games[
            (team_games['GAME_DATE'] == info['game_date']) &
            (team_games['TEAM_ABBREVIATION'].isin(teams))
        ]
        candidates = list(same_date_teams.groupby('GAME_ID'))
    else:
        return None

    verified = []
    for nba_id, rows in candidates:
        if len(rows) != 2 or set(rows['TEAM_ABBREVIATION']) != teams:
            continue
        if set(rows['GAME_DATE']) != {info['game_date']} or set(rows['WL']) != {'W', 'L'} or not rows['PTS'].notna().all():
            continue
        scores = {str(r.TEAM_ABBREVIATION): float(r.PTS) for r in rows.itertuples(index=False)}
        if scores[info['home']] == scores[info['away']]:
            continue
        winner = max(scores, key=scores.get)
        recorded_winner = rows.loc[rows['WL'] == 'W', 'TEAM_ABBREVIATION'].iloc[0]
        if winner != recorded_winner:
            continue
        verified.append((scores, str(nba_id)))
    return verified[0] if len(verified) == 1 else None
'''
OLD_SETTLE = '''    scores = _complete_game(info, team_data)
    if scores is None:
        return 'PENDING'
    kind = leg['kind']'''
NEW_SETTLE = '''    verified = _complete_game(info, team_data)
    if verified is None:
        return 'PENDING'
    scores, verified_nba_game_id = verified
    kind = leg['kind']'''
OLD_PLAYER = '''        rows = player_data[(player_data['GAME_ID'] == info['nba_game_id']) &
                           (player_data['PLAYER_ID'] == leg['player_id'])]'''
NEW_PLAYER = '''        rows = player_data[(player_data['GAME_ID'] == verified_nba_game_id) &
                           (player_data['PLAYER_ID'] == leg['player_id'])]'''

TEST_CONTENT = '''"""ESPN fallback schedule IDs: safe immutable capture and verified-only results."""
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
'''


def replace_exact(content: str, old: str, new: str, label: str) -> str:
    occurrences = content.count(old)
    if occurrences != 1:
        raise ValueError(f'{label}: expected one source match, found {occurrences}. No files changed.')
    return content.replace(old, new, 1)


def main() -> None:
    if not TARGET.is_file():
        raise SystemExit('Missing src/paper_track_joint.py; run from Parlay Guard repository root.')
    original = TARGET.read_text(encoding='utf-8')
    if MARKER in original:
        print('ESPN fallback fix is already installed; no changes made.')
        return
    changed = original
    for old, new, label in [
        (OLD_CAPTURE, NEW_CAPTURE, 'Capture game-ID validation'),
        (OLD_COMPLETE, NEW_COMPLETE, 'Completed-game resolution'),
        (OLD_SETTLE, NEW_SETTLE, 'Settlement verification'),
        (OLD_PLAYER, NEW_PLAYER, 'Player appearance matching'),
    ]:
        changed = replace_exact(changed, old, new, label)
    ast.parse(changed, filename=str(TARGET))
    ast.parse(TEST_CONTENT, filename=str(TEST))
    if TEST.is_file():
        raise SystemExit(f'{TEST} already exists; refusing to overwrite it.')
    backup_dir = Path.home() / '.parlay_guard_backups' / (datetime.now().strftime('%Y%m%d_%H%M%S') + '_espn_fallback')
    backup = backup_dir / 'src' / 'paper_track_joint.py'
    backup.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(TARGET, backup)
    TARGET.write_text(changed, encoding='utf-8')
    TEST.parent.mkdir(parents=True, exist_ok=True)
    TEST.write_text(TEST_CONTENT, encoding='utf-8')
    print('Installed ESPN-schedule fallback support for immutable paper snapshots.')
    print('Updated:', TARGET)
    print('Added:', TEST)
    print('Original backed up:', backup)
    print('Next: python -m pytest tests/ -q')
    print('Then restart Uvicorn and refresh while a qualifying game is still upcoming.')


if __name__ == '__main__':
    main()
