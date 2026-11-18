"""Analyze whether scorer-dependent NBA teams lose more when their star underperforms.

Run: python -u -m src.scoring_dependency
Uses only prior games to select the star and calculate pregame features.
Actual star points and game outcome are used ONLY for retrospective analysis.
"""
from collections import defaultdict, deque

import numpy as np
import pandas as pd

from .common import DATA, MODELS

WINDOW = 10
MIN_HISTORY = 10
MIN_STAR_APPEARANCES = 5
UNDERPERFORMANCE_RATIO = 0.80


def build_dependency(player_games, team_games):
    players = player_games.copy()
    teams = team_games.copy()
    for frame in (players, teams):
        frame['GAME_ID'] = frame['GAME_ID'].astype(str).str.zfill(10)
        frame['GAME_DATE'] = pd.to_datetime(frame['GAME_DATE'])
    players['TEAM_ID'] = pd.to_numeric(players['TEAM_ID'], errors='raise').astype('int64')
    players['PLAYER_ID'] = pd.to_numeric(players['PLAYER_ID'], errors='raise').astype('int64')
    players['PTS'] = pd.to_numeric(players['PTS'], errors='coerce').fillna(0)
    teams['TEAM_ID'] = pd.to_numeric(teams['TEAM_ID'], errors='raise').astype('int64')
    teams['PTS'] = pd.to_numeric(teams['PTS'], errors='coerce')
    if players.duplicated(['GAME_ID', 'TEAM_ID', 'PLAYER_ID']).any():
        raise ValueError('Duplicate player-game rows')
    if teams.duplicated(['GAME_ID', 'TEAM_ID']).any():
        raise ValueError('Duplicate team-game rows')
    player_lookup = {}
    names = {}
    for (game_id, team_id), group in players.groupby(['GAME_ID', 'TEAM_ID'], sort=False):
        player_lookup[(game_id, team_id)] = dict(zip(group['PLAYER_ID'], group['PTS']))
        names.update(zip(group['PLAYER_ID'], group['PLAYER_NAME']))
    teams = teams.sort_values(['GAME_DATE', 'GAME_ID', 'TEAM_ID']).reset_index(drop=True)
    history = defaultdict(lambda: deque(maxlen=WINDOW))
    rows = []
    # Batch by date so no game on a date can use another result from that date.
    for date, day in teams.groupby('GAME_DATE', sort=True):
        pending = []
        for game in day.itertuples(index=False):
            team_id = int(game.TEAM_ID)
            game_id = str(game.GAME_ID)
            previous = list(history[team_id])
            current = player_lookup.get((game_id, team_id), {})
            if len(previous) >= MIN_HISTORY:
                total_team_pts = sum(entry['team_points'] for entry in previous)
                star_totals = defaultdict(float)
                appearances = defaultdict(int)
                for entry in previous:
                    for pid, pts in entry['player_points'].items():
                        star_totals[pid] += pts
                        appearances[pid] += 1
                eligible = [pid for pid in star_totals if appearances[pid] >= MIN_STAR_APPEARANCES]
                if eligible and total_team_pts > 0:
                    star_id = max(eligible, key=lambda pid: (star_totals[pid], -pid))
                    star_points = [entry['player_points'].get(star_id, 0.0) for entry in previous]
                    star_mean = float(np.mean(star_points))
                    played = star_id in current
                    actual = float(current[star_id]) if played else np.nan
                    rows.append({
                        'GAME_ID': game_id, 'GAME_DATE': date, 'TEAM_ID': team_id,
                        'TEAM_ABBREVIATION': game.TEAM_ABBREVIATION,
                        'IS_WIN': int(game.IS_WIN), 'STAR_PLAYER_ID': star_id,
                        'STAR_PLAYER_NAME': names.get(star_id, str(star_id)),
                        'STAR_APPEARANCES_L10': appearances[star_id],
                        'TOP_SCORER_SHARE_L10': star_totals[star_id] / total_team_pts,
                        'TOP_SCORER_PPG_L10': star_mean,
                        'TOP_SCORER_PTS_STD_L10': float(np.std(star_points)),
                        'STAR_PLAYED': played, 'STAR_ACTUAL_PTS': actual,
                        'STAR_PERFORMANCE_RATIO': actual / star_mean if played and star_mean > 0 else np.nan,
                    })
            pending.append((team_id, {'team_points': float(game.PTS), 'player_points': current}))
        for team_id, entry in pending:
            history[team_id].append(entry)
    return pd.DataFrame(rows)


def main():
    player_path = DATA / 'nba_player_games.csv'
    team_path = DATA / 'nba_games_updated.csv'
    for path in (player_path, team_path):
        if not path.exists():
            raise FileNotFoundError(f'Missing {path}')
    players = pd.read_csv(player_path, dtype={'GAME_ID': str})
    teams = pd.read_csv(team_path, dtype={'GAME_ID': str})
    required_player = {'GAME_ID', 'GAME_DATE', 'TEAM_ID', 'PLAYER_ID', 'PLAYER_NAME', 'PTS'}
    required_team = {'GAME_ID', 'GAME_DATE', 'TEAM_ID', 'TEAM_ABBREVIATION', 'PTS', 'IS_WIN'}
    if required_player - set(players.columns) or required_team - set(teams.columns):
        raise ValueError(f'Missing player columns: {sorted(required_player-set(players.columns))}; '
                         f'team columns: {sorted(required_team-set(teams.columns))}')
    result = build_dependency(players, teams)
    if result.empty:
        raise ValueError('No eligible games; inspect data and rolling window requirements')
    DATA.mkdir(parents=True, exist_ok=True)
    MODELS.mkdir(parents=True, exist_ok=True)
    output = DATA / 'nba_scoring_dependency.csv'
    result.to_csv(output, index=False)
    print(f'Saved {len(result):,} team-game dependency rows to {output}')
    print(f'Leading scorer played in {result.STAR_PLAYED.mean():.1%} of eligible team-games')
    print(f'Median pregame top-scorer share: {result.TOP_SCORER_SHARE_L10.median():.1%}')
    analysis = result.loc[result.STAR_PLAYED & result.STAR_PERFORMANCE_RATIO.notna()].copy()
    median = analysis['TOP_SCORER_SHARE_L10'].median()
    analysis['dependency'] = np.where(analysis['TOP_SCORER_SHARE_L10'] >= median, 'High', 'Low')
    analysis['star_performance'] = np.where(
        analysis['STAR_PERFORMANCE_RATIO'] < UNDERPERFORMANCE_RATIO,
        'Below 80% of pregame average', 'At least 80% of pregame average'
    )
    summary = (analysis.groupby(['dependency', 'star_performance'], as_index=False)
               .agg(games=('IS_WIN', 'size'), wins=('IS_WIN', 'sum'), win_rate=('IS_WIN', 'mean'),
                    mean_star_share=('TOP_SCORER_SHARE_L10', 'mean')))
    summary.to_csv(MODELS / 'scoring_dependency_summary.csv', index=False)
    print('\n===== SCORING DEPENDENCY VS STAR UNDERPERFORMANCE =====')
    print(f'Dependency split: {median:.1%} top-scorer point share (median among games analyzed)')
    print('Underperformance: actual points below 80% of prior-10-team-game average')
    print(summary.to_string(index=False, float_format=lambda x: f'{x:.3f}'))
    print('\nThis is a retrospective association, NOT a pregame prediction or causal estimate.')
    print('Games where the preselected star did not play are excluded from this comparison.')
    print('Outputs: data/nba_scoring_dependency.csv and models/scoring_dependency_summary.csv')


if __name__ == '__main__':
    main()
