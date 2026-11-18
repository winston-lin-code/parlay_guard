"""Create leakage-safe NBA player underperformance training features.

Run: python -u -m src.player_underperformance_features

Input:
    data/nba_player_games.csv
    data/nba_games_updated.csv
Output:
    data/nba_player_underperformance_dataset.csv

One row per player appearance with >=10 prior appearances. All predictive
features are computed using games on STRICTLY EARLIER DATES. The target is
1 if actual points < 80% of the player's prior-10-appearance PPG.

This script builds features; it does not train a prediction model.
"""
from collections import defaultdict, deque

import numpy as np
import pandas as pd

from .common import DATA

PLAYER_WINDOW = 10
TEAM_WINDOW = 10
UNDERPERFORMANCE_FRACTION = 0.80


def build_player_features(player_games: pd.DataFrame, team_games: pd.DataFrame) -> pd.DataFrame:
    players = player_games.copy()
    teams = team_games.copy()

    player_required = {'GAME_ID', 'GAME_DATE', 'TEAM_ID', 'PLAYER_ID', 'PLAYER_NAME', 'PTS', 'MIN', 'FGA'}
    team_required = {'GAME_ID', 'GAME_DATE', 'TEAM_ID', 'PTS', 'IS_HOME'}
    for label, frame, required in [('player', players, player_required), ('team', teams, team_required)]:
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f'{label} CSV missing columns: {sorted(missing)}')

    for frame in (players, teams):
        frame['GAME_ID'] = frame['GAME_ID'].astype(str).str.zfill(10)
        frame['GAME_DATE'] = pd.to_datetime(frame['GAME_DATE'], errors='raise').dt.normalize()
        frame['TEAM_ID'] = pd.to_numeric(frame['TEAM_ID'], errors='raise').astype('int64')

    players['PLAYER_ID'] = pd.to_numeric(players['PLAYER_ID'], errors='raise').astype('int64')
    for col in ('PTS', 'MIN', 'FGA'):
        players[col] = pd.to_numeric(players[col], errors='coerce')
    teams['PTS'] = pd.to_numeric(teams['PTS'], errors='coerce')

    if players.duplicated(['GAME_ID', 'TEAM_ID', 'PLAYER_ID']).any():
        raise ValueError('Duplicate player-game rows detected')
    if teams.duplicated(['GAME_ID', 'TEAM_ID']).any():
        raise ValueError('Duplicate team-game rows detected')
    if teams['PTS'].isna().any():
        raise ValueError('Missing team PTS values')

    # Map each team-game to its opposing team. Each NBA game must have 2 teams.
    opponent_map = {}
    for game_id, group in teams.groupby('GAME_ID', sort=False):
        if len(group) != 2:
            raise ValueError(f'Game {game_id} has {len(group)} team rows; expected 2')
        a, b = group.iloc[0], group.iloc[1]
        opponent_map[(game_id, int(a.TEAM_ID))] = int(b.TEAM_ID)
        opponent_map[(game_id, int(b.TEAM_ID))] = int(a.TEAM_ID)

    players = players.sort_values(['GAME_DATE', 'GAME_ID', 'TEAM_ID', 'PLAYER_ID'])
    teams = teams.sort_values(['GAME_DATE', 'GAME_ID', 'TEAM_ID'])
    players_by_date = {date: group for date, group in players.groupby('GAME_DATE', sort=True)}

    # Histories are updated only AFTER processing every game on the current date.
    player_history = defaultdict(lambda: deque(maxlen=PLAYER_WINDOW))
    team_history = defaultdict(lambda: deque(maxlen=TEAM_WINDOW))
    rows = []

    for date, todays_teams in teams.groupby('GAME_DATE', sort=True):
        todays_players = players_by_date.get(date)
        pending_players = []
        pending_teams = []

        if todays_players is not None:
            home_lookup = {
                (str(r.GAME_ID), int(r.TEAM_ID)): bool(r.IS_HOME)
                for r in todays_teams.itertuples(index=False)
            }
            for p in todays_players.itertuples(index=False):
                game_id = str(p.GAME_ID)
                team_id = int(p.TEAM_ID)
                player_id = int(p.PLAYER_ID)
                opponent_id = opponent_map.get((game_id, team_id))
                if opponent_id is None:
                    raise ValueError(f'No opponent for player-game {game_id}/{team_id}')

                # Only count actual appearances; no missed team games as zeroes.
                # A nonnumeric or nonpositive MIN is treated as no appearance.
                minutes = float(p.MIN) if pd.notna(p.MIN) else np.nan
                points = float(p.PTS) if pd.notna(p.PTS) else np.nan
                attempts = float(p.FGA) if pd.notna(p.FGA) else np.nan
                if not np.isfinite(minutes) or minutes <= 0 or not np.isfinite(points):
                    continue

                past = list(player_history[player_id])
                opp_past = list(team_history[opponent_id])
                own_past = list(team_history[team_id])

                if len(past) >= PLAYER_WINDOW:
                    last5 = past[-5:]
                    pts10 = np.array([x['points'] for x in past], dtype=float)
                    pts5 = np.array([x['points'] for x in last5], dtype=float)
                    min10 = np.array([x['minutes'] for x in past], dtype=float)
                    min5 = np.array([x['minutes'] for x in last5], dtype=float)
                    fga10 = np.array([x['fga'] for x in past], dtype=float)
                    fga5 = np.array([x['fga'] for x in last5], dtype=float)
                    ppg10 = float(pts10.mean())
                    days_rest = int((date - past[-1]['date']).days - 1)
                    if days_rest < 0:
                        raise ValueError(f'Unexpected negative rest for player {player_id}')

                    # Opponent points allowed = points scored by opponent's
                    # opponents in their previous games, never current game.
                    opp_allowed = [x['points_allowed'] for x in opp_past]
                    own_scored = [x['points_scored'] for x in own_past]
                    rows.append({
                        'GAME_ID': game_id,
                        'GAME_DATE': date,
                        'TEAM_ID': team_id,
                        'OPPONENT_TEAM_ID': opponent_id,
                        'PLAYER_ID': player_id,
                        'PLAYER_NAME': str(p.PLAYER_NAME),
                        'IS_HOME': int(home_lookup[(game_id, team_id)]),
                        'PLAYER_PPG_LAST_5': float(pts5.mean()),
                        'PLAYER_PPG_LAST_10': ppg10,
                        'PLAYER_PTS_STD_LAST_10': float(pts10.std(ddof=0)),
                        'PLAYER_MIN_LAST_5': float(np.nanmean(min5)),
                        'PLAYER_MIN_LAST_10': float(np.nanmean(min10)),
                        'PLAYER_FGA_LAST_5': float(np.nanmean(fga5)),
                        'PLAYER_FGA_LAST_10': float(np.nanmean(fga10)),
                        'PLAYER_DAYS_REST': days_rest,
                        'OPP_PTS_ALLOWED_LAST_5': float(np.mean(opp_allowed[-5:])) if len(opp_allowed) >= 5 else np.nan,
                        'OPP_PTS_ALLOWED_LAST_10': float(np.mean(opp_allowed)) if len(opp_allowed) >= 10 else np.nan,
                        'TEAM_PTS_SCORED_LAST_10': float(np.mean(own_scored)) if len(own_scored) >= 10 else np.nan,
                        'UNDERPERFORMANCE_THRESHOLD': UNDERPERFORMANCE_FRACTION * ppg10,
                        'ACTUAL_PTS': points,
                        'UNDERPERFORMED': int(points < UNDERPERFORMANCE_FRACTION * ppg10),
                    })

                pending_players.append((player_id, {
                    'date': date, 'points': points, 'minutes': minutes, 'fga': attempts,
                }))

        # Current team scores are added to history only after all rows for
        # the date have been computed. Opponent score is points allowed.
        todays_team_scores = {
            (str(t.GAME_ID), int(t.TEAM_ID)): float(t.PTS)
            for t in todays_teams.itertuples(index=False)
        }
        for t in todays_teams.itertuples(index=False):
            gid, tid = str(t.GAME_ID), int(t.TEAM_ID)
            opp_id = opponent_map[(gid, tid)]
            pending_teams.append((tid, {
                'date': date,
                'points_scored': float(t.PTS),
                'points_allowed': todays_team_scores[(gid, opp_id)],
            }))

        for pid, entry in pending_players:
            player_history[pid].append(entry)
        for tid, entry in pending_teams:
            team_history[tid].append(entry)

    return pd.DataFrame(rows)


def main() -> None:
    player_path = DATA / 'nba_player_games.csv'
    team_path = DATA / 'nba_games_updated.csv'
    for path in (player_path, team_path):
        if not path.exists():
            raise FileNotFoundError(f'Missing input file: {path}')

    print('Loading historical NBA player and team games...')
    players = pd.read_csv(player_path, dtype={'GAME_ID': str})
    teams = pd.read_csv(team_path, dtype={'GAME_ID': str})
    print(f'Player-game rows: {len(players):,}; team-game rows: {len(teams):,}')

    result = build_player_features(players, teams)
    if result.empty:
        raise ValueError('No eligible player-games. Check the historical data.')
    output = DATA / 'nba_player_underperformance_dataset.csv'
    result.to_csv(output, index=False)

    print(f'\nSaved {len(result):,} player-game training rows to {output}')
    print(f'Games covered: {result.GAME_ID.nunique():,}')
    print(f'Players covered: {result.PLAYER_ID.nunique():,}')
    print(f'Underperformance rate: {result.UNDERPERFORMED.mean():.1%}')
    print('\nMissing values in opponent features:')
    print(result[['OPP_PTS_ALLOWED_LAST_5', 'OPP_PTS_ALLOWED_LAST_10', 'TEAM_PTS_SCORED_LAST_10']].isna().sum().to_string())
    print('\nExample training rows:')
    print(result[['GAME_DATE', 'PLAYER_NAME', 'PLAYER_PPG_LAST_10', 'ACTUAL_PTS', 'UNDERPERFORMED']].head(8).to_string(index=False))
    print('\nIMPORTANT: ACTUAL_PTS and UNDERPERFORMED are targets/outcomes, not pregame model features.')
    print('UNDERPERFORMANCE_THRESHOLD is derived from pregame history and may be used for diagnostics.')


if __name__ == '__main__':
    main()
