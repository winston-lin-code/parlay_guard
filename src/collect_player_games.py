"""Collect NBA player game logs matching games in Parlay Guard's model dataset.

Run from project root: python -u -m src.collect_player_games
Optional: python -u -m src.collect_player_games --refresh
"""
import argparse
import time
from pathlib import Path

import pandas as pd
from nba_api.stats.endpoints import playergamelogs

from .common import DATA


def season_for_date(date):
    """NBA season begins in calendar year of October; Jan-Sep belong to prior season."""
    return date.year if date.month >= 10 else date.year - 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--refresh', action='store_true', help='Redownload even if output exists')
    parser.add_argument('--pause', type=float, default=1.5, help='Seconds between requests')
    args = parser.parse_args()

    source = DATA / 'nba_model_dataset.csv'
    output = DATA / 'nba_player_games.csv'
    if not source.exists():
        raise FileNotFoundError(f'Missing {source}; run src.merge_features first')
    if output.exists() and not args.refresh:
        print(f'{output} already exists. Use --refresh to download again.')
        return

    games = pd.read_csv(source, dtype={'GAME_ID': str}, usecols=['GAME_ID', 'GAME_DATE'])
    games['GAME_DATE'] = pd.to_datetime(games['GAME_DATE'])
    ids = set(games['GAME_ID'].str.zfill(10))
    seasons = sorted({season_for_date(d) for d in games['GAME_DATE']})
    print(f'Collecting {len(ids)} games across seasons: {seasons}', flush=True)

    all_frames = []
    for i, year in enumerate(seasons, 1):
        season = f'{year}-{str(year + 1)[-2:]}'
        print(f'[{i}/{len(seasons)}] Downloading {season}...', flush=True)
        try:
            endpoint = playergamelogs.PlayerGameLogs(
                season_nullable=season,
                season_type_nullable='Regular Season',
                timeout=60,
            )
            frame = endpoint.get_data_frames()[0]
        except Exception as exc:
            raise RuntimeError(f'Failed downloading {season}. Retry later with --refresh.') from exc
        if frame.empty:
            print(f'  Warning: no records returned for {season}', flush=True)
            continue
        if 'GAME_ID' not in frame:
            raise ValueError(f'No GAME_ID in {season}: {list(frame.columns)}')
        frame['GAME_ID'] = frame['GAME_ID'].astype(str).str.zfill(10)
        frame = frame[frame['GAME_ID'].isin(ids)].copy()
        print(f'  Matched {len(frame)} player-game rows', flush=True)
        all_frames.append(frame)
        if i < len(seasons):
            time.sleep(args.pause)

    if not all_frames:
        raise RuntimeError('No matching player games downloaded; no output written.')
    result = pd.concat(all_frames, ignore_index=True)
    needed = ['GAME_ID', 'GAME_DATE', 'TEAM_ID', 'PLAYER_ID', 'PLAYER_NAME', 'MIN', 'PTS', 'FGA', 'FTA']
    missing = [c for c in needed if c not in result]
    if missing:
        raise ValueError(f'NBA API response missing columns: {missing}. Available: {list(result.columns)}')
    result = result[needed].drop_duplicates(['GAME_ID', 'PLAYER_ID'])
    result['GAME_DATE'] = pd.to_datetime(result['GAME_DATE']).dt.strftime('%Y-%m-%d')
    result = result.sort_values(['GAME_DATE', 'GAME_ID', 'TEAM_ID', 'PLAYER_ID'])
    DATA.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)
    matched = result['GAME_ID'].nunique()
    print(f'Saved {len(result):,} player-game rows covering {matched:,}/{len(ids):,} games to {output}', flush=True)
    if matched < len(ids):
        print('Note: some games were not matched (e.g. playoffs or incomplete seasons).', flush=True)


if __name__ == '__main__':
    main()
