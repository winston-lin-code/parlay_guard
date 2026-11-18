"""Optional: fetch regular-season team game logs from NBA Stats.

NBA Stats can throttle or block automated requests. Existing user CSVs are preferred.
"""
import argparse
import time
import pandas as pd
from nba_api.stats.endpoints import leaguegamefinder
from .common import DATA


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seasons', nargs='+', default=['2022-23','2023-24','2024-25','2025-26'])
    args = parser.parse_args()
    frames = []
    for season in args.seasons:
        print(f'Fetching {season} ...', flush=True)
        endpoint = leaguegamefinder.LeagueGameFinder(
            league_id_nullable='00', season_nullable=season,
            season_type_nullable='Regular Season', timeout=60
        )
        season_df = endpoint.get_data_frames()[0]
        season_df['SEASON'] = season
        frames.append(season_df)
        time.sleep(2)
    df = pd.concat(frames, ignore_index=True)
    df['GAME_ID'] = df['GAME_ID'].astype(str).str.zfill(10)
    path = DATA / 'nba_all_seasons.csv'
    df.to_csv(path, index=False)
    print(f'Saved {len(df)} team-game rows to {path}')

if __name__ == '__main__':
    main()
