"""Normalize team-game records and verify exactly two teams per game."""
import pandas as pd
from .common import DATA


def main():
    df = pd.read_csv(DATA / 'nba_all_seasons.csv', dtype={'GAME_ID': str})
    required = {'GAME_ID','GAME_DATE','TEAM_ID','TEAM_ABBREVIATION','MATCHUP','WL','PTS'}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f'Missing columns: {sorted(missing)}')
    df['GAME_ID'] = df['GAME_ID'].str.zfill(10)
    df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
    df['IS_WIN'] = df['WL'].eq('W')
    df['IS_HOME'] = df['MATCHUP'].str.contains('vs.', regex=False, na=False)
    away_format = df['MATCHUP'].str.contains('@', regex=False, na=False)
    after_at = df['MATCHUP'].str.split(' @ ').str[-1].str.strip()
    df.loc[away_format, 'IS_HOME'] = (
        df.loc[away_format, 'TEAM_ABBREVIATION'].eq(after_at.loc[away_format])
    )
    if df.duplicated(['GAME_ID','TEAM_ID']).any():
        raise ValueError('Duplicate GAME_ID/TEAM_ID rows')
    counts = df.groupby('GAME_ID').agg(rows=('TEAM_ID','size'), homes=('IS_HOME','sum'), wins=('IS_WIN','sum'))
    invalid = counts[(counts['rows'] != 2) | (counts['homes'] != 1) | (counts['wins'] != 1)]
    if not invalid.empty:
        raise ValueError(f'{len(invalid)} games failed 2-team/home/winner validation. Examples: {invalid.head().to_dict("index")}')
    out = DATA / 'nba_games_updated.csv'
    df.to_csv(out, index=False)
    print(f'Saved {len(df)} team-game rows ({len(counts)} games) to {out}')

if __name__ == '__main__':
    main()
