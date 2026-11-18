"""Produce one row per game from home and away team rows."""
import pandas as pd
from .common import DATA


def main():
    df = pd.read_csv(DATA / 'nba_games_updated.csv', dtype={'GAME_ID': str})
    home = df.loc[df['IS_HOME'], ['GAME_ID','GAME_DATE','SEASON','TEAM_ABBREVIATION','PTS','IS_WIN']].copy()
    away = df.loc[~df['IS_HOME'], ['GAME_ID','TEAM_ABBREVIATION','PTS']].copy()
    games = home.merge(away, on='GAME_ID', validate='one_to_one', suffixes=('_HOME','_AWAY'))
    games = games.rename(columns={
        'GAME_DATE':'GAME_DATE','SEASON':'SEASON',
        'TEAM_ABBREVIATION_HOME':'HOME_TEAM', 'TEAM_ABBREVIATION_AWAY':'AWAY_TEAM',
        'PTS_HOME':'HOME_PTS','PTS_AWAY':'AWAY_PTS','IS_WIN':'HOME_WIN'
    })
    games['GAME_DATE'] = pd.to_datetime(games['GAME_DATE'])
    games = games.sort_values(['GAME_DATE','GAME_ID']).reset_index(drop=True)
    if len(games) * 2 != len(df):
        raise ValueError('Lost games during merge')
    out = DATA / 'nba_games_merged.csv'
    games.to_csv(out, index=False)
    print(f'Saved {len(games)} games to {out}')

if __name__ == '__main__':
    main()
