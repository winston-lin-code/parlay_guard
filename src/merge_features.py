"""Join pregame home/away feature values to game outcomes."""
import pandas as pd
from .common import DATA
from .elo import add_elo_features

FEATURES = ['WIN_PCT_LAST_5','WIN_PCT_LAST_10','DAYS_REST','POINT_DIFF_LAST_10']


def main():
    games = pd.read_csv(DATA / 'nba_games_merged.csv', dtype={'GAME_ID': str})
    teams = pd.read_csv(DATA / 'nba_team_features.csv', dtype={'GAME_ID': str})
    if not teams['IS_HOME'].isin([True,False]).all():
        raise ValueError('IS_HOME must be Boolean')
    for side, mask in [('HOME', teams['IS_HOME']), ('AWAY', ~teams['IS_HOME'])]:
        selected = teams.loc[mask, ['GAME_ID','TEAM_ABBREVIATION'] + FEATURES].copy()
        selected = selected.rename(columns={
            'TEAM_ABBREVIATION': f'{side}_TEAM_CHECK',
            **{name: f'{side}_{name}' for name in FEATURES}
        })
        games = games.merge(selected, on='GAME_ID', how='left', validate='one_to_one')
        if games[f'{side}_TEAM_CHECK'].isna().any():
            raise ValueError(f'Missing {side} feature rows')
        if not games[f'{side}_TEAM'].eq(games[f'{side}_TEAM_CHECK']).all():
            raise ValueError(f'{side} team mismatch during join')
        games = games.drop(columns=f'{side}_TEAM_CHECK')
    games = add_elo_features(games)
    if games['GAME_ID'].duplicated().any():
        raise ValueError('Duplicate merged games')
    out = DATA / 'nba_model_dataset.csv'
    games.to_csv(out, index=False)
    print(f'Saved {len(games)} games to {out}')
    print('Missing values per feature:')
    print(games[[f'{side}_{f}' for side in ('HOME','AWAY') for f in FEATURES]].isna().sum())

if __name__ == '__main__':
    main()
