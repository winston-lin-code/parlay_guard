"""Leakage-safe NBA Elo ratings: pregame values, daily batched updates.

Defaults are experimental, not fitted to the validation or test periods.
"""
import pandas as pd

INITIAL_ELO = 1500.0
K_FACTOR = 20.0
HOME_ADVANTAGE = 65.0
OFFSEASON_REGRESSION = 0.25


def add_elo_features(games):
    required = {'GAME_ID', 'GAME_DATE', 'SEASON', 'HOME_TEAM', 'AWAY_TEAM', 'HOME_WIN'}
    missing = required - set(games.columns)
    if missing:
        raise ValueError(f'Missing game columns for Elo: {sorted(missing)}')
    df = games.copy()
    df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
    if df['GAME_ID'].duplicated().any():
        raise ValueError('Elo requires one row per GAME_ID')
    df = df.sort_values(['GAME_DATE', 'GAME_ID']).reset_index(drop=True)
    ratings = {}
    previous_season = None
    records = []

    # Never use same-day results to create another same-day pregame feature.
    for date, day in df.groupby('GAME_DATE', sort=True):
        seasons = day['SEASON'].astype(str).unique()
        if len(seasons) != 1:
            raise ValueError(f'Multiple seasons on {date.date()}: {seasons}')
        season = seasons[0]
        if previous_season is not None and season != previous_season:
            ratings = {team: INITIAL_ELO + (r - INITIAL_ELO) * (1 - OFFSEASON_REGRESSION)
                       for team, r in ratings.items()}
        previous_season = season
        updates = {}
        for row in day.itertuples(index=False):
            home, away = row.HOME_TEAM, row.AWAY_TEAM
            if home == away:
                raise ValueError(f'Identical teams in game {row.GAME_ID}')
            home_rating = ratings.get(home, INITIAL_ELO)
            away_rating = ratings.get(away, INITIAL_ELO)
            diff = home_rating - away_rating
            expected = 1 / (1 + 10 ** (-(diff + HOME_ADVANTAGE) / 400))
            records.append({
                'GAME_ID': row.GAME_ID,
                'HOME_ELO': home_rating,
                'AWAY_ELO': away_rating,
                'ELO_DIFF': diff,
            })
            outcome = int(row.HOME_WIN)
            if outcome not in (0, 1):
                raise ValueError(f'Invalid HOME_WIN for {row.GAME_ID}')
            change = K_FACTOR * (outcome - expected)
            updates[home] = updates.get(home, 0.0) + change
            updates[away] = updates.get(away, 0.0) - change
        for team, change in updates.items():
            ratings[team] = ratings.get(team, INITIAL_ELO) + change

    return games.merge(pd.DataFrame(records), on='GAME_ID', how='left', validate='one_to_one')
