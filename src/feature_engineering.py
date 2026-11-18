"""Leakage-safe pregame rolling team features."""
import pandas as pd
from .common import DATA

WINDOWS = (5, 10)


def build_features(df):
    df = df.copy()
    df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
    df = df.sort_values(['TEAM_ID','GAME_DATE','GAME_ID']).reset_index(drop=True)
    df['IS_WIN'] = df['WL'].eq('W')
    for n in WINDOWS:
        df[f'WIN_PCT_LAST_{n}'] = df.groupby('TEAM_ID')['IS_WIN'].transform(
            lambda s: s.shift(1).rolling(n, min_periods=n).mean()
        )
    # Derive scoring margin from the two PTS values for each game.
    # This avoids depending on the optional PLUS_MINUS field.
    if 'PTS' in df.columns:
        pts = pd.to_numeric(df['PTS'], errors='coerce')
        counts = df.groupby('GAME_ID')['GAME_ID'].transform('size')
        if not counts.eq(2).all():
            raise ValueError('Point differential requires exactly two team rows per GAME_ID')
        opponent_pts = pts.groupby(df['GAME_ID']).transform('sum') - pts
        df['POINT_DIFF'] = pts - opponent_pts
        df['POINT_DIFF_LAST_10'] = df.groupby('TEAM_ID')['POINT_DIFF'].transform(
            lambda values: values.shift(1).rolling(10, min_periods=10).mean()
        )
    # Prior games only; same-day doubleheaders are not disambiguated by tipoff time.
    df['DAYS_REST'] = df.groupby('TEAM_ID')['GAME_DATE'].diff().dt.days.sub(1).clip(lower=0)
    return df


def main():
    df = pd.read_csv(DATA / 'nba_games_updated.csv', dtype={'GAME_ID': str})
    df = build_features(df)
    if 'POINT_DIFF_LAST_10' not in df.columns:
        raise ValueError('Missing PTS column required for point differential')
    for n in WINDOWS:
        values = df[f'WIN_PCT_LAST_{n}'].dropna()
        assert values.between(0, 1).all()
    out = DATA / 'nba_team_features.csv'
    df.to_csv(out, index=False)
    print('Missing rolling values:')
    print(df[[f'WIN_PCT_LAST_{n}' for n in WINDOWS]].isna().sum())
    print(f'Saved {len(df)} rows to {out}')

if __name__ == '__main__':
    main()
