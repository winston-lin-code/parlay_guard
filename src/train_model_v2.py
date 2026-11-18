"""Train a versioned Parlay Guard model using historical + current-season completed games.

Run: python -u -m src.train_model_v2 --cutoff 2026-11-18
Cutoff is exclusive; never uses games on/after that date.
"""
import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .common import DATA, MODELS
from .feature_engineering import build_features
from .elo import add_elo_features

FEATURES = [f'{side}_{name}' for side in ('HOME', 'AWAY') for name in
            ('WIN_PCT_LAST_5', 'WIN_PCT_LAST_10', 'DAYS_REST', 'POINT_DIFF_LAST_10')]
INPUTS = FEATURES + ['ELO_DIFF']
REQUIRED = {'GAME_ID', 'GAME_DATE', 'TEAM_ID', 'TEAM_ABBREVIATION', 'MATCHUP', 'WL', 'PTS'}


def read_rows(path, cutoff):
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path, dtype={'GAME_ID': str, 'TEAM_ID': str})
    missing = REQUIRED - set(df.columns)
    if missing:
        raise ValueError(f'{path}: missing {sorted(missing)}')
    df = df.copy()
    df['GAME_ID'] = df['GAME_ID'].astype(str).str.strip().str.zfill(10)
    df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'], errors='raise').dt.normalize()
    df = df[df['GAME_DATE'] < cutoff].copy()
    df['PTS'] = pd.to_numeric(df['PTS'], errors='raise')
    df['TEAM_ABBREVIATION'] = df['TEAM_ABBREVIATION'].astype(str).str.strip().str.upper()
    df['WL'] = df['WL'].astype(str).str.strip().str.upper()
    if not df['WL'].isin(['W', 'L']).all():
        raise ValueError(f'{path}: invalid WL values')
    return df


def validate_and_label(df):
    if df.duplicated(['GAME_ID', 'TEAM_ID']).any():
        raise ValueError('Duplicate team rows after combining data')
    # The @ format can be duplicated on both team rows in some NBA exports.
    # Parse the actual home abbreviation, not whether the row contains @.
    def home_team(row):
        m = str(row.MATCHUP).strip().upper()
        if ' @ ' in m:
            return m.split(' @ ', 1)[1].strip()
        if ' VS. ' in m:
            return m.split(' VS. ', 1)[0].strip()
        raise ValueError(f'Unknown MATCHUP format in {row.GAME_ID}: {row.MATCHUP}')
    df = df.copy()
    df['IS_HOME'] = [team == home_team(row) for team, row in zip(df['TEAM_ABBREVIATION'], df.itertuples(index=False))]
    df['IS_WIN'] = df['WL'].eq('W')
    counts = df.groupby('GAME_ID').agg(rows=('TEAM_ID', 'size'), homes=('IS_HOME', 'sum'), wins=('IS_WIN', 'sum'), dates=('GAME_DATE', 'nunique'))
    bad = counts[(counts.rows != 2) | (counts.homes != 1) | (counts.wins != 1) | (counts.dates != 1)]
    if not bad.empty:
        raise ValueError(f'Incomplete/inconsistent games: {bad.head(8).to_dict("index")}')
    return df


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cutoff', required=True, help='Exclusive YYYY-MM-DD cutoff')
    p.add_argument('--historical', default=str(DATA / 'nba_games_updated.csv'))
    p.add_argument('--season-csv', default=str(DATA / 'nba_2026_27_latest_raw.csv'))
    p.add_argument('--output', default=str(MODELS / 'elo_2026_27_v2.joblib'))
    args = p.parse_args()
    cutoff = pd.Timestamp(args.cutoff).normalize()
    historical = read_rows(Path(args.historical), cutoff)
    season = read_rows(Path(args.season_csv), cutoff)
    if season.empty:
        raise ValueError('No current-season games before cutoff')
    overlap = set(historical.GAME_ID) & set(season.GAME_ID)
    if overlap:
        print(f'Historical data overlaps current season in {len(overlap)} games; using season CSV for overlapping games.')
        historical = historical[~historical.GAME_ID.isin(overlap)]
    combined = validate_and_label(pd.concat([historical, season], ignore_index=True))
    print(f'Validated {combined.GAME_ID.nunique()} games; newest {combined.GAME_DATE.max().date()}')
    teams = build_features(combined)
    home = teams[teams.IS_HOME].copy()
    away = teams[~teams.IS_HOME].copy()
    games = home[['GAME_ID', 'GAME_DATE', 'TEAM_ABBREVIATION', 'IS_WIN', 'PTS']].rename(columns={
        'TEAM_ABBREVIATION': 'HOME_TEAM', 'IS_WIN': 'HOME_WIN', 'PTS': 'HOME_PTS'})
    games = games.merge(away[['GAME_ID', 'TEAM_ABBREVIATION', 'PTS']].rename(columns={
        'TEAM_ABBREVIATION': 'AWAY_TEAM', 'PTS': 'AWAY_PTS'}), on='GAME_ID', validate='one_to_one')
    for side, side_rows in [('HOME', home), ('AWAY', away)]:
        cols = ['GAME_ID'] + [f.replace(f'{side}_', '') for f in FEATURES if f.startswith(f'{side}_')]
        extra = side_rows[cols].rename(columns={c: f'{side}_{c}' for c in cols if c != 'GAME_ID'})
        games = games.merge(extra, on='GAME_ID', validate='one_to_one')
    # SEASON_ID = 22026 -> 2026; infer earlier seasons from season id if available,
    # otherwise NBA season start year (Oct-Dec => current calendar year).
    if 'SEASON_ID' in home.columns:
        season_lookup = home[['GAME_ID', 'SEASON_ID']].copy()
        season_lookup['SEASON'] = season_lookup.SEASON_ID.astype(str).str[-4:]
        games = games.merge(season_lookup[['GAME_ID', 'SEASON']], on='GAME_ID', validate='one_to_one')
    else:
        games['SEASON'] = (games.GAME_DATE.dt.year - (games.GAME_DATE.dt.month < 7).astype(int)).astype(str)
    # If historic SEASON_ID is missing but SEASON exists, preserve original season labels.
    games = add_elo_features(games)
    games = games.sort_values(['GAME_DATE', 'GAME_ID']).reset_index(drop=True)
    if games[INPUTS].isna().all(axis=0).any():
        raise ValueError('An entire model feature is missing')
    model = Pipeline([
        ('preprocessor', ColumnTransformer([('numeric', Pipeline([
            ('imputer', SimpleImputer(strategy='median', add_indicator=True)),
            ('scaler', StandardScaler()),
        ]), INPUTS)], sparse_threshold=0)),
        ('classifier', GradientBoostingClassifier(n_estimators=100, learning_rate=0.05,
                                                  max_depth=2, random_state=42)),
    ])
    model.fit(games[INPUTS], games.HOME_WIN.astype(int))
    out = Path(args.output)
    if out.exists():
        raise FileExistsError(f'{out} already exists; choose a new versioned output path')
    out.parent.mkdir(parents=True, exist_ok=True)
    artifact = {'model': model, 'inputs': INPUTS, 'last_historical_date': str(games.GAME_DATE.max().date()),
                'training_cutoff_exclusive': str(cutoff.date()), 'training_games': int(len(games)),
                'model_version': '2026_27_v2'}
    joblib.dump(artifact, out)
    summary = {k: v for k, v in artifact.items() if k != 'model'}
    out.with_suffix('.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f'Saved {out}. This is a new trained model, NOT an independently evaluated model.')
    print('Original frozen model and saved forecasts were not changed.')


if __name__ == '__main__':
    main()
