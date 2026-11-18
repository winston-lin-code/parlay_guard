"""Compare pregame leading-scorer point projections using walk-forward validation.

Run: python -u -m src.leading_scorer_points
Prerequisites:
    python -u -m src.scoring_dependency
    python -u -m src.player_underperformance_features

The star is chosen from earlier team games, never the current game's box score.
Only prior-game player features are used for prediction. Final holdout is untouched.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .common import DATA, MODELS

HOLDOUT_START = pd.Timestamp('2025-12-29')
FEATURES = [
    'IS_HOME', 'PLAYER_PPG_LAST_5', 'PLAYER_PPG_LAST_10',
    'PLAYER_PTS_STD_LAST_10', 'PLAYER_MIN_LAST_5', 'PLAYER_MIN_LAST_10',
    'PLAYER_FGA_LAST_5', 'PLAYER_FGA_LAST_10', 'PLAYER_DAYS_REST',
    'OPP_PTS_ALLOWED_LAST_5', 'OPP_PTS_ALLOWED_LAST_10',
    'TEAM_PTS_SCORED_LAST_10', 'TOP_SCORER_SHARE_L10',
]


def load_data():
    star_path = DATA / 'nba_scoring_dependency.csv'
    player_path = DATA / 'nba_player_underperformance_dataset.csv'
    for path, command in [(star_path, 'src.scoring_dependency'),
                          (player_path, 'src.player_underperformance_features')]:
        if not path.exists():
            raise FileNotFoundError(f'Missing {path}. First run python -u -m {command}')
    stars = pd.read_csv(star_path, dtype={'GAME_ID': str})
    players = pd.read_csv(player_path, dtype={'GAME_ID': str})
    star_cols = ['GAME_ID', 'TEAM_ID', 'STAR_PLAYER_ID', 'STAR_PLAYED',
                 'TOP_SCORER_SHARE_L10']
    player_cols = ['GAME_ID', 'TEAM_ID', 'PLAYER_ID', 'GAME_DATE', 'PLAYER_NAME',
                   'ACTUAL_PTS'] + [f for f in FEATURES if f != 'TOP_SCORER_SHARE_L10']
    for name, frame, cols in [('star', stars, star_cols), ('player', players, player_cols)]:
        missing = set(cols) - set(frame.columns)
        if missing:
            raise ValueError(f'{name} data missing columns: {sorted(missing)}')
    for frame in (stars, players):
        frame['GAME_ID'] = frame['GAME_ID'].astype(str).str.zfill(10)
        frame['TEAM_ID'] = pd.to_numeric(frame['TEAM_ID'], errors='raise').astype('int64')
    stars['STAR_PLAYER_ID'] = pd.to_numeric(stars['STAR_PLAYER_ID'], errors='raise').astype('int64')
    players['PLAYER_ID'] = pd.to_numeric(players['PLAYER_ID'], errors='raise').astype('int64')
    if stars.duplicated(['GAME_ID', 'TEAM_ID']).any():
        raise ValueError('Duplicate star selection per team-game')
    if players.duplicated(['GAME_ID', 'TEAM_ID', 'PLAYER_ID']).any():
        raise ValueError('Duplicate player-game')
    if stars['STAR_PLAYED'].dtype != bool:
        stars['STAR_PLAYED'] = stars['STAR_PLAYED'].astype(str).str.lower().map({'true': True, 'false': False})
        if stars['STAR_PLAYED'].isna().any():
            raise ValueError('Invalid STAR_PLAYED values')
    stars = stars.loc[stars['STAR_PLAYED'], star_cols].copy()
    merged = stars.merge(players[player_cols], how='left',
                         left_on=['GAME_ID', 'TEAM_ID', 'STAR_PLAYER_ID'],
                         right_on=['GAME_ID', 'TEAM_ID', 'PLAYER_ID'],
                         validate='one_to_one', indicator=True)
    unmatched = int((merged['_merge'] != 'both').sum())
    merged = merged.loc[merged['_merge'] == 'both'].drop(columns=['_merge'])
    merged['GAME_DATE'] = pd.to_datetime(merged['GAME_DATE'], errors='raise').dt.normalize()
    for feature in FEATURES + ['ACTUAL_PTS']:
        merged[feature] = pd.to_numeric(merged[feature], errors='coerce')
    merged = merged.dropna(subset=['ACTUAL_PTS', 'PLAYER_PPG_LAST_10'])
    merged = merged.sort_values(['GAME_DATE', 'GAME_ID', 'TEAM_ID']).reset_index(drop=True)
    return merged, unmatched


def make_models():
    return {
        'ridge_regression': Pipeline([
            ('imputer', SimpleImputer(strategy='median', add_indicator=True)),
            ('scaler', StandardScaler()),
            ('model', Ridge(alpha=100.0)),
        ]),
        'gradient_boosting': Pipeline([
            ('imputer', SimpleImputer(strategy='median', add_indicator=True)),
            ('model', HistGradientBoostingRegressor(
                max_iter=150, learning_rate=0.05, max_leaf_nodes=15,
                l2_regularization=10.0, random_state=42,
            )),
        ]),
    }


def score(y, prediction):
    prediction = np.maximum(np.asarray(prediction, dtype=float), 0.0)
    return {
        'mae': float(mean_absolute_error(y, prediction)),
        'rmse': float(np.sqrt(mean_squared_error(y, prediction))),
        'bias': float(np.mean(prediction - np.asarray(y, dtype=float))),
    }


def main():
    df, unmatched = load_data()
    holdout = df.loc[df.GAME_DATE >= HOLDOUT_START]
    dev = df.loc[df.GAME_DATE < HOLDOUT_START]
    if dev.empty:
        raise ValueError('No pre-holdout leading scorer appearances')
    # Use the same date boundaries as the full player underperformance experiment,
    # not a new split based on the reduced star-only dataset.
    full = pd.read_csv(DATA / 'nba_player_underperformance_dataset.csv', usecols=['GAME_DATE'])
    all_dates = np.array(sorted(pd.to_datetime(full['GAME_DATE']).dt.normalize().loc[
        lambda s: s < HOLDOUT_START].unique()))
    bounds = [int(len(all_dates) * f) for f in (0.40, 0.60, 0.80)]
    if len(set(bounds)) != 3:
        raise ValueError('Insufficient dates for three distinct folds')
    starts = [pd.Timestamp(all_dates[i]) for i in bounds]
    print(f'Eligible leading-scorer appearances: {len(df):,} across {df.GAME_ID.nunique():,} games')
    print(f'Star selections without matching player feature rows: {unmatched:,} (excluded)')
    print(f'Development rows: {len(dev):,}; final holdout rows NOT evaluated: {len(holdout):,}')
    print(f'Final holdout begins {HOLDOUT_START.date()}')
    print('Features:', ', '.join(FEATURES))
    print('Target ACTUAL_PTS is NEVER used as a predictive feature.')
    print('Only games in which the preselected leading scorer actually played are evaluated.')
    results = []
    for i, start in enumerate(starts):
        end = starts[i + 1] if i < 2 else HOLDOUT_START
        train = dev.loc[dev.GAME_DATE < start]
        valid = dev.loc[(dev.GAME_DATE >= start) & (dev.GAME_DATE < end)]
        if len(train) < 100 or len(valid) < 100:
            raise ValueError(f'Fold {i + 1} too small: train={len(train)}, valid={len(valid)}')
        if set(train.GAME_ID) & set(valid.GAME_ID):
            raise AssertionError('Same game appears in training and validation')
        print(f'\nFold {i + 1}/3: train={len(train):,}, validate={len(valid):,} '
              f'[{start.date()}, {end.date()})', flush=True)
        y_train, y_valid = train['ACTUAL_PTS'], valid['ACTUAL_PTS']
        predictions = {'rolling_10_ppg': valid['PLAYER_PPG_LAST_10'].to_numpy()}
        for name, model in make_models().items():
            model.fit(train[FEATURES], y_train)
            predictions[name] = model.predict(valid[FEATURES])
        for name, pred in predictions.items():
            metric = score(y_valid, pred)
            results.append({'fold': i + 1, 'model': name, 'train_rows': len(train),
                            'validation_rows': len(valid), 'validation_start': start.date(),
                            'validation_end_exclusive': end.date(), **metric})
            print(f'  {name:20s} MAE={metric["mae"]:.3f} points '
                  f'RMSE={metric["rmse"]:.3f} bias={metric["bias"]:+.3f}', flush=True)
    folds = pd.DataFrame(results)
    summary = (folds.groupby('model', as_index=False)
               .agg(mean_mae=('mae', 'mean'), mean_rmse=('rmse', 'mean'),
                    mean_bias=('bias', 'mean'), validation_rows=('validation_rows', 'sum'))
               .sort_values('mean_mae'))
    print('\n===== LEADING SCORER POINTS WALK-FORWARD SUMMARY =====')
    print(summary.to_string(index=False, float_format=lambda x: f'{x:.4f}'))
    print('\n===== PER-FOLD MAE (LOWER IS BETTER) =====')
    print(folds.pivot(index='fold', columns='model', values='mae').to_string(float_format=lambda x: f'{x:.3f}'))
    MODELS.mkdir(parents=True, exist_ok=True)
    folds.to_csv(MODELS / 'leading_scorer_points_folds.csv', index=False)
    summary.to_csv(MODELS / 'leading_scorer_points_summary.csv', index=False)
    print('\nSaved models/leading_scorer_points_folds.csv and models/leading_scorer_points_summary.csv')
    print('Final holdout NOT evaluated. These predictions are conditional on the star playing.')


if __name__ == '__main__':
    main()
