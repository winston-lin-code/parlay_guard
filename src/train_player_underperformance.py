"""Chronological, leakage-aware player underperformance model comparison.

Run from project root:
    python -u -m src.train_player_underperformance

Requires data/nba_player_underperformance_dataset.csv from:
    python -u -m src.player_underperformance_features

Evaluates expanding-window folds; never evaluates games on/after 2025-12-29.
Outputs per-fold metrics and summary CSVs to models/.
"""

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .common import DATA, MODELS

HOLDOUT_START = pd.Timestamp('2025-12-29')
SEED = 42
FEATURES = [
    'IS_HOME',
    'PLAYER_PPG_LAST_5',
    'PLAYER_PPG_LAST_10',
    'PLAYER_PTS_STD_LAST_10',
    'PLAYER_MIN_LAST_5',
    'PLAYER_MIN_LAST_10',
    'PLAYER_FGA_LAST_5',
    'PLAYER_FGA_LAST_10',
    'PLAYER_DAYS_REST',
    'OPP_PTS_ALLOWED_LAST_5',
    'OPP_PTS_ALLOWED_LAST_10',
    'TEAM_PTS_SCORED_LAST_10',
]
TARGET = 'UNDERPERFORMED'


def models():
    return {
        'majority_baseline': DummyClassifier(strategy='prior'),
        'logistic_regression': Pipeline([
            ('imputer', SimpleImputer(strategy='median', add_indicator=True)),
            ('scaler', StandardScaler()),
            ('model', LogisticRegression(max_iter=1500, random_state=SEED)),
        ]),
        'gradient_boosting': Pipeline([
            ('imputer', SimpleImputer(strategy='median', add_indicator=True)),
            ('model', HistGradientBoostingClassifier(
                max_iter=150, learning_rate=0.05, max_leaf_nodes=15,
                l2_regularization=1.0, random_state=SEED,
            )),
        ]),
    }


def metrics(y, probabilities):
    p = np.clip(np.asarray(probabilities, dtype=float), 1e-7, 1 - 1e-7)
    return {
        'log_loss': log_loss(y, p, labels=[0, 1]),
        'brier': brier_score_loss(y, p),
        'accuracy': accuracy_score(y, p >= 0.5),
        'roc_auc': roc_auc_score(y, p) if len(np.unique(y)) == 2 else np.nan,
    }


def main():
    path = DATA / 'nba_player_underperformance_dataset.csv'
    if not path.exists():
        raise FileNotFoundError(
            f'Missing {path}. Run python -u -m src.player_underperformance_features first.'
        )
    df = pd.read_csv(path, dtype={'GAME_ID': str})
    required = set(FEATURES + [TARGET, 'GAME_DATE', 'GAME_ID', 'PLAYER_ID', 'PLAYER_PPG_LAST_10'])
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f'Missing required columns: {sorted(missing)}')
    df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'], errors='raise').dt.normalize()
    if df.duplicated(['GAME_ID', 'PLAYER_ID']).any():
        raise ValueError('Duplicate GAME_ID / PLAYER_ID rows found')
    df[TARGET] = pd.to_numeric(df[TARGET], errors='raise').astype(int)
    if not df[TARGET].isin([0, 1]).all():
        raise ValueError('Target must contain only 0 and 1')
    for col in FEATURES:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.sort_values(['GAME_DATE', 'GAME_ID', 'PLAYER_ID']).reset_index(drop=True)

    holdout = df[df.GAME_DATE >= HOLDOUT_START]
    development = df[df.GAME_DATE < HOLDOUT_START]
    dates = np.array(sorted(development.GAME_DATE.unique()))
    if len(dates) < 30:
        raise ValueError('Insufficient unique pre-holdout dates for validation')
    # Expanding training windows; each validation window consists of complete dates.
    bounds = [int(len(dates) * f) for f in (0.40, 0.60, 0.80, 1.00)]
    if len(set(bounds)) != 4:
        raise ValueError('Fold boundaries overlap')

    print(f'Loaded {len(df):,} player-game rows across {df.GAME_ID.nunique():,} games')
    print(f'Final holdout starts {HOLDOUT_START.date()}: {len(holdout):,} rows excluded (NOT evaluated)')
    print(f'Development rows: {len(development):,}; underperformance rate: {development[TARGET].mean():.1%}')
    print('Predictive features:', ', '.join(FEATURES))
    print('Excluded outcomes and identifiers: ACTUAL_PTS, UNDERPERFORMED (target only), GAME_ID, PLAYER_ID')

    rows = []
    for i in range(3):
        train_end = pd.Timestamp(dates[bounds[i]])
        valid_end = pd.Timestamp(dates[bounds[i + 1]]) if i < 2 else HOLDOUT_START
        train = development[development.GAME_DATE < train_end]
        valid = development[(development.GAME_DATE >= train_end) & (development.GAME_DATE < valid_end)]
        if train.empty or valid.empty or train[TARGET].nunique() != 2 or valid[TARGET].nunique() != 2:
            raise ValueError(f'Fold {i+1} lacks sufficient examples of both classes')
        # No game is split between train and validation; dates are strictly separated.
        if set(train.GAME_ID).intersection(valid.GAME_ID):
            raise AssertionError('GAME_ID leakage between train and validation')
        print(f'\nFold {i+1}/3: train {len(train):,} rows before {train_end.date()}; '
              f'validate {len(valid):,} rows [{train_end.date()}, {valid_end.date()})', flush=True)
        x_train, y_train = train[FEATURES], train[TARGET]
        x_valid, y_valid = valid[FEATURES], valid[TARGET]
        for name, model in models().items():
            model.fit(x_train, y_train)
            probability = model.predict_proba(x_valid)[:, 1]
            result = metrics(y_valid, probability)
            rows.append({
                'fold': i + 1, 'model': name,
                'train_rows': len(train), 'validation_rows': len(valid),
                'validation_start': str(train_end.date()),
                'validation_end_exclusive': str(valid_end.date()),
                'validation_underperformance_rate': float(y_valid.mean()),
                **result,
            })
            print(f'  {name:21s} log_loss={result["log_loss"]:.5f} '
                  f'brier={result["brier"]:.5f} accuracy={result["accuracy"]:.2%} '
                  f'roc_auc={result["roc_auc"]:.4f}', flush=True)

    details = pd.DataFrame(rows)
    summary = details.groupby('model', as_index=False).agg(
        mean_log_loss=('log_loss', 'mean'),
        mean_brier=('brier', 'mean'),
        mean_accuracy=('accuracy', 'mean'),
        mean_roc_auc=('roc_auc', 'mean'),
    ).sort_values('mean_log_loss')
    print('\n===== PLAYER UNDERPERFORMANCE WALK-FORWARD SUMMARY =====')
    print(summary.to_string(index=False, float_format=lambda x: f'{x:.5f}'))
    print('\n===== PER-FOLD LOG LOSS =====')
    print(details.pivot(index='fold', columns='model', values='log_loss').to_string(float_format=lambda x: f'{x:.5f}'))
    print('\nNote: majority_baseline predicts the training-fold underperformance rate.')
    print('All rows on/after 2025-12-29 remain untouched. No final model is fitted or saved.')

    MODELS.mkdir(parents=True, exist_ok=True)
    details.to_csv(MODELS / 'player_underperformance_folds.csv', index=False)
    summary.to_csv(MODELS / 'player_underperformance_summary.csv', index=False)
    print('Saved models/player_underperformance_folds.csv and models/player_underperformance_summary.csv')


if __name__ == '__main__':
    main()
