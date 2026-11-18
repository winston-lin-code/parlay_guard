"""Compare pregame leading-scorer underperformance probabilities.

Run: python -u -m src.leading_scorer_underperformance
Uses existing scoring-dependency and player-feature CSVs; never evaluates final holdout.
Underperformance means ACTUAL_PTS < 0.8 * PLAYER_PPG_LAST_10.
All results are conditional on the preselected leading scorer appearing in the game.
"""

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from scipy.stats import norm

from .common import DATA, MODELS
from .leading_scorer_points import FEATURES, HOLDOUT_START, load_data, make_models

EPS = 1e-6


def classification_model():
    return Pipeline([
        ('imputer', SimpleImputer(strategy='median', add_indicator=True)),
        ('scaler', StandardScaler()),
        ('model', LogisticRegression(max_iter=2000, C=1.0, random_state=42)),
    ])


def metrics(y, p):
    p = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
    y = np.asarray(y, dtype=int)
    return {
        'log_loss': float(log_loss(y, p, labels=[0, 1])),
        'brier': float(brier_score_loss(y, p)),
        'accuracy': float(accuracy_score(y, p >= 0.5)),
        'roc_auc': float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else np.nan,
    }


def residual_probability(train, valid):
    """Use only past training dates to estimate a Ridge residual distribution.

    A Ridge model is fitted on the earlier 70% of training dates and its errors
    on the remaining 30% estimate scoring noise. A separate Ridge model fitted
    on all training rows makes validation point projections. No validation
    outcome enters either fit or residual calibration.
    """
    dates = np.array(sorted(train['GAME_DATE'].unique()))
    split_index = int(len(dates) * 0.70)
    if split_index <= 0 or split_index >= len(dates):
        raise ValueError('Insufficient training dates for residual calibration')
    split = pd.Timestamp(dates[split_index])
    earlier = train.loc[train.GAME_DATE < split]
    calibration = train.loc[train.GAME_DATE >= split]
    if len(earlier) < 100 or len(calibration) < 100:
        raise ValueError('Insufficient rows for residual calibration')

    ridge_early = make_models()['ridge_regression']
    ridge_early.fit(earlier[FEATURES], earlier['ACTUAL_PTS'])
    calibration_pred = ridge_early.predict(calibration[FEATURES])
    residuals = calibration['ACTUAL_PTS'].to_numpy(dtype=float) - calibration_pred
    # Empirical CDF avoids assuming normally distributed scoring residuals.
    sorted_residuals = np.sort(residuals)

    ridge_full = make_models()['ridge_regression']
    ridge_full.fit(train[FEATURES], train['ACTUAL_PTS'])
    projected_points = ridge_full.predict(valid[FEATURES])
    threshold = 0.8 * valid['PLAYER_PPG_LAST_10'].to_numpy(dtype=float)
    # Add-one smoothing prevents 0/1 probabilities on a finite calibration set.
    count_below = np.searchsorted(sorted_residuals, threshold - projected_points, side='left')
    probability = (count_below + 0.5) / (len(sorted_residuals) + 1.0)
    return probability, len(calibration), projected_points


def main():
    df, unmatched = load_data()
    df = df.copy()
    df['UNDERPERFORMED'] = (
        df['ACTUAL_PTS'] < 0.8 * df['PLAYER_PPG_LAST_10']
    ).astype(int)
    dev = df.loc[df.GAME_DATE < HOLDOUT_START].copy()
    held_out = df.loc[df.GAME_DATE >= HOLDOUT_START]

    full_dates = pd.read_csv(DATA / 'nba_player_underperformance_dataset.csv', usecols=['GAME_DATE'])
    dates = np.array(sorted(pd.to_datetime(full_dates['GAME_DATE']).dt.normalize().loc[
        lambda s: s < HOLDOUT_START].unique()))
    bounds = [int(len(dates) * f) for f in (0.40, 0.60, 0.80)]
    starts = [pd.Timestamp(dates[i]) for i in bounds]

    print(f'Eligible leading-scorer appearances: {len(df):,}; unmatched excluded: {unmatched:,}')
    print(f'Development: {len(dev):,}; final holdout excluded: {len(held_out):,}')
    print(f'Development underperformance rate: {dev.UNDERPERFORMED.mean():.1%}')
    print('Label: ACTUAL_PTS < 80% of pregame PLAYER_PPG_LAST_10')
    print('All predictions conditional on the selected star playing.')
    print('Final holdout starts 2025-12-29 and is NOT evaluated.')

    results = []
    prediction_rows = []
    for fold, start in enumerate(starts, 1):
        end = starts[fold] if fold < 3 else HOLDOUT_START
        train = dev.loc[dev.GAME_DATE < start]
        valid = dev.loc[(dev.GAME_DATE >= start) & (dev.GAME_DATE < end)]
        if len(train) < 200 or len(valid) < 100:
            raise ValueError(f'Fold {fold} too small')
        if set(train.GAME_ID) & set(valid.GAME_ID):
            raise AssertionError('Game overlap between train and validation')

        baseline_p = np.full(len(valid), train.UNDERPERFORMED.mean(), dtype=float)
        classifier = classification_model()
        classifier.fit(train[FEATURES], train.UNDERPERFORMED)
        logistic_p = classifier.predict_proba(valid[FEATURES])[:, 1]
        ridge_p, calibration_n, ridge_points = residual_probability(train, valid)

        preds = {
            'constant_baseline': baseline_p,
            'direct_logistic': logistic_p,
            'ridge_residual_cdf': ridge_p,
        }
        print(f'\nFold {fold}/3: train={len(train):,}, validate={len(valid):,} '
              f'[{start.date()}, {end.date()}); ridge calibration={calibration_n:,}', flush=True)
        for name, p in preds.items():
            scored = metrics(valid.UNDERPERFORMED, p)
            results.append({'fold': fold, 'model': name, 'train_rows': len(train),
                            'validation_rows': len(valid), **scored})
            print(f'  {name:20s} log_loss={scored["log_loss"]:.5f} '
                  f'brier={scored["brier"]:.5f} '
                  f'accuracy={scored["accuracy"]:.2%} '
                  f'roc_auc={scored["roc_auc"]:.4f}', flush=True)

        for name, p in preds.items():
            prediction_rows.append(pd.DataFrame({
                'GAME_ID': valid.GAME_ID.to_numpy(),
                'GAME_DATE': valid.GAME_DATE.to_numpy(),
                'TEAM_ID': valid.TEAM_ID.to_numpy(),
                'STAR_PLAYER_ID': valid.STAR_PLAYER_ID.to_numpy(),
                'fold': fold, 'model': name,
                'actual_underperformed': valid.UNDERPERFORMED.to_numpy(),
                'predicted_probability': p,
            }))

    fold_df = pd.DataFrame(results)
    summary = (fold_df.groupby('model', as_index=False)
               .agg(mean_log_loss=('log_loss', 'mean'), mean_brier=('brier', 'mean'),
                    mean_accuracy=('accuracy', 'mean'), mean_roc_auc=('roc_auc', 'mean'),
                    validation_rows=('validation_rows', 'sum'))
               .sort_values('mean_log_loss'))
    print('\n===== LEADING SCORER UNDERPERFORMANCE SUMMARY =====')
    print(summary.to_string(index=False, float_format=lambda x: f'{x:.5f}'))
    print('\n===== PER-FOLD LOG LOSS =====')
    print(fold_df.pivot(index='fold', columns='model', values='log_loss').to_string(
        float_format=lambda x: f'{x:.5f}'))

    MODELS.mkdir(parents=True, exist_ok=True)
    fold_df.to_csv(MODELS / 'leading_scorer_underperformance_folds.csv', index=False)
    summary.to_csv(MODELS / 'leading_scorer_underperformance_summary.csv', index=False)
    pd.concat(prediction_rows, ignore_index=True).to_csv(
        MODELS / 'leading_scorer_underperformance_oof_predictions.csv', index=False)
    print('\nSaved fold, summary, and out-of-sample predictions CSVs in models/.')
    print('Final holdout NOT evaluated; no final model fitted or saved.')


if __name__ == '__main__':
    main()
