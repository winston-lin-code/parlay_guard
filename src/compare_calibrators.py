"""Leakage-safe walk-forward comparison of raw, Platt and isotonic probabilities.

Run: python -u -m src.compare_calibrators
The final 15% held-out test period is never evaluated.
"""
import json
import time

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from .common import DATA, MODELS
from .walk_forward import make_model, score


def ece(y, p, n_bins=10):
    y, p = np.asarray(y), np.asarray(p)
    indices = np.minimum((p * n_bins).astype(int), n_bins - 1)
    return float(sum(
        np.sum(indices == b) / len(y) * abs(y[indices == b].mean() - p[indices == b].mean())
        for b in range(n_bins) if np.any(indices == b)
    ))


def logit(p):
    p = np.clip(np.asarray(p), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p)).reshape(-1, 1)


def main():
    path = DATA / 'nba_model_dataset.csv'
    if not path.exists():
        raise FileNotFoundError(f'Missing {path}; rebuild features first.')
    df = pd.read_csv(path, dtype={'GAME_ID': str})
    df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
    df = df.sort_values(['GAME_DATE', 'GAME_ID']).reset_index(drop=True)
    dates = np.array(sorted(df['GAME_DATE'].unique()))
    boundaries = [int(len(dates) * f) for f in (0.40, 0.55, 0.70, 0.85)]
    if len(set(boundaries)) != 4:
        raise ValueError('Insufficient distinct dates for three folds')
    test_start = pd.Timestamp(dates[boundaries[-1]])
    print(f'Final held-out test starts {test_start.date()} (never evaluated)', flush=True)

    fold_rows, predictions = [], []
    for fold in range(3):
        valid_start = pd.Timestamp(dates[boundaries[fold]])
        valid_end = pd.Timestamp(dates[boundaries[fold + 1]])
        prior_dates = dates[:boundaries[fold]]
        # Reserve the most recent 20% of *prior* dates for fitting calibrators.
        calibration_start = pd.Timestamp(prior_dates[int(len(prior_dates) * 0.80)])
        train = df[df['GAME_DATE'] < calibration_start]
        calibrate = df[(df['GAME_DATE'] >= calibration_start) & (df['GAME_DATE'] < valid_start)]
        validate = df[(df['GAME_DATE'] >= valid_start) & (df['GAME_DATE'] < valid_end)]
        if min(len(train), len(calibrate), len(validate)) < 100:
            raise ValueError(f'Fold {fold+1} has too few games')
        for part, label in [(train, 'train'), (calibrate, 'calibration')]:
            if part['HOME_WIN'].nunique() != 2:
                raise ValueError(f'Fold {fold+1}: {label} lacks both outcomes')

        model, inputs = make_model(include_team_identity=False, include_point_diff=True,elo_mode="difference")
        print("Model features:", inputs)
        missing = set(inputs) - set(df.columns)
        if missing:
            raise ValueError(f'Missing features {sorted(missing)}; rebuild features first')
        start = time.perf_counter()
        model.fit(train[inputs], train['HOME_WIN'].astype(int))
        p_cal = model.predict_proba(calibrate[inputs])[:, 1]
        p_val = model.predict_proba(validate[inputs])[:, 1]
        y_cal = calibrate['HOME_WIN'].astype(int).to_numpy()
        y_val = validate['HOME_WIN'].astype(int).to_numpy()

        # Platt scaling: sigmoid of the model's log-odds.
        platt = LogisticRegression(C=1.0, max_iter=1000)
        platt.fit(logit(p_cal), y_cal)
        p_platt = platt.predict_proba(logit(p_val))[:, 1]

        # Isotonic calibration: monotonic mapping fitted ONLY on calibration games.
        isotonic = IsotonicRegression(out_of_bounds='clip')
        isotonic.fit(p_cal, y_cal)
        p_iso = isotonic.predict(p_val)

        print(f'\nFold {fold+1}/3 | train {len(train)} | calibrate {len(calibrate)} '
              f'| evaluate {len(validate)} | {time.perf_counter()-start:.2f}s', flush=True)
        print(f'  train before {calibration_start.date()} | calibrate before {valid_start.date()}'
              f' | evaluate before {valid_end.date()}')
        for method, probs in [('raw', p_val), ('platt', p_platt), ('isotonic', p_iso)]:
            metrics = score(y_val, probs)
            calibration_error = ece(y_val, np.asarray(probs))
            fold_rows.append({
                'fold': fold+1, 'method': method,
                'train_games': len(train), 'calibration_games': len(calibrate),
                'evaluation_games': len(validate),
                'train_end_exclusive': str(calibration_start.date()),
                'calibration_end_exclusive': str(valid_start.date()),
                'evaluation_end_exclusive': str(valid_end.date()),
                **metrics, 'ece_10_bins': calibration_error,
            })
            for game_id, date, outcome, probability in zip(
                validate['GAME_ID'], validate['GAME_DATE'], y_val, probs
            ):
                predictions.append({
                    'fold': fold+1, 'method': method, 'GAME_ID': game_id,
                    'GAME_DATE': str(date.date()), 'actual_home_win': int(outcome),
                    'predicted_probability': float(probability),
                })
            print(f'  {method:8s} log_loss={metrics["log_loss"]:.5f} '
                  f'brier={metrics["brier"]:.5f} ece={calibration_error:.5f} '
                  f'accuracy={metrics["accuracy"]:.2%}', flush=True)

    details = pd.DataFrame(fold_rows)
    prediction_df = pd.DataFrame(predictions)
    summary = []
    for method, group in prediction_df.groupby('method', sort=False):
        metrics = score(group['actual_home_win'].to_numpy(), group['predicted_probability'].to_numpy())
        summary.append({'method': method, 'games': len(group), **metrics,
                        'ece_10_bins': ece(group['actual_home_win'], group['predicted_probability']),
                        'mean_fold_log_loss': float(details.loc[details['method'] == method, 'log_loss'].mean())})
    summary_df = pd.DataFrame(summary).sort_values('log_loss')
    print('\n===== POOLED OUT-OF-TIME CALIBRATION COMPARISON =====')
    print(summary_df.to_string(index=False, float_format=lambda v: f'{v:.5f}'))
    print('\n===== PER-FOLD LOG LOSS (lower is better) =====')
    print(details.pivot(index='fold', columns='method', values='log_loss').to_string(float_format=lambda v: f'{v:.5f}'))
    print('\nNote: raw probabilities here use less training data than the earlier calibration_analysis.py')
    print('so comparisons should be made WITHIN this script, on identical evaluation games.')
    print('ECE depends on binning; do not select a method using ECE alone.')
    MODELS.mkdir(parents=True, exist_ok=True)
    details.to_csv(MODELS / 'calibrator_comparison_folds.csv', index=False)
    summary_df.to_csv(MODELS / 'calibrator_comparison_summary.csv', index=False)
    prediction_df.to_csv(MODELS / 'calibrator_comparison_predictions.csv', index=False)
    (MODELS / 'calibrator_comparison_report.json').write_text(json.dumps({
        'held_out_test_start': str(test_start.date()),
        'summary': summary_df.to_dict(orient='records'),
        'folds': details.to_dict(orient='records'),
        'note': 'Calibrators fitted on preceding calibration windows only; no final test evaluation.'
    }, indent=2), encoding='utf-8')
    print('\nSaved comparison CSVs and JSON in models/. Final held-out test NOT evaluated.')


if __name__ == '__main__':
    main()
