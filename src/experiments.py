"""Compare NBA win probability models on a chronological validation split.

Run from project root: python -m src.experiments
The held-out test period is never used for model selection.
"""
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,
                              HistGradientBoostingClassifier)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .common import DATA, MODELS
from .train import INPUTS, NUMERIC, CATEGORICAL


def make_preprocessor():
    return ColumnTransformer(
        [
            ('num', Pipeline([
                ('impute', SimpleImputer(strategy='median', add_indicator=True)),
                ('scale', StandardScaler()),
            ]), NUMERIC),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), CATEGORICAL),
        ],
        sparse_threshold=0,
    )


def evaluate(y, probabilities):
    probabilities = np.clip(np.asarray(probabilities), 1e-7, 1 - 1e-7)
    return {
        'accuracy': float(accuracy_score(y, probabilities >= 0.5)),
        'brier': float(brier_score_loss(y, probabilities)),
        'log_loss': float(log_loss(y, probabilities, labels=[0, 1])),
        'roc_auc': float(roc_auc_score(y, probabilities)),
    }


def main():
    df = pd.read_csv(DATA / 'nba_model_dataset.csv', dtype={'GAME_ID': str})
    df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
    df = df.sort_values(['GAME_DATE', 'GAME_ID']).reset_index(drop=True)
    dates = np.array(sorted(df['GAME_DATE'].unique()))
    if len(dates) < 30:
        raise ValueError('Not enough unique game dates')

    first_cut = dates[int(len(dates) * 0.70)]
    second_cut = dates[int(len(dates) * 0.85)]
    train = df[df['GAME_DATE'] < first_cut]
    valid = df[(df['GAME_DATE'] >= first_cut) & (df['GAME_DATE'] < second_cut)]
    test = df[df['GAME_DATE'] >= second_cut]
    if min(len(train), len(valid), len(test)) == 0:
        raise ValueError('Empty train/validation/test split')

    x_train, y_train = train[INPUTS], train['HOME_WIN'].astype(int)
    x_valid, y_valid = valid[INPUTS], valid['HOME_WIN'].astype(int)

    candidates = {
        'logistic_C_0.1': LogisticRegression(C=0.1, max_iter=3000),
        'logistic_C_1': LogisticRegression(C=1.0, max_iter=3000),
        'logistic_C_10': LogisticRegression(C=10.0, max_iter=3000),
        'random_forest': RandomForestClassifier(
            n_estimators=300, max_depth=6, min_samples_leaf=10,
            random_state=42, n_jobs=-1),
        'gradient_boosting': GradientBoostingClassifier(
            n_estimators=100, learning_rate=0.05, max_depth=2,
            random_state=42),
        'hist_gradient_boosting': HistGradientBoostingClassifier(
            max_iter=120, learning_rate=0.04, max_leaf_nodes=15,
            l2_regularization=5.0, random_state=42),
    }

    baseline_p = float(y_train.mean())
    results = [{
        'model': 'train_rate_baseline',
        **evaluate(y_valid, np.full(len(y_valid), baseline_p)),
    }]
    best_model = None
    best_name = None
    best_loss = float('inf')

    for name, estimator in candidates.items():
        model = Pipeline([
            ('prep', make_preprocessor()),
            ('classifier', estimator),
        ])
        model.fit(x_train, y_train)
        probabilities = model.predict_proba(x_valid)[:, 1]
        scores = evaluate(y_valid, probabilities)
        results.append({'model': name, **scores})
        print(f"{name:24s} accuracy={scores['accuracy']:.3f} "
              f"brier={scores['brier']:.4f} log_loss={scores['log_loss']:.4f}")
        if scores['log_loss'] < best_loss:
            best_loss = scores['log_loss']
            best_name = name
            best_model = model

    results.sort(key=lambda row: row['log_loss'])
    print('\nValidation ranking (lower log loss is better):')
    print(pd.DataFrame(results).to_string(index=False, float_format=lambda v: f'{v:.4f}'))
    print(f'\nBest trained model: {best_name}')
    print('Held-out test set was NOT evaluated during this experiment.')

    report = {
        'split_dates': {
            'train_before': str(pd.Timestamp(first_cut).date()),
            'test_from': str(pd.Timestamp(second_cut).date()),
        },
        'train_games': len(train),
        'validation_games': len(valid),
        'held_out_test_games': len(test),
        'best_trained_model': best_name,
        'ranking': results,
    }
    (MODELS / 'experiment_results.json').write_text(json.dumps(report, indent=2))
    pd.DataFrame(results).to_csv(MODELS / 'experiment_results.csv', index=False)
    joblib.dump({
        'model': best_model,
        'inputs': INPUTS,
        'name': best_name,
        'last_historical_date': str(train['GAME_DATE'].max().date()),
        'validation_scores': next(row for row in results if row['model'] == best_name),
        'note': 'Candidate trained on training split only; not calibrated or final-tested.',
    }, MODELS / 'nba_best_candidate.joblib')
    print('\nSaved models/experiment_results.csv, models/experiment_results.json,')
    print('and models/nba_best_candidate.joblib')


if __name__ == '__main__':
    main()
