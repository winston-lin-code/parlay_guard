"""Chronological, out-of-time evaluation with probability calibration."""
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from .common import DATA, MODELS
from .merge_features import FEATURES

NUMERIC = [f'{side}_{name}' for side in ('HOME','AWAY') for name in FEATURES]
CATEGORICAL = ['HOME_TEAM','AWAY_TEAM']
INPUTS = NUMERIC + CATEGORICAL


def make_model():
    preprocess = ColumnTransformer([
        ('num', Pipeline([
            ('impute', SimpleImputer(strategy='median', add_indicator=True)),
            ('scale', StandardScaler())
        ]), NUMERIC),
        ('cat', OneHotEncoder(handle_unknown='ignore'), CATEGORICAL)
    ])
    return Pipeline([
        ('prep', preprocess),
        ('classifier', LogisticRegression(max_iter=2000))
    ])


def metrics(y, p):
    result = {
        'n_games': int(len(y)),
        'accuracy': float(accuracy_score(y, p >= 0.5)),
        'brier': float(brier_score_loss(y, p)),
        'log_loss': float(log_loss(y, p, labels=[0,1])),
    }
    if len(np.unique(y)) == 2:
        result['roc_auc'] = float(roc_auc_score(y, p))
    return result


def main():
    df = pd.read_csv(DATA / 'nba_model_dataset.csv', dtype={'GAME_ID': str})
    df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
    df = df.sort_values(['GAME_DATE','GAME_ID']).reset_index(drop=True)
    # Split on unique dates, never splitting games played on the same date.
    dates = np.array(sorted(df['GAME_DATE'].unique()))
    if len(dates) < 30:
        raise ValueError('Insufficient historical dates for time-based evaluation')
    first_cut, second_cut = dates[int(len(dates)*0.70)], dates[int(len(dates)*0.85)]
    train = df[df['GAME_DATE'] < first_cut]
    valid = df[(df['GAME_DATE'] >= first_cut) & (df['GAME_DATE'] < second_cut)]
    test = df[df['GAME_DATE'] >= second_cut]
    if min(len(train), len(valid), len(test)) == 0:
        raise ValueError('Empty chronological split')
    model = make_model()
    model.fit(train[INPUTS], train['HOME_WIN'].astype(int))
    valid_probs = model.predict_proba(valid[INPUTS])[:,1]
    # Calibration using validation data only, with Platt scaling.
    # If the validation labels lack both classes, skip calibration.
    calibrated = len(np.unique(valid['HOME_WIN'])) == 2
    if calibrated:
        calibrator = LogisticRegression(max_iter=1000)
        calibrator.fit(valid_probs.reshape(-1,1), valid['HOME_WIN'].astype(int))
    else:
        calibrator = None
    raw_test_probs = model.predict_proba(test[INPUTS])[:,1]
    test_probs = (calibrator.predict_proba(raw_test_probs.reshape(-1,1))[:,1]
                  if calibrator is not None else raw_test_probs)
    report = {
        'split_dates': {'train_before': str(pd.Timestamp(first_cut).date()),
                        'test_from': str(pd.Timestamp(second_cut).date())},
        'train_games': int(len(train)), 'validation': metrics(valid['HOME_WIN'].astype(int), valid_probs),
        'test': metrics(test['HOME_WIN'].astype(int), test_probs),
        'calibration': 'Platt on validation predictions' if calibrated else 'none'
    }
    artifact = {'model':model, 'calibrator':calibrator, 'inputs':INPUTS,
                'last_historical_date':str(df['GAME_DATE'].max().date()), 'report':report}
    joblib.dump(artifact, MODELS / 'nba_home_win.joblib')
    (MODELS / 'evaluation.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print('Saved models/nba_home_win.joblib and models/evaluation.json')
    print('Note: training uses historical results only; no live injury or lineup inputs.')

if __name__ == '__main__':
    main()
