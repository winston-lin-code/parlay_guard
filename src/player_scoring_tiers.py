"""Evaluate player underperformance models separately by pregame scoring tier.

Run from the project root:
    python -u -m src.player_scoring_tiers

Uses the same chronological date boundaries as train_player_underperformance.py.
The 2025-12-29+ final holdout is never evaluated.
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
from .train_player_underperformance import FEATURES, HOLDOUT_START, TARGET, SEED

# Disjoint tiers, plus cumulative cutoffs for the star-player use case.
TIERS = [
    ('all_players', 0, None),
    ('under_10_ppg', 0, 10),
    ('10_to_15_ppg', 10, 15),
    ('15_to_20_ppg', 15, 20),
    ('20_to_25_ppg', 20, 25),
    ('25_plus_ppg', 25, None),
    ('10_plus_ppg', 10, None),
    ('15_plus_ppg', 15, None),
    ('20_plus_ppg', 20, None),
]


def build_models():
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


def evaluate(y_true, p):
    p = np.clip(np.asarray(p, dtype=float), 1e-7, 1 - 1e-7)
    return {
        'log_loss': log_loss(y_true, p, labels=[0, 1]),
        'brier': brier_score_loss(y_true, p),
        'accuracy': accuracy_score(y_true, p >= 0.5),
        'roc_auc': roc_auc_score(y_true, p) if y_true.nunique() == 2 else np.nan,
    }


def main():
    path = DATA / 'nba_player_underperformance_dataset.csv'
    if not path.exists():
        raise FileNotFoundError(f'Missing {path}; run python -u -m src.player_underperformance_features')
    df = pd.read_csv(path, dtype={'GAME_ID': str})
    required = set(FEATURES) | {TARGET, 'GAME_DATE', 'GAME_ID', 'PLAYER_ID', 'PLAYER_PPG_LAST_10'}
    if required - set(df.columns):
        raise ValueError(f'Missing columns: {sorted(required - set(df.columns))}')
    df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'], errors='raise').dt.normalize()
    df[TARGET] = pd.to_numeric(df[TARGET], errors='raise').astype(int)
    if not df[TARGET].isin([0, 1]).all():
        raise ValueError('Target must be 0 or 1')
    if df.duplicated(['GAME_ID', 'PLAYER_ID']).any():
        raise ValueError('Duplicate player-game rows')
    for col in FEATURES:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df['PLAYER_PPG_LAST_10'] = pd.to_numeric(df['PLAYER_PPG_LAST_10'], errors='coerce')
    dev = df.loc[df.GAME_DATE < HOLDOUT_START].copy()
    dates = np.array(sorted(dev.GAME_DATE.unique()))
    if len(dates) < 30:
        raise ValueError('Not enough development dates')
    bounds = [int(len(dates) * f) for f in (0.4, 0.6, 0.8, 1.0)]
    if len(set(bounds)) != 4:
        raise ValueError('Fold boundaries overlap')

    print(f'Development rows: {len(dev):,}; holdout rows excluded: {len(df) - len(dev):,}')
    print('Scoring tiers are selected using PLAYER_PPG_LAST_10 (pregame only).')
    print('Training is restricted to each tier; baselines are refitted within each fold and tier.')
    print('Final holdout begins 2025-12-29 and is NOT evaluated.\n')
    records = []
    for tier, lower, upper in TIERS:
        mask = dev.PLAYER_PPG_LAST_10 >= lower
        if upper is not None:
            mask &= dev.PLAYER_PPG_LAST_10 < upper
        subset = dev.loc[mask]
        print(f'===== {tier} | {len(subset):,} rows | underperformance {subset[TARGET].mean():.1%} =====', flush=True)
        for fold in range(3):
            start = pd.Timestamp(dates[bounds[fold]])
            end = pd.Timestamp(dates[bounds[fold + 1]]) if fold < 2 else HOLDOUT_START
            train = subset.loc[subset.GAME_DATE < start]
            valid = subset.loc[(subset.GAME_DATE >= start) & (subset.GAME_DATE < end)]
            if len(train) < 100 or len(valid) < 50 or train[TARGET].nunique() != 2 or valid[TARGET].nunique() != 2:
                print(f'  Fold {fold + 1}: insufficient observations; skipped')
                continue
            if set(train.GAME_ID).intersection(valid.GAME_ID):
                raise AssertionError('Game leakage across folds')
            for model_name, model in build_models().items():
                model.fit(train[FEATURES], train[TARGET])
                p = model.predict_proba(valid[FEATURES])[:, 1]
                stats = evaluate(valid[TARGET], p)
                records.append({
                    'tier': tier, 'fold': fold + 1, 'model': model_name,
                    'train_rows': len(train), 'validation_rows': len(valid),
                    'train_underperformance_rate': train[TARGET].mean(),
                    'validation_underperformance_rate': valid[TARGET].mean(),
                    'validation_start': str(start.date()),
                    'validation_end_exclusive': str(end.date()),
                    **stats,
                })
            fold_rows = [r for r in records if r['tier'] == tier and r['fold'] == fold + 1]
            print(f'  Fold {fold + 1}: train={len(train):,}, validate={len(valid):,} | ' +
                  ' | '.join(f'{r["model"]} LL={r["log_loss"]:.4f} acc={r["accuracy"]:.1%}' for r in fold_rows), flush=True)
        print()

    if not records:
        raise ValueError('No folds could be evaluated')
    details = pd.DataFrame(records)
    summary = details.groupby(['tier', 'model'], as_index=False).agg(
        folds=('fold', 'nunique'),
        total_validation_rows=('validation_rows', 'sum'),
        mean_log_loss=('log_loss', 'mean'),
        mean_brier=('brier', 'mean'),
        mean_accuracy=('accuracy', 'mean'),
        mean_roc_auc=('roc_auc', 'mean'),
    )
    order = {name: i for i, (name, _, _) in enumerate(TIERS)}
    summary['tier_order'] = summary.tier.map(order)
    summary = summary.sort_values(['tier_order', 'mean_log_loss']).drop(columns='tier_order')
    print('===== SCORING-TIER WALK-FORWARD SUMMARY =====')
    print(summary.to_string(index=False, float_format=lambda x: f'{x:.5f}'))
    baseline = summary.loc[summary.model == 'majority_baseline', ['tier', 'mean_log_loss']].rename(columns={'mean_log_loss': 'baseline_log_loss'})
    improvements = summary.loc[summary.model != 'majority_baseline'].merge(baseline, on='tier')
    improvements['log_loss_improvement'] = improvements.baseline_log_loss - improvements.mean_log_loss
    print('\n===== LOG LOSS IMPROVEMENT OVER SAME-TIER BASELINE (POSITIVE IS BETTER) =====')
    print(improvements[['tier', 'model', 'folds', 'log_loss_improvement']].to_string(index=False, float_format=lambda x: f'{x:.5f}'))
    print('\nCAUTION: Tiers overlap for cumulative groups (10+/15+/20+). Do not add their counts.')
    print('These results are exploratory; final held-out games remain untouched.')
    MODELS.mkdir(parents=True, exist_ok=True)
    details.to_csv(MODELS / 'player_scoring_tier_folds.csv', index=False)
    summary.to_csv(MODELS / 'player_scoring_tier_summary.csv', index=False)
    improvements.to_csv(MODELS / 'player_scoring_tier_improvements.csv', index=False)
    print('Saved models/player_scoring_tier_folds.csv, player_scoring_tier_summary.csv, and player_scoring_tier_improvements.csv')


if __name__ == '__main__':
    main()
