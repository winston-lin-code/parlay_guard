"""Compare pregame scorer dependency features against the existing Elo model.
Run: python -u -m src.dependency_walk_forward
Does not evaluate the final 15% held-out test period.
"""
import numpy as np
import pandas as pd
from .common import DATA, MODELS
from .walk_forward import make_model, score

FEATURES = ['TOP_SCORER_SHARE_L10', 'TOP_SCORER_PPG_L10', 'TOP_SCORER_PTS_STD_L10']


def prepare():
    df = pd.read_csv(DATA / 'nba_model_dataset.csv', dtype={'GAME_ID': str})
    dep = pd.read_csv(DATA / 'nba_scoring_dependency.csv', dtype={'GAME_ID': str})
    teams = pd.read_csv(DATA / 'nba_games_updated.csv', dtype={'GAME_ID': str})
    mapping = teams[['GAME_ID', 'TEAM_ID', 'IS_HOME']].drop_duplicates()
    if mapping.duplicated(['GAME_ID', 'TEAM_ID']).any():
        raise ValueError('Duplicate team-game mappings')
    dep = dep.merge(mapping, on=['GAME_ID', 'TEAM_ID'], validate='one_to_one')
    for side, home in [('HOME', True), ('AWAY', False)]:
        selected = dep.loc[dep['IS_HOME'].astype(bool) == home, ['GAME_ID'] + FEATURES].copy()
        selected = selected.rename(columns={c: f'{side}_{c}' for c in FEATURES})
        df = df.merge(selected, on='GAME_ID', how='left', validate='one_to_one')
    df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
    return df.sort_values(['GAME_DATE', 'GAME_ID']).reset_index(drop=True)


def main():
    df = prepare()
    dates = np.array(sorted(df['GAME_DATE'].unique()))
    boundaries = [int(len(dates) * f) for f in (0.40, 0.55, 0.70, 0.85)]
    test_start = pd.Timestamp(dates[boundaries[-1]])
    print(f'Final held-out test starts {test_start.date()} (NOT evaluated)')
    added = [f'{side}_{f}' for side in ('HOME', 'AWAY') for f in FEATURES]
    print('Missing dependency features:', df[added].isna().sum().to_dict())
    rows = []
    for fold in range(3):
        start, end = (pd.Timestamp(dates[boundaries[i]]) for i in (fold, fold + 1))
        train = df[df['GAME_DATE'] < start]
        valid = df[(df['GAME_DATE'] >= start) & (df['GAME_DATE'] < end)]
        for name, with_dependency in [('elo_baseline', False), ('elo_plus_scoring_dependency', True)]:
            model, inputs = make_model(include_team_identity=False, include_point_diff=True,
                                       elo_mode='difference')
            if with_dependency:
                # Extend the numeric transformer in the EXISTING sklearn pipeline.
                numeric_pipeline = model.named_steps['preprocessor'].transformers[0][1]
                numeric = list(model.named_steps['preprocessor'].transformers[0][2]) + added
                model.named_steps['preprocessor'].transformers = [
                    ('numeric', numeric_pipeline, numeric)
                ]
                inputs += added
            if set(inputs) - set(df.columns):
                raise ValueError(f'Missing inputs: {set(inputs) - set(df.columns)}')
            model.fit(train[inputs], train['HOME_WIN'].astype(int))
            probabilities = model.predict_proba(valid[inputs])[:, 1]
            metrics = score(valid['HOME_WIN'].astype(int), probabilities)
            rows.append({'fold': fold + 1, 'model': name, 'games': len(valid), **metrics})
            print(f'Fold {fold + 1} {name}: log_loss={metrics["log_loss"]:.5f} '
                  f'brier={metrics["brier"]:.5f} accuracy={metrics["accuracy"]:.2%}', flush=True)
    details = pd.DataFrame(rows)
    summary = details.groupby('model', as_index=False).agg(
        mean_log_loss=('log_loss', 'mean'), mean_brier=('brier', 'mean'),
        mean_accuracy=('accuracy', 'mean'), mean_roc_auc=('roc_auc', 'mean'))
    print('\n===== SCORING DEPENDENCY WALK-FORWARD SUMMARY =====')
    print(summary.to_string(index=False, float_format=lambda x: f'{x:.5f}'))
    pivot = details.pivot(index='fold', columns='model', values='log_loss')
    print('\n===== PER-FOLD LOG LOSS =====')
    print(pivot.to_string(float_format=lambda x: f'{x:.5f}'))
    print('\nDependency minus baseline (negative is better):')
    print((pivot['elo_plus_scoring_dependency'] - pivot['elo_baseline']).to_string(float_format=lambda x: f'{x:+.5f}'))
    MODELS.mkdir(parents=True, exist_ok=True)
    details.to_csv(MODELS / 'dependency_walk_forward_folds.csv', index=False)
    summary.to_csv(MODELS / 'dependency_walk_forward_summary.csv', index=False)
    print('\nHeld-out test NOT evaluated.')


if __name__ == '__main__':
    main()
