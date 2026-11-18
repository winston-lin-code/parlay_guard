"""Elo-adjusted retrospective association of scorer underperformance and wins.
Run: python -u -m src.dependency_impact
Requires: pip install statsmodels
"""
import numpy as np
import pandas as pd
from .common import DATA, MODELS


def main():
    import statsmodels.api as sm
    from scipy.special import expit

    dep = pd.read_csv(DATA / 'nba_scoring_dependency.csv', dtype={'GAME_ID': str})
    teams = pd.read_csv(DATA / 'nba_games_updated.csv', dtype={'GAME_ID': str})
    games = pd.read_csv(DATA / 'nba_model_dataset.csv', dtype={'GAME_ID': str})
    if not {'GAME_ID', 'TEAM_ID', 'IS_HOME'}.issubset(teams):
        raise ValueError('nba_games_updated.csv must contain GAME_ID, TEAM_ID, IS_HOME')
    if not {'GAME_ID', 'ELO_DIFF'}.issubset(games):
        raise ValueError('Run python -m src.merge_features to add ELO_DIFF')
    side = teams[['GAME_ID', 'TEAM_ID', 'IS_HOME']].drop_duplicates()
    if side.duplicated(['GAME_ID', 'TEAM_ID']).any():
        raise ValueError('Duplicate team-game mappings')
    df = dep.merge(side, on=['GAME_ID', 'TEAM_ID'], validate='one_to_one')
    df = df.merge(games[['GAME_ID', 'ELO_DIFF']], on='GAME_ID', validate='many_to_one')
    df = df.loc[df['STAR_PLAYED'].astype(str).str.lower().eq('true')].copy()
    df = df.dropna(subset=['STAR_PERFORMANCE_RATIO', 'TOP_SCORER_SHARE_L10', 'ELO_DIFF', 'IS_WIN'])
    median = df['TOP_SCORER_SHARE_L10'].median()
    df['high_dependency'] = (df['TOP_SCORER_SHARE_L10'] >= median).astype(int)
    df['underperformed'] = (df['STAR_PERFORMANCE_RATIO'] < 0.80).astype(int)
    df['interaction'] = df['high_dependency'] * df['underperformed']
    df['team_elo_diff_100'] = np.where(df['IS_HOME'].astype(bool), df['ELO_DIFF'], -df['ELO_DIFF']) / 100.0
    df['home'] = df['IS_HOME'].astype(int)
    predictors = ['high_dependency', 'underperformed', 'interaction', 'team_elo_diff_100', 'home']
    X = sm.add_constant(df[predictors], has_constant='add')
    y = df['IS_WIN'].astype(int)
    # Both teams appear in each game: cluster uncertainty by GAME_ID.
    fit = sm.GLM(y, X, family=sm.families.Binomial()).fit(
        cov_type='cluster', cov_kwds={'groups': df['GAME_ID']}
    )
    estimates = pd.DataFrame({
        'term': fit.params.index,
        'coefficient': fit.params.values,
        'odds_ratio': np.exp(fit.params.values),
        'p_value': fit.pvalues.values,
        'ci_low_odds_ratio': np.exp(fit.conf_int().iloc[:, 0].values),
        'ci_high_odds_ratio': np.exp(fit.conf_int().iloc[:, 1].values),
    })
    MODELS.mkdir(parents=True, exist_ok=True)
    estimates.to_csv(MODELS / 'dependency_elo_adjusted_coefficients.csv', index=False)
    print(f'Analyzed {len(df):,} team-games across {df.GAME_ID.nunique():,} games')
    print(f'High dependency threshold: {median:.1%}')
    print('\n===== ELO-ADJUSTED LOGISTIC REGRESSION =====')
    print(estimates.to_string(index=False, float_format=lambda v: f'{v:.4f}'))
    # Illustrative model-based probabilities for an average-strength home team.
    print('\n===== ADJUSTED WIN PROBABILITIES (ELO DIFF = 0, HOME) =====')
    for high in (0, 1):
        for under in (0, 1):
            row = pd.DataFrame([{'const': 1, 'high_dependency': high,
                                 'underperformed': under, 'interaction': high * under,
                                 'team_elo_diff_100': 0, 'home': 1}])[X.columns]
            print(f'High dependency={high}, underperformed={under}: {float(fit.predict(row).iloc[0]):.1%}')
    print('\nInteraction coefficient estimates whether the underperformance association differs by dependency,')
    print('after adjusting for pregame Elo and home court. This is NOT causal or pregame prediction.')
    print('Note: one game may contribute two team-game observations; clustered SE accounts for this.')


if __name__ == '__main__':
    main()
