import pandas as pd
from src.feature_engineering import build_features


def test_rolling_uses_only_prior_games():
    dates = pd.date_range('2024-01-01', periods=11)
    rows = [{'TEAM_ID':1,'GAME_ID':str(i),'GAME_DATE':dates[i],
             'WL':'W' if i < 5 else 'L'} for i in range(11)]
    result = build_features(pd.DataFrame(rows))
    assert pd.isna(result.loc[0,'WIN_PCT_LAST_5'])
    assert result.loc[5,'WIN_PCT_LAST_5'] == 1.0
    assert result.loc[6,'WIN_PCT_LAST_5'] == 0.8
    assert result.loc[10,'WIN_PCT_LAST_10'] == 0.5


def test_team_groups_are_independent():
    rows = []
    for team, wl in [(1,'W'),(2,'L')]:
        for i in range(6):
            rows.append({'TEAM_ID':team,'GAME_ID':f'{team}-{i}',
                         'GAME_DATE':f'2024-01-{i+1:02d}', 'WL':wl})
    result = build_features(pd.DataFrame(rows))
    assert result[result.TEAM_ID.eq(1)].iloc[-1]['WIN_PCT_LAST_5'] == 1.0
    assert result[result.TEAM_ID.eq(2)].iloc[-1]['WIN_PCT_LAST_5'] == 0.0
