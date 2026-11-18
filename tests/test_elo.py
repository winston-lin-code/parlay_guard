import pandas as pd
import pytest
from src.elo import add_elo_features, INITIAL_ELO, K_FACTOR, HOME_ADVANTAGE


def test_elo_is_pregame_and_same_day_is_batched():
    games = pd.DataFrame([
        {'GAME_ID':'1','GAME_DATE':'2024-01-01','SEASON':'2023-24','HOME_TEAM':'A','AWAY_TEAM':'B','HOME_WIN':1},
        {'GAME_ID':'2','GAME_DATE':'2024-01-01','SEASON':'2023-24','HOME_TEAM':'A','AWAY_TEAM':'C','HOME_WIN':0},
        {'GAME_ID':'3','GAME_DATE':'2024-01-02','SEASON':'2023-24','HOME_TEAM':'A','AWAY_TEAM':'B','HOME_WIN':1},
    ])
    result = add_elo_features(games).set_index('GAME_ID')
    assert result.loc['1','HOME_ELO'] == INITIAL_ELO
    assert result.loc['2','HOME_ELO'] == INITIAL_ELO
    expected = 1/(1+10**(-HOME_ADVANTAGE/400))
    assert result.loc['3','HOME_ELO'] == pytest.approx(INITIAL_ELO + K_FACTOR*(1-expected) - K_FACTOR*expected)
    assert result.loc['3','AWAY_ELO'] == pytest.approx(INITIAL_ELO-K_FACTOR*(1-expected))


def test_offseason_regression():
    games = pd.DataFrame([
        {'GAME_ID':'1','GAME_DATE':'2024-04-01','SEASON':'2023-24','HOME_TEAM':'A','AWAY_TEAM':'B','HOME_WIN':1},
        {'GAME_ID':'2','GAME_DATE':'2024-10-01','SEASON':'2024-25','HOME_TEAM':'A','AWAY_TEAM':'B','HOME_WIN':1},
    ])
    result = add_elo_features(games).set_index('GAME_ID')
    expected = 1/(1+10**(-HOME_ADVANTAGE/400))
    assert result.loc['2','HOME_ELO'] == pytest.approx(1500 + 0.75*20*(1-expected))
