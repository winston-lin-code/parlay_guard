"""Retrospective, strictly pre-date Parlay Guard backtest.

Run: python -u -m src.backtest_date --date 2026-11-17
Uses frozen model and completed games strictly before the target date.
Predictions are retrospectively reconstructed, NOT pre-tipoff timestamped forecasts.
"""
import argparse
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss, brier_score_loss, roc_auc_score

from .common import DATA, MODELS
from .predict_today_backup import (as_date, load_history,
                            fetch_current_season, clean_team_history, build_forecasts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--date', default='2026-11-17')
    parser.add_argument('--season-csv', type=Path, default=None)
    args = parser.parse_args()
    target = as_date(args.date)
    season = f'{target.year}-{(target.year+1)%100:02d}' if target.month >= 7 else f'{target.year-1}-{target.year%100:02d}'
    print(f'===== PARLAY GUARD RETROSPECTIVE BACKTEST: {target} =====', flush=True)
    bundle = joblib.load(MODELS / 'elo_final_holdout_model.joblib')
    print('Loaded frozen model (no training)', flush=True)
    # Reconstruct target-day matchups from completed NBA team-game logs.
    # Target-day outcomes are used only for schedule/results, NEVER as model features.
    current = fetch_current_season(season, args.season_csv)
    current['GAME_ID'] = current.GAME_ID.astype(str).str.zfill(10)
    current['GAME_DATE'] = pd.to_datetime(current.GAME_DATE).dt.normalize()
    current['IS_HOME'] = current.IS_HOME.astype(str).str.lower().isin(['true', '1'])
    day = current.loc[current.GAME_DATE.eq(pd.Timestamp(target))].copy()
    if day.empty:
        raise ValueError(f'No completed regular-season games found for {target} in the season log')
    checks = day.groupby('GAME_ID').agg(teams=('TEAM_ID', 'size'), homes=('IS_HOME', 'sum'),
                                         wins=('WL', lambda s: (s == 'W').sum()))
    invalid = checks.loc[(checks.teams != 2) | (checks.homes != 1) | (checks.wins != 1)]
    if not invalid.empty:
        raise ValueError(f'Incomplete target-date game results: {list(invalid.index)}')
    home = day.loc[day.IS_HOME, ['GAME_ID', 'TEAM_ABBREVIATION']].rename(
        columns={'TEAM_ABBREVIATION': 'HOME_TEAM'})
    away = day.loc[~day.IS_HOME, ['GAME_ID', 'TEAM_ABBREVIATION']].rename(
        columns={'TEAM_ABBREVIATION': 'AWAY_TEAM'})
    schedule = home.merge(away, on='GAME_ID', validate='one_to_one')
    schedule['GAME_DATE'] = pd.Timestamp(target)
    schedule['TIPOFF_ET'] = 'historical (time unavailable)'
    schedule['TIPOFF_UTC'] = ''
    schedule = schedule[['GAME_ID', 'GAME_DATE', 'TIPOFF_ET', 'TIPOFF_UTC', 'HOME_TEAM', 'AWAY_TEAM']]
    print(f'Completed matchups found: {len(schedule)}', flush=True)
    old = load_history(DATA / 'nba_games_updated.csv')
    history = clean_team_history(old, current, target)
    if pd.to_datetime(history.GAME_DATE).max().date() >= target:
        raise AssertionError('Leakage: history contains target date')
    print(f'History cutoff: {pd.to_datetime(history.GAME_DATE).max().date()} (strictly before target)', flush=True)
    predictions = build_forecasts(schedule, history, bundle, season, target)
    current = current.copy()
    current['GAME_ID'] = current.GAME_ID.astype(str).str.zfill(10)
    current['GAME_DATE'] = pd.to_datetime(current.GAME_DATE).dt.normalize()
    target_results = current.loc[(current.GAME_DATE == pd.Timestamp(target)) &
                                 current.GAME_ID.isin(predictions.GAME_ID)].copy()
    target_results['IS_HOME'] = target_results.IS_HOME.astype(str).str.lower().isin(['true','1'])
    counts = target_results.groupby('GAME_ID').agg(teams=('TEAM_ID','size'), homes=('IS_HOME','sum'), wins=('WL',lambda s: (s=='W').sum()))
    complete = set(counts.loc[(counts.teams==2)&(counts.homes==1)&(counts.wins==1)].index)
    missing = set(predictions.GAME_ID) - complete
    if missing:
        raise ValueError(f'{len(missing)} game(s) missing complete results: {sorted(missing)}')
    actual = target_results.loc[target_results.IS_HOME, ['GAME_ID','TEAM_ABBREVIATION','WL']].copy()
    actual['HOME_WIN'] = actual.WL.eq('W').astype(int)
    actual = actual.rename(columns={'TEAM_ABBREVIATION':'ACTUAL_HOME_TEAM'})[['GAME_ID','ACTUAL_HOME_TEAM','HOME_WIN']]
    out = predictions.merge(actual, on='GAME_ID', validate='one_to_one')
    if not out.HOME_TEAM.eq(out.ACTUAL_HOME_TEAM).all():
        raise ValueError('Schedule and result home teams disagree')
    out['ACTUAL_WINNER'] = np.where(out.HOME_WIN.eq(1),out.HOME_TEAM,out.AWAY_TEAM)
    y = out.HOME_WIN.to_numpy()
    print('\n===== BACKTEST METRICS =====',flush=True)
    for name,col in [('gradient_boosting','P_HOME_WIN_GB'),('elo_formula','P_HOME_WIN_ELO')]:
        p=out[col].to_numpy(dtype=float)
        if not np.isfinite(p).all() or ((p<0)|(p>1)).any():
            raise ValueError(f'Invalid {col}')
        out[name.upper()+'_CORRECT']=(p>=0.5)==y
        auc=roc_auc_score(y,p) if len(set(y))==2 else float('nan')
        print(f'{name:18} accuracy={accuracy_score(y,p>=0.5):.2%} '
              f'log_loss={log_loss(y,np.clip(p,1e-7,1-1e-7),labels=[0,1]):.5f} '
              f'brier={brier_score_loss(y,p):.5f} auc={auc:.5f}',flush=True)
    print('\n===== GAME-LEVEL PICKS =====',flush=True)
    print(out[['AWAY_TEAM','HOME_TEAM','P_HOME_WIN_GB','P_HOME_WIN_ELO','GB_PICK','ELO_PICK','ACTUAL_WINNER']].to_string(index=False,
          formatters={'P_HOME_WIN_GB':lambda x:f'{x:.1%}','P_HOME_WIN_ELO':lambda x:f'{x:.1%}'}),flush=True)
    output=MODELS / f'backtest_{target}.csv'
    out.to_csv(output,index=False)
    print(f'\nSaved {output}',flush=True)
    print('RETROSPECTIVE reconstruction, NOT a prediction recorded before tipoff.',flush=True)

if __name__=='__main__':
    main()
