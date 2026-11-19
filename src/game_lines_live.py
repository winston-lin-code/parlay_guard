"""Pre-game NBA projected margin and total, learned from completed games only.

Train chronological Ridge models on rolling team scoring/allowing form. Holdout
metrics use 2025-26; the production model refits on all completed games. This is
not a calibrated betting-edge model, and it does not account for injuries.
"""
from __future__ import annotations
from collections import defaultdict, deque
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error

FEATURES = [
    'home_for_5','home_for_10','home_against_5','home_against_10',
    'away_for_5','away_for_10','away_against_5','away_against_10',
    'home_for_std_10','away_for_std_10','home_rest','away_rest',
]


def _features(home, away, day, history):
    record = {}
    for prefix, team in (('home',home), ('away',away)):
        rows = list(history.get(team, ()))
        if len(rows) < 10:
            return None
        recent = rows[-10:]
        for metric, index in (('for',0),('against',1)):
            vals = [float(r[index]) for r in recent]
            record[f'{prefix}_{metric}_5'] = float(np.mean(vals[-5:]))
            record[f'{prefix}_{metric}_10'] = float(np.mean(vals))
            if metric == 'for':
                record[f'{prefix}_for_std_10'] = float(np.std(vals, ddof=1))
        last_date = pd.Timestamp(recent[-1][2]).normalize()
        record[f'{prefix}_rest'] = float(min(8, max(0, (pd.Timestamp(day).normalize() - last_date).days-1)))
    return record


def feature_training(completed: pd.DataFrame):
    needed = {'GAME_DATE','HOME_TEAM','AWAY_TEAM','HOME_PTS','AWAY_PTS'}
    missing = needed - set(completed.columns)
    if missing: raise ValueError(f'Missing completed-game columns: {sorted(missing)}')
    games = completed.copy()
    games['GAME_DATE'] = pd.to_datetime(games['GAME_DATE']).dt.normalize()
    games = games.sort_values(['GAME_DATE','GAME_ID'] if 'GAME_ID' in games else ['GAME_DATE'])
    hist = defaultdict(lambda: deque(maxlen=10))
    records=[]
    for day, group in games.groupby('GAME_DATE', sort=True):
        daily_updates=[]
        for g in group.itertuples(index=False):
            home,away = str(g.HOME_TEAM).upper(),str(g.AWAY_TEAM).upper()
            feat=_features(home,away,day,hist)
            if feat is not None:
                feat.update(date=day, home_margin=float(g.HOME_PTS)-float(g.AWAY_PTS),
                            total=float(g.HOME_PTS)+float(g.AWAY_PTS))
                records.append(feat)
            daily_updates.append((home,away,float(g.HOME_PTS),float(g.AWAY_PTS)))
        # Same-date games do not see each other's results.
        for home,away,hp,ap in daily_updates:
            hist[home].append((hp,ap,day))
            hist[away].append((ap,hp,day))
    return pd.DataFrame(records),hist


def fit_line_models(completed: pd.DataFrame):
    rows,hist=feature_training(completed)
    if len(rows)<400:
        raise ValueError(f'Not enough completed games with 10-game history to train lines ({len(rows)})')
    metrics={}
    holdout=rows[(rows.date>=pd.Timestamp('2025-10-01'))&(rows.date<pd.Timestamp('2026-07-01'))]
    train=rows[rows.date<pd.Timestamp('2025-10-01')]
    if len(train)>=300 and len(holdout)>=50:
        for target in ('home_margin','total'):
            model=make_pipeline(StandardScaler(),Ridge(alpha=150.0))
            model.fit(train[FEATURES],train[target]); predicted=model.predict(holdout[FEATURES])
            metrics[target+'_holdout_mae']=round(float(mean_absolute_error(holdout[target],predicted)),2)
        metrics['holdout_games']=len(holdout)
    models={}
    for target in ('home_margin','total'):
        model=make_pipeline(StandardScaler(),Ridge(alpha=150.0))
        model.fit(rows[FEATURES],rows[target]);models[target]=model
    metrics['trained_games']=len(rows)
    metrics['latest_completed_game']=str(pd.Timestamp(completed['GAME_DATE'].max()).date())
    return models,hist,metrics


def predict_line(models,hist,*,home:str,away:str,game_day):
    feat=_features(home.upper(),away.upper(),pd.Timestamp(game_day),hist)
    if feat is None: return None
    x=pd.DataFrame([feat],columns=FEATURES)
    margin=float(models['home_margin'].predict(x)[0]);total=float(models['total'].predict(x)[0])
    return {'home_margin':round(margin,1), 'projected_total':round(total,1),
            'home_points':round((total+margin)/2,1),
            'away_points':round((total-margin)/2,1),
            'home_spread':round(-margin,1),
            'method':'out-of-sample-audited ridge (team rolling scoring)',
            'validated_for_market_ev':False}
