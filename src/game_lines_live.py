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
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import brier_score_loss, log_loss, mean_absolute_error

from .elo import HOME_ADVANTAGE, add_elo_features

FEATURES = [
    'home_for_5','home_for_10','home_against_5','home_against_10',
    'away_for_5','away_for_10','away_against_5','away_against_10',
    'home_for_std_10','away_for_std_10','home_rest','away_rest',
]
ELO_FEATURE = 'ELO_DIFF'


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


def feature_training(completed: pd.DataFrame, *, include_elo=False, include_game_id=False):
    needed = {'GAME_DATE','HOME_TEAM','AWAY_TEAM','HOME_PTS','AWAY_PTS'}
    missing = needed - set(completed.columns)
    if missing: raise ValueError(f'Missing completed-game columns: {sorted(missing)}')
    games = completed.copy()
    games['GAME_DATE'] = pd.to_datetime(games['GAME_DATE']).dt.normalize()
    if include_elo:
        games = add_elo_features(games)
    games = games.sort_values(['GAME_DATE','GAME_ID'] if 'GAME_ID' in games else ['GAME_DATE'])
    hist = defaultdict(lambda: deque(maxlen=10))
    records=[]
    for day, group in games.groupby('GAME_DATE', sort=True):
        daily_updates=[]
        for g in group.itertuples(index=False):
            home,away = str(g.HOME_TEAM).upper(),str(g.AWAY_TEAM).upper()
            feat=_features(home,away,day,hist)
            if feat is not None:
                if include_elo:
                    feat[ELO_FEATURE] = getattr(g, ELO_FEATURE)
                if include_game_id:
                    feat['game_id'] = str(g.GAME_ID)
                feat.update(date=day, home_margin=float(g.HOME_PTS)-float(g.AWAY_PTS),
                            total=float(g.HOME_PTS)+float(g.AWAY_PTS))
                records.append(feat)
            daily_updates.append((home,away,float(g.HOME_PTS),float(g.AWAY_PTS)))
        # Same-date games do not see each other's results.
        for home,away,hp,ap in daily_updates:
            hist[home].append((hp,ap,day))
            hist[away].append((ap,hp,day))
    return pd.DataFrame(records),hist


def _fit_regression(rows, target, features):
    model = make_pipeline(StandardScaler(), Ridge(alpha=150.0))
    return model.fit(rows[features], rows[target])


def _fit_margin_probability(train, features):
    # The probability model sees only walk-forward margin predictions.
    predictions = []
    outcomes = []
    for _, month in train.groupby(train.date.dt.to_period('M')):
        earlier = train[train.date < month.date.min()]
        if len(earlier) < 300:
            continue
        model = _fit_regression(earlier, 'home_margin', features)
        predictions.extend(np.round(model.predict(month[features]), 1))
        outcomes.extend(month.home_margin > 0)
    if len(predictions) < 50 or len(set(outcomes)) < 2:
        return None
    model = LogisticRegression(fit_intercept=False)
    model.fit(np.asarray(predictions).reshape(-1, 1), outcomes)
    # A positive slope and zero intercept give 50% at a pick'em.
    return model if model.coef_[0, 0] > 0 else None


def fit_line_models(completed: pd.DataFrame, *, include_elo=False):
    rows,hist=feature_training(completed, include_elo=include_elo)
    if len(rows)<400:
        raise ValueError(f'Not enough completed games with 10-game history to train lines ({len(rows)})')
    metrics={}
    holdout=rows[(rows.date>=pd.Timestamp('2025-10-01'))&(rows.date<pd.Timestamp('2026-07-01'))]
    train=rows[rows.date<pd.Timestamp('2025-10-01')]
    margin_features = FEATURES + [ELO_FEATURE] if include_elo else FEATURES
    target_features = {'home_margin': margin_features, 'total': FEATURES}
    probability_model = None
    models={}
    if len(train)>=300 and len(holdout)>=50:
        for target, features in target_features.items():
            model = _fit_regression(train, target, features)
            predicted = model.predict(holdout[features])
            metrics[target+'_holdout_mae']=round(float(mean_absolute_error(holdout[target],predicted)),2)
            if include_elo and target == 'home_margin':
                probability_model = _fit_margin_probability(train, margin_features)
                if probability_model is not None:
                    outcome = holdout.home_margin > 0
                    unified = probability_model.predict_proba(np.round(predicted, 1).reshape(-1, 1))[:, 1]
                    elo = 1 / (1 + 10 ** (-(holdout[ELO_FEATURE] + HOME_ADVANTAGE) / 400))
                    for label, probability in (('margin', unified), ('elo', elo)):
                        metrics[f'{label}_holdout_brier'] = float(brier_score_loss(outcome, probability))
                        metrics[f'{label}_holdout_log_loss'] = float(log_loss(outcome, probability, labels=[False, True]))
        metrics['holdout_games']=len(holdout)
    for target, features in target_features.items():
        models[target] = _fit_regression(rows, target, features)
    if include_elo:
        models['margin_probability'] = probability_model
    metrics['trained_games']=len(rows)
    metrics['latest_completed_game']=str(pd.Timestamp(completed['GAME_DATE'].max()).date())
    return models,hist,metrics


def predict_line(models,hist,*,home:str,away:str,game_day,elo_diff=None):
    feat=_features(home.upper(),away.upper(),pd.Timestamp(game_day),hist)
    if feat is None: return None
    uses_elo = 'margin_probability' in models
    if uses_elo:
        if elo_diff is None or not np.isfinite(elo_diff) or models['margin_probability'] is None:
            return None
        feat[ELO_FEATURE] = elo_diff
    margin_features = FEATURES + [ELO_FEATURE] if uses_elo else FEATURES
    x = pd.DataFrame([feat], columns=margin_features)
    margin = float(models['home_margin'].predict(x)[0])
    total = float(models['total'].predict(x[FEATURES])[0])
    if uses_elo:
        margin, total = round(margin, 1), round(total, 1)
        return {
            'home_margin': margin,
            'projected_total': total,
            'home_points': round((total + margin) / 2, 2),
            'away_points': round((total - margin) / 2, 2),
            'home_spread': -margin,
            'home_win_probability': float(models['margin_probability'].predict_proba([[margin]])[0, 1]),
            'method': 'Ridge score model with pregame Elo',
            'validated_for_market_ev': False,
        }
    return {'home_margin':round(margin,1), 'projected_total':round(total,1),
            'home_points':round((total+margin)/2,1),
            'away_points':round((total-margin)/2,1),
            'home_spread':round(-margin,1),
            'method':'out-of-sample-audited ridge (team rolling scoring)',
            'validated_for_market_ev':False}
