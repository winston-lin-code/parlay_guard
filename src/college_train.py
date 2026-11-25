"""Historical, independently trained college-moneyline models.

Source: SportsDataverse ESPN completed game schedules (parquet GitHub releases).
Every feature is computed before updating the ratings with that game's result.
No future games or in-progress games enter training. Independent CFB / MBB fits.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import io
import json
import math
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / 'data' / 'college_guard' / 'historical'
MODELS = ROOT / 'models' / 'college_guard'
DATASET_URLS = {
    'mbb': 'https://github.com/sportsdataverse/sportsdataverse-data/releases/download/espn_mens_college_basketball_schedules/mbb_schedule_{season}.parquet',
    'cfb': 'https://github.com/sportsdataverse/sportsdataverse-data/releases/download/cfb_schedules/cfb_schedules_{season}.parquet',
}
FEATURES = ['elo_difference', 'form_difference', 'site_home', 'rest_difference', 'experience_difference']


def _id(value):
    if value is None or pd.isna(value): return ''
    return str(int(value)) if isinstance(value, (int, float)) and math.isfinite(value) else str(value).strip()


def _true(value):
    return str(value).lower() in ('true', '1', 'yes') if not isinstance(value, bool) else value


def parse_schedule(df, sport, now=None, season_hint=None):
    """Normalize public historical schedules; exclude all non-final/ambiguous outcomes."""
    now = pd.Timestamp(now or datetime.now(timezone.utc)).tz_convert('UTC')
    out = []
    for row in df.to_dict('records'):
        if sport == 'mbb':
            state = str(row.get('status_type_name') or '')
            if not state.startswith('STATUS_FINAL') or not _true(row.get('status_type_completed')):
                continue
            if _true(row.get('home_non_div1_team')) or _true(row.get('away_non_div1_team')):continue
            date = row.get('game_date_time') or row.get('start_date') or row.get('date')
            home_score, away_score = row.get('home_score'), row.get('away_score')
            home, away = _id(row.get('home_id')), _id(row.get('away_id'))
            game_id = _id(row.get('id'))
            season = row.get('season') or season_hint
            neutral = _true(row.get('neutral_site'))
        else:
            if not _true(row.get('completed')):continue
            if str(row.get('home_division','')).lower() != 'fbs' or str(row.get('away_division','')).lower() != 'fbs':continue
            date = row.get('start_date')
            home_score, away_score = row.get('home_points'), row.get('away_points')
            home, away = _id(row.get('home_id')), _id(row.get('away_id'))
            game_id = _id(row.get('game_id'))
            season = row.get('season') or season_hint
            neutral = _true(row.get('neutral_site'))
        if not home or not away or home == away or not game_id or pd.isna(home_score) or pd.isna(away_score):continue
        try:
            hs, aws = int(home_score), int(away_score)
            ts = pd.Timestamp(date)
            if pd.isna(ts):continue
            ts = ts.tz_localize('UTC') if ts.tzinfo is None else ts.tz_convert('UTC')
            season = int(season if season is not None else (ts.year + (ts.month>=8 if sport=='mbb' else 0)))
        except (ValueError,TypeError,OverflowError):continue
        if ts >= now or hs == aws or hs < 0 or aws < 0:continue
        out.append({'game_id':game_id,'date':ts.isoformat(),'season':season,'home_id':home,
                    'away_id':away,'home_win':int(hs>aws),'neutral':neutral,
                    'home_points':hs,'away_points':aws})
    return sorted({x['game_id']:x for x in out}.values(),key=lambda x:(x['date'],x['game_id']))


def season_list(sport, now=None):
    t = now or datetime.now(timezone.utc)
    return list(range(t.year - (2 if sport=='mbb' else 3), t.year + (2 if sport=='mbb' else 1)))


def download_schedule(sport, season, *, session=None, refresh=False, cache=CACHE):
    """Cache finished seasons; recache current incomplete season on request."""
    cache = Path(cache);cache.mkdir(parents=True,exist_ok=True)
    path = cache / f'{sport}_{season}.parquet'
    if path.exists() and not refresh:return pd.read_parquet(path)
    url = DATASET_URLS[sport].format(season=season)
    sess=session or requests.Session()
    response=sess.get(url,timeout=(10,75),headers={'User-Agent':'ParlayGuardResearch/1.0'})
    response.raise_for_status()
    if len(response.content) < 128 or not response.content[:4] == b'PAR1':
        raise ValueError(f'Invalid parquet release for {sport} season {season}; no data saved')
    tmp=path.with_suffix('.parquet.tmp')
    tmp.write_bytes(response.content)
    try:
        frame=pd.read_parquet(tmp)
        if frame.empty:raise ValueError('Empty downloaded schedule')
        tmp.replace(path)
    finally:tmp.unlink(missing_ok=True)
    return frame


def initial_team():
    return {'elo':1500.0,'wins':[],'games':0,'last_date':None,'season':None}


def snapshot_features(states,home,away,when,season,neutral=False):
    # Inference must NOT mutate saved team states.
    def view(team):
        s=copy.deepcopy(states.get(team) or initial_team())
        if s['season'] is not None and s['season'] != season:
            s['elo'] = 1500 + .65*(s['elo']-1500)
            s['wins'] = []
            s['games']=0
            s['last_date']=None
        last = pd.Timestamp(s['last_date']) if s['last_date'] else None
        rest = min(14,max(0,(pd.Timestamp(when)-last).total_seconds()/86400)) if last else 7.0
        return s,rest
    h,hr=view(home); a,ar=view(away)
    form=lambda s: sum(s['wins'][-10:])/len(s['wins'][-10:]) if s['wins'] else .5
    feat=[(h['elo']-a['elo'])/400.0,form(h)-form(a),0.0 if neutral else 1.0,
          (hr-ar)/7.0,max(-2.0,min(2.0,(h['games']-a['games'])/10.0))]
    return feat,h,a


def training_rows(records, sport):
    states={};X=[];y=[];dates=[]
    k=26 if sport=='cfb' else 24
    for g in sorted(records,key=lambda x:(x['date'],x['game_id'])):
        home,away=g['home_id'],g['away_id'];date=g['date'];season=g['season']
        f,h,a=snapshot_features(states,home,away,date,season,g['neutral'])
        X.append(f);y.append(g['home_win']);dates.append(date)
        # Rating update occurs AFTER pregame feature capture; result never enters its own features.
        elo_prob=1.0/(1+10**(-(h['elo']-a['elo']+(0 if g['neutral'] else 70))/400.0))
        delta=k*(g['home_win']-elo_prob)
        h['elo']+=delta; a['elo']-=delta
        h['wins']=(h['wins']+[g['home_win']])[-10:]
        a['wins']=(a['wins']+[1-g['home_win']])[-10:]
        h['games']+=1;a['games']+=1
        h['last_date']=date;a['last_date']=date
        h['season']=season;a['season']=season
        states[home]=h; states[away]=a
    return X,y,states,dates


def train(sport, records, *, model_path=None, min_games=None):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import brier_score_loss, log_loss, accuracy_score
    from joblib import dump
    if sport not in DATASET_URLS:raise ValueError('Unknown sport')
    min_games = min_games if min_games is not None else (500 if sport=='mbb' else 240)
    records = sorted({g['game_id']:g for g in records}.values(),key=lambda x:(x['date'],x['game_id']))
    if len(records) < min_games:raise ValueError(f'{sport}: {len(records)} completed games; need at least {min_games}')
    X,y,states,dates=training_rows(records,sport)
    split=int(len(X)*.8)
    if len(set(y[:split]))<2 or len(set(y[split:]))<2:raise ValueError('Not enough outcomes to validate')
    model=LogisticRegression(max_iter=1000,C=.5)
    model.fit(X[:split],y[:split])
    probabilities=model.predict_proba(X[split:])[:,1]
    brier=float(brier_score_loss(y[split:],probabilities))
    base=sum(y[:split])/split
    base_brier=float(brier_score_loss(y[split:],[base]*len(y[split:])))
    metrics={'train_games':split,'holdout_games':len(y)-split,'holdout_from_utc':dates[split],
             'holdout_to_utc':dates[-1],'brier':round(brier,5),
             'baseline_brier':round(base_brier,5),
             'log_loss':round(float(log_loss(y[split:],probabilities)),5),
             'accuracy':round(float(accuracy_score(y[split:],probabilities>=.5)),5),
             'vs_constant_baseline':'better' if brier<base_brier else 'worse'}
    result={'sport':sport,'features':FEATURES,'model':model,'team_states':states,
            'last_game_utc':dates[-1],'trained_utc':datetime.now(timezone.utc).isoformat(),
            'metrics':metrics,'model_version':1,'source':'SportsDataverse ESPN schedules',
            'note':'Only pregame team ratings/form. Injuries are displayed separately; NOT modeled.'}
    path=Path(model_path or (MODELS / f'{sport}_moneyline.joblib'))
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.joblib.tmp');dump(result,tmp);tmp.replace(path)
    return metrics


def load_model(sport,path=None):
    from joblib import load
    path=Path(path or MODELS / f'{sport}_moneyline.joblib')
    if not path.exists():return None
    # Never load untrusted model artifacts: joblib uses pickle.
    bundle=load(path)
    if bundle.get('sport')!=sport or bundle.get('features')!=FEATURES or bundle.get('model_version')!=1:
        raise ValueError(f'Incompatible {sport} model bundle')
    return bundle


def predict(bundle,home_id,away_id,when,season,neutral=False):
    f,_,_=snapshot_features(bundle['team_states'],str(home_id),str(away_id),when,season,neutral)
    return float(bundle['model'].predict_proba([f])[0,1])


def main(argv=None):
    ap=argparse.ArgumentParser(description='Train separate CFB and MBB models on real historical ESPN schedules')
    ap.add_argument('--sports',default='mbb,cfb',help='mbb,cfb')
    ap.add_argument('--seasons',default='',help='Optional comma-separated season years, e.g. 2023,2024,2025')
    ap.add_argument('--refresh-current',action='store_true',help='Re-download newest season for fresh results')
    args=ap.parse_args(argv)
    try:import pyarrow # noqa F401
    except ImportError:ap.error('Install parquet support first: python -m pip install pyarrow')
    failures=0
    for sport in args.sports.split(','):
        if sport not in DATASET_URLS:ap.error('Valid sports: mbb,cfb')
        years=([int(y) for y in args.seasons.split(',') if y] if args.seasons else season_list(sport))
        rows=[];ok=0
        for season in years:
            print(f'[{sport}] Downloading/verifying real final games for season {season}...',flush=True)
            try:
                frame=download_schedule(sport,season,refresh=args.refresh_current and season==max(years))
                data=parse_schedule(frame,sport,season_hint=season)
                rows.extend(data);ok+=1
                print(f'[{sport}] season {season}: {len(data)} final games',flush=True)
            except Exception as exc:
                print(f'[{sport}] skipped season {season}: {exc}',flush=True)
        if not ok:
            print(f'[{sport}] NO DATA; model not created',flush=True)
            failures+=1
            continue
        try:
            result=train(sport,rows)
            print(f'[{sport}] FIT COMPLETED on {len(rows)} real games; out-of-sample metrics: {json.dumps(result)}',flush=True)
        except Exception as exc:
            failures+=1
            print(f'[{sport}] TRAINING FAILED (no pretend estimates): {exc}',flush=True)
    if failures:
        raise SystemExit(f'{failures} college model(s) could not be trained on historical data.')


if __name__=='__main__':main()
