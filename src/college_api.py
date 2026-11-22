"""Read-only FastAPI college moneyline research extension for Parlay Guard.
Separate cached data flow: does not mutate NBA snapshots or frozen paper experiments.
"""
from __future__ import annotations
import os
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import requests
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse

from .college_sources import LEAGUES, ET, ROOT, fetch_scoreboard, match_markets, injury_report, utc
from .college_train import load_model, predict

router=APIRouter()
ROOT=Path(__file__).resolve().parents[1]
_cache={};_mutex=threading.Lock()


def _date(offset, now):
    return (now.astimezone(ET).date()+timedelta(days=offset)).isoformat()


def score_game(game, model, quotes, reports, now, min_ev=.05, fee_buffer=.01):
    """Two-sided yes contracts. No bet recommendation for injury flags or stale quotes."""
    result={**game,'market':{},'predictions':None,'best_pick':None,'injuries':{},'status':'MODEL_NOT_TRAINED'}
    for side in ('home','away'):
        team=game[side]
        result['injuries'][side]=reports.get(team['id'],{
          'team_id':team['id'],'injury_status':'not_checked','roster_status':'not_checked',
          'injuries':[],'roster_changes':[]})
    if not model:return result
    metrics=model['metrics']
    result['model_metrics']=metrics
    if game['home']['id'] not in model['team_states'] or game['away']['id'] not in model['team_states']:
        result['status']='UNKNOWN_TEAM_MODEL'
        return result
    try:p_home=predict(model,game['home']['id'],game['away']['id'],game['start_utc'],game['season'],game['neutral'])
    except Exception as exc:
        result['status']='MODEL_ERROR';result['error']=str(exc)[:120];return result
    result['predictions']={'home':round(p_home,5),'away':round(1-p_home,5),
                           'model_trained_utc':model['trained_utc'],
                           'model_last_game_utc':model['last_game_utc']}
    candidates=[]
    for side in ('home','away'):
        team=game[side];q=quotes.get(team['id'])
        if not q:continue
        ask=q.get('ask');shares=q.get('shares'); observed=utc(q.get('observed_utc'))
        if not isinstance(ask,(int,float)) or not 0<ask<1 or not isinstance(shares,(int,float)) or shares<=0 or not observed:continue
        if not timedelta(0)<=now-observed<timedelta(minutes=15):continue
        if now >= utc(game['start_utc']):continue
        probability=p_home if side=='home' else 1-p_home
        effective=ask+fee_buffer
        if effective>=1:continue
        ev=probability/effective-1
        result['market'][side]={**q,'effective_cost':round(effective,5),'estimated_ev':round(ev,5),
                                 'model_probability':round(probability,5)}
        if ev>=min_ev:candidates.append((ev,side,team))
    if not result['market']:
        result['status']='NO_MATCHED_LIVE_ASK';return result
    if not candidates:
        result['status']='NO_ESTIMATED_EDGE';return result
    candidates.sort(reverse=True,key=lambda x:x[0])
    ev,side,team=candidates[0]
    result['best_pick']={'side':side,'team_id':team['id'],'team_name':team['display_name'],
                         'estimated_ev':round(ev,5),'ask':result['market'][side]['ask'],
                         'model_probability':result['market'][side]['model_probability'],
                         'effective_cost':result['market'][side]['effective_cost']}
    risk=any(r['injuries'] or r['roster_changes'] for r in result['injuries'].values())
    if risk:result['status']='ROSTER_INJURY_REVIEW'
    elif metrics.get('vs_constant_baseline')!='better':result['status']='UNVALIDATED_MODEL'
    else:result['status']='POSITIVE_ESTIMATED_EV_UNVERIFIED_INJURIES'
    return result


def assemble_college(now, date_offset=0, *,session=None, check_rosters=True):
    day=_date(date_offset,now)
    out={'asof_utc':now.isoformat(),'date':day,'date_offset':date_offset,
         'sports':{},'games':[],'warnings':[],
         'disclaimer':'Research only. Estimated EV is NOT a verified edge or executable trade. ESPN roster/injury coverage is incomplete. Polymarket.com quotes are not Polymarket US prices. Never place orders automatically.'}
    session=session or requests.Session()
    for sport, config in LEAGUES.items():
        meta={'label':config['label'],'trained':False,'model_metrics':None,'games_found':0,
              'games_quoted':0,'training_command':'python -m src.college_train --sports '+sport,
              'market_source':'Polymarket.com CLOB (read-only)'}
        model=None
        try:
            model=load_model(sport)
            if model:
                meta['trained']=True;meta['model_metrics']=model['metrics'];meta['last_game_utc']=model['last_game_utc']
        except Exception as exc:out['warnings'].append(f'{sport} model unavailable: {exc}')
        out['sports'][sport]=meta
        try:
            games=fetch_scoreboard(session,sport,day,now)
        except (requests.RequestException,ValueError,TypeError) as exc:
            out['warnings'].append(f'{sport} ESPN schedule failed: {exc}');continue
        games.sort(key=lambda g:g['start_utc'])
        games=games[:int(os.getenv('PG_COLLEGE_MAX_GAMES','55'))]
        meta['games_found']=len(games)
        quotes={}
        try:
            quotes=match_markets(session,sport,games,now)
        except (requests.RequestException,ValueError,TypeError) as exc:
            out['warnings'].append(f'{sport} Polymarket quotes failed: {exc}')
        reports={}
        if check_rosters and games:
            try:reports=injury_report(sport,games,max_teams=int(os.getenv('PG_COLLEGE_ROSTER_TEAMS','28')),now=now)
            except Exception as exc:out['warnings'].append(f'{sport} roster checks failed: {exc}')
        fee=float(os.getenv('PG_COLLEGE_FEE_BUFFER','0.01'))
        min_ev=float(os.getenv('PG_COLLEGE_MIN_EV','0.05'))
        if not 0<=fee<.1:fee=.01
        if not 0<=min_ev<1:min_ev=.05
        for game in games:
            q=quotes.get(game['espn_id'],{})
            if q:meta['games_quoted']+=1
            result=score_game(game,model,q,reports,now,min_ev,fee)
            out['games'].append(result)
    out['games'].sort(key=lambda g:g['start_utc'])
    return out


def get_cached(offset=0):
    now=datetime.now(timezone.utc)
    key=(offset,)
    with _mutex:
        old=_cache.get(key)
        if old and (now-old[0]).total_seconds() < 600:return old[1]
        # Single thread computes shared cache. This can take 15-40s on college slates.
        try:
            result=assemble_college(now,offset)
        except Exception as exc:
            result={'date':_date(offset,now),'asof_utc':now.isoformat(),'sports':{},'games':[],
                    'warnings':[f'College refresh failed: {type(exc).__name__}: {exc}'],
                    'disclaimer':'No current EV is available; no trading.'}
        _cache[key]=(now,result)
        return result


@router.get('/api/college/live')
def college_live(date_offset:int=Query(default=0,ge=0,le=2)):
    return JSONResponse(get_cached(date_offset),headers={'Cache-Control':'no-store,max-age=0'})


@router.post('/api/college/refresh')
def refresh_college():
    with _mutex:_cache.clear()
    return {'queued':False,'cache_invalidated':True,'message':'Next /api/college/live request will fetch fresh read-only data.'}


@router.get('/college')
def college_dashboard():
    path=ROOT/'static'/'college.html'
    if not path.is_file():raise HTTPException(503,'College dashboard asset missing')
    return FileResponse(path,media_type='text/html',headers={'Cache-Control':'no-store'})


@router.get('/college-preview.js')
def college_preview_js():
    path=ROOT/'static'/'college_preview.js'
    if not path.is_file():raise HTTPException(503,'College dashboard preview missing')
    return FileResponse(path,media_type='application/javascript',headers={'Cache-Control':'no-store'})
