"""Parlay Guard automatic live NBA research snapshot.

Local development service (no order placement). Uses the user's EXISTING frozen
winner and player models, the Polymarket scanner, audited feature reconstructor
and positive-EV screen. Actual two-leg prop tickets are NOT exchange quoted.
"""
from __future__ import annotations
import json
import math
import os
import re
import subprocess
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo
import pandas as pd
import requests
from .game_lines_live import fit_line_models,predict_line
from .live_player_data import update_live_history

ROOT=Path(__file__).resolve().parents[1]
ET=ZoneInfo('America/New_York')
GAMMA='https://gamma-api.polymarket.com'
CLOB='https://clob.polymarket.com'
SLUG=re.compile(r'^nba-([a-z]{2,4})-([a-z]{2,4})-(\d{4}-\d\d-\d\d)$')
BASE_HISTORY=ROOT/'data/processed/player_training_data_v2.csv'
OVERLAY=ROOT/'data/live_player_training_data_v2.csv'
MARKETS=ROOT/'data/polymarket_nba_ev.csv'
SHORTLIST=ROOT/'data/prop_ev_research_shortlist.csv'
FEATURES=ROOT/'data/current_player_features.csv'

NICKNAMES={
 'atl':'hawks','bos':'celtics','bkn':'nets','cha':'hornets','chi':'bulls',
 'cle':'cavaliers','dal':'mavericks','den':'nuggets','det':'pistons',
 'gsw':'warriors','hou':'rockets','ind':'pacers','lac':'clippers',
 'lal':'lakers','mem':'grizzlies','mia':'heat','mil':'bucks',
 'min':'timberwolves','nop':'pelicans','nyk':'knicks','okc':'thunder',
 'orl':'magic','phi':'76ers','phx':'suns','por':'trail blazers',
 'sac':'kings','sas':'spurs','tor':'raptors','uta':'jazz','was':'wizards'
}


def as_num(value):
    try:
        num=float(value)
        return num if math.isfinite(num) else None
    except (ValueError,TypeError):return None


def utc(value):
    try:
        t=pd.to_datetime(value,utc=True,errors='coerce')
        if pd.isna(t):return None
        return t.to_pydatetime()
    except (ValueError,TypeError):return None


def array(value):
    if isinstance(value,list):return value
    if isinstance(value,str):
        try:
            v=json.loads(value);return v if isinstance(v,list) else []
        except (ValueError,TypeError):pass
    return []


def book_ask(book):
    asks=[]
    for level in book.get('asks') or []:
        p=as_num(level.get('price'));size=as_num(level.get('size'))
        if p is not None and size is not None and 0<p<1 and size>0:asks.append((p,size))
    return min(asks) if asks else (None,None)


def match_team(label,away,home):
    l=str(label).strip().lower()
    found=[abbr for abbr in (away,home) if l==abbr or NICKNAMES[abbr] in l]
    return found[0] if len(found)==1 else None


def choose_market_line(markets,kind):
    """Find the most balanced listed market, not an executable betting quote."""
    candidates=[]
    for m in markets:
        if str(m.get('sportsMarketType')).lower()!=kind or m.get('closed'):continue
        q=str(m.get('question') or '')
        if kind=='spreads':
            mt=re.search(r'^Spread:\s*(.+?)\s*\(([+-]?\d+(?:\.\d+)?)\)$',q,re.I)
            if not mt:continue
            line=float(mt.group(2))
            label=mt.group(1).strip()
        else:
            mt=re.search(r'O/U\s+(\d+(?:\.\d+)?)',q,re.I)
            if not mt:continue
            line=float(mt.group(1));label='Game total'
        prices=[as_num(v) for v in array(m.get('outcomePrices'))]
        if len(prices)!=2 or any(p is None for p in prices):continue
        balance=abs(prices[0]-0.5)
        candidates.append((balance,{'market_question':q,'line':line,
                    'team_label':label,'market_id':str(m.get('id') or ''),
                    'indicative_yes_probability':prices[0],
                    'is_executable_quote':False}))
    return min(candidates,key=lambda x:x[0])[1] if candidates else None


def discover_live_events(session,now,*,max_games=12,pages=2):
    items=[];seen=set()
    for p in range(pages):
        r=session.get(GAMMA+'/events',params={'series_id':'10345','active':'true','closed':'false','limit':100,'offset':p*100},timeout=(5,20))
        r.raise_for_status();batch=r.json()
        if not isinstance(batch,list):raise ValueError('Gamma events response must be a list')
        for e in batch:
            slug=str(e.get('slug') or '');mt=SLUG.fullmatch(slug)
            if not mt or slug in seen:continue
            seen.add(slug)
            when=utc(e.get('gameStartTime'))
            if when is None:
                for m in e.get('markets') or []:
                    when=utc(m.get('gameStartTime'))
                    if when is not None:break
            if when is None or when<=now:continue
            away,home,slug_day=mt.groups()
            if away not in NICKNAMES or home not in NICKNAMES:continue
            # A game's NBA ET date is authoritative, not Gamma event-creation date.
            if when.astimezone(ET).date().isoformat()!=slug_day:continue
            items.append({'slug':slug,'event':str(e.get('title') or ''),
                 'away':away.upper(),'home':home.upper(),'tipoff_utc':when.isoformat(),
                 'game_date':slug_day,'markets':e.get('markets') or []})
        if len(batch)<100:break
    items.sort(key=lambda x:x['tipoff_utc'])
    return items[:max_games]


def moneyline_quotes(session,events):
    token_to_team={};by_slug={}
    for e in events:
        by_slug[e['slug']]={}
        for market in e['markets']:
            if str(market.get('sportsMarketType') or '').lower()!='moneyline' or market.get('closed'):
                continue
            names=array(market.get('outcomes')); tokens=array(market.get('clobTokenIds'))
            if len(names)!=2 or len(tokens)!=2:continue
            matched={}
            for label,token in zip(names,tokens):
                abbr=match_team(label,e['away'].lower(),e['home'].lower())
                if abbr:matched[abbr.upper()]=str(token)
            if set(matched)=={e['home'],e['away']}:
                for abbr,token in matched.items():token_to_team[token]=(e['slug'],abbr)
                break
    unique=list(token_to_team)
    for start in range(0,len(unique),100):
        ids=unique[start:start+100]
        response=session.post(CLOB+'/books',json=[{'token_id':t} for t in ids],timeout=(5,18))
        response.raise_for_status()
        books=response.json()
        if not isinstance(books,list):raise ValueError('Invalid CLOB batch response')
        for b in books:
            if not isinstance(b,dict):continue
            token=str(b.get('asset_id') or b.get('token_id') or '')
            if token not in token_to_team:continue
            slug,team=token_to_team[token];price,size=book_ask(b)
            by_slug[slug][team]={'ask':price,'shares':size,'token_id':token}
    return by_slug


def _command(args,*,timeout=180):
    proc=subprocess.run([sys.executable,*args],cwd=str(ROOT),capture_output=True,
                        text=True,encoding='utf-8',errors='replace',timeout=timeout)
    if proc.returncode:
        raise RuntimeError(f'{" ".join(args)} failed: {(proc.stderr or proc.stdout)[-600:]}')
    return proc.stdout[-350:]


def update_forecasts(game_dates,now,errors):
    """Refresh frozen-model predictions from NBA source; fall back to recent cache."""
    for index,day in enumerate(sorted(game_dates)[:3]):
        season_start=int(day[:4]) if int(day[5:7])>=7 else int(day[:4])-1
        csv=ROOT/f'data/nba_{season_start}_{(season_start+1)%100:02d}_latest_raw.csv'
        cmd=['-m','src.predict_today','--date',day]
        # Reuse the just-refreshed NBA season feed for additional dates.
        if index>0 and csv.exists():
            try:
                _command(cmd+['--season-csv',str(csv)],timeout=80)
                continue
            except Exception as exc:
                errors.append(f'Winner forecast {day}: {str(exc)[:140]}')
                continue
        try:
            # Without --season-csv, src.predict_today contacts official NBA Stats
            # for the latest COMPLETED team games and refreshes its local cache.
            _command(cmd,timeout=170)
        except Exception as exc:
            fresh=False
            if csv.exists():
                try:
                    dates=pd.read_csv(csv,usecols=['GAME_DATE'])
                    latest=pd.to_datetime(dates.GAME_DATE,errors='coerce').max()
                    fresh=pd.notna(latest) and 0<=(pd.Timestamp(day)-latest).days<=2
                except (OSError,KeyError,ValueError):pass
            if fresh:
                try:
                    _command(cmd+['--season-csv',str(csv)],timeout=80)
                    errors.append(f'NBA team API unavailable for {day}; used verified recent cached team results.')
                    continue
                except Exception as fallback_exc:
                    errors.append(f'Winner forecast {day}: {str(fallback_exc)[:150]}')
            else:
                errors.append(f'Winner forecast {day}: {str(exc)[:150]}')


def latest_game_forecasts(events,now):
    by_match={}
    dates={g['game_date'] for g in events}
    for day in dates:
        files=sorted(
            (ROOT / 'models').glob(f'live_predictions_v1_{day}_*.csv'),
            reverse=True
        )
        for path in files[:5]:
            try: data=pd.read_csv(path,dtype={'GAME_ID':str})
            except (OSError,ValueError):continue
            required={'GAME_DATE','GENERATED_AT_UTC','AWAY_TEAM','HOME_TEAM','TIPOFF_UTC','P_HOME_WIN_ELO'}
            if not required.issubset(data):continue
            for row in data.to_dict('records'):
                if str(row['GAME_DATE'])!=day:continue
                start=utc(row['TIPOFF_UTC']);generated=utc(row['GENERATED_AT_UTC'])
                if start is None or generated is None or not generated<=now<start:continue
                # Do not display forecasts older than 6 hours as live.
                if (now-generated).total_seconds()>6*3600:continue
                away,home=str(row['AWAY_TEAM']).upper(),str(row['HOME_TEAM']).upper()
                p=as_num(row['P_HOME_WIN_ELO']);p2=as_num(row.get('P_HOME_WIN_GB'))
                if p is None or not 0<=p<=1:continue
                by_match.setdefault((day,away,home),{
                    'home_probability':round(p,5),
                    'home_gb_probability':round(p2,5) if p2 is not None else None,
                    'predicted_winner':home if p>=.5 else away,
                    'predicted_winner_probability':round(max(p,1-p),5),
                    'model':'Frozen Elo home-win model',
                    'forecast_generated_at_utc':generated.isoformat(),
                    'nba_game_id':str(row.get('GAME_ID') or ''),
                })
    return by_match


def lines_from_history(now,errors):
    try:
        from .predict_today import (
            load_history,
            clean_team_history,
            make_completed_games,
            fetch_current_season,
        )
        from .common import DATA
        day=now.astimezone(ET).date()
        past=load_history(DATA/'nba_games_updated.csv')
        season=f'{day.year}-{(day.year+1)%100:02d}' if day.month>=7 else f'{day.year-1}-{day.year%100:02d}'
        current_path=DATA/f'nba_{season.replace("-","_")}_latest_raw.csv'
        if not current_path.exists():raise FileNotFoundError(f'{current_path.name} not found')
        current = fetch_current_season(season, current_path)
        historical=clean_team_history(past,current,day)
        completed=make_completed_games(historical)
        models,hist,metrics=fit_line_models(completed)
        current_age=(day-pd.to_datetime(current.GAME_DATE).max().date()).days
        if current_age>3:raise ValueError(f'Current NBA team results are stale ({current_age} days)')
        return models,hist,metrics
    except Exception as exc:
        errors.append('Spread/total model: '+str(exc)[:175]);return None,None,{}


def eligible_props(shortlist_path,features_path,events,now):
    if not shortlist_path.exists() or not features_path.exists():return {}
    picks=pd.read_csv(shortlist_path);features=pd.read_csv(features_path)
    need={'event_slug','player_name','side','line','stat','ask_price','stressed_ev_roi','game_start_utc','observed_utc'}
    if not need.issubset(picks) or not {'event_slug','player_name','team'}.issubset(features):return {}
    clean_name=lambda value: re.sub(r'[^a-z0-9]','',str(value).casefold())
    feats={(str(row.event_slug),clean_name(row.player_name)):str(row.team).upper()
           for row in features.itertuples()}
    picks['observed_utc']=pd.to_datetime(picks.observed_utc,utc=True,errors='coerce')
    picks['game_start_utc']=pd.to_datetime(picks.game_start_utc,utc=True,errors='coerce')
    eligible_slugs={e['slug'] for e in events}
    results={}
    for row in picks.sort_values('stressed_ev_roi',ascending=False).to_dict('records'):
        slug=str(row.get('event_slug') or '')
        if slug not in eligible_slugs:continue
        if pd.isna(row['observed_utc']) or pd.isna(row['game_start_utc']):continue
        if not (row['observed_utc']<=pd.Timestamp(now) <row['game_start_utc']):continue
        if (pd.Timestamp(now)-row['observed_utc']).total_seconds()>20*60:continue
        price=as_num(row.get('ask_price'));ev=as_num(row.get('stressed_ev_roi'))
        if price is None or ev is None or ev<=0:continue
        team=feats.get((slug,clean_name(row['player_name'])))
        if team is None:continue # Unknown roster assignment is NEVER inferred.
        item={'player':str(row['player_name']),'team':team,'stat':str(row['stat']),
              'side':str(row['side']),'line':float(row['line']),
              'ask':price,'stressed_ev_roi':round(ev,4),
              'model_probability':as_num(row.get('model_probability')),
              'market_url':str(row.get('market_url') or ''),
              'top_ask_shares':as_num(row.get('ask_size_shares'))}
        results.setdefault(slug,[]).append(item)
    return results


def assemble_data(events,win_forecasts,lines,history,metrics,props,quotes,now,errors):
    games=[]
    for event in events:
        slug=event['slug'];day=event['game_date'];home=event['home'];away=event['away']
        win=win_forecasts.get((day,away,home))
        line=None
        if lines is not None:
            line=predict_line(lines,history,home=home,away=away,game_day=day)
        moneyline=quotes.get(slug,{})
        candidates=(props.get(slug,[]) if win else [])
        chosen=[p for p in candidates if p['team']==win['predicted_winner']][:3] if win else []
        # No forcing inferior selections to fill a parlay; 2 distinct players required.
        gs={**{k:v for k,v in event.items() if k!='markets'},
            'winner_forecast':win,'game_line_prediction':line,
            'market_spread':choose_market_line(event['markets'],'spreads'),
            'market_total':choose_market_line(event['markets'],'totals'),
            'moneylines':moneyline,'selected_props':chosen,
            'selection_policy':'Only ranked, liquid positive-stress-EV props from the predicted winning team',
            'injury_status':'unverified'}
        games.append(gs)
    return {'updated_at_utc':now.isoformat(),'games':games,'line_model_metrics':metrics,
            'warnings':errors[-12:],
            'disclaimer': 'Research only. Prop-parlay combined odds are hypothetical, not executable Polymarket Combo prices; injury status and model calibration are unverified.'}


def run_update(*,max_games=10,fetch_players=True,refresh_team=True):
    now=datetime.now(timezone.utc);warnings=[]
    with requests.Session() as session:
        session.headers['User-Agent']='ParlayGuardLiveResearch/1.0'
        events=discover_live_events(session,now,max_games=max_games)
        if not events:
            return assemble_data([],{},None,None,{}, {},{},now,['No future NBA game markets available.'])
        # The established scanner is the single source for market settlement rules.
        _command(['src/polymarket_nba_ev.py','--pages','2','--max-games',str(max_games),
                  '--no-orderbooks'],timeout=80)
        if fetch_players:
            day=now.astimezone(ET).date()
            season=f'{day.year}-{(day.year+1)%100:02d}' if day.month>=7 else f'{day.year-1}-{day.year%100:02d}'
            if BASE_HISTORY.exists():
                try:
                    stats=update_live_history(BASE_HISTORY,OVERLAY,season,day)
                    warnings.append(f'Player logs refreshed: {stats["new_game_rows"]} added completed appearances; latest {stats["latest_player_game"]}.')
                except Exception as exc:
                    warnings.append('Player game-log refresh failed: '+str(exc)[:140])
        history_path=OVERLAY if OVERLAY.exists() else BASE_HISTORY
        prediction_ok=True
        try:
            _command(['-m','src.predict_current_props','--history',str(history_path),
                      '--max-history-age-days','4'],timeout=120)
        except Exception as exc:
            prediction_ok=False
            warnings.append('Player projections unavailable: '+str(exc)[:155])
        _command(['src/polymarket_nba_ev.py','--pages','2','--max-games',str(max_games)],timeout=150)
        try:
            _command(['screen_prop_ev.py','--stake','20'],timeout=55)
        except Exception as exc:
            warnings.append('Prop quality screen unavailable: '+str(exc)[:140])
        # Quote timestamps are later than the refresh's start time.
        # Compare to the CURRENT clock, not the clock before network requests.
        props=(eligible_props(SHORTLIST,FEATURES,events,datetime.now(timezone.utc))
               if prediction_ok else {}) # Fail closed if new player inference failed.
        if refresh_team:
            update_forecasts({e['game_date'] for e in events},now,warnings)
        preds=latest_game_forecasts(events,datetime.now(timezone.utc))
        models,hist,metrics=lines_from_history(now,warnings)
        try:money=moneyline_quotes(session,events)
        except Exception as exc:
            warnings.append('Moneyline order books failed: '+str(exc)[:160]);money={}
        return assemble_data(events,preds,models,hist,metrics,props,money,
                             datetime.now(timezone.utc),warnings)
