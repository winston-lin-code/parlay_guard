"""Read-only cross-sport EV portfolio inputs for Parlay Guard.

NBA and college model predictions come from EXISTING /api/live/data and
/api/college/live endpoints, directly consumed in the browser. This adapter
reads previously generated published-provider reports and a voluntary manual
CSV of independently sourced forecasts/asks for other sports.

This module does not place orders, access a betting balance, mutate models,
or alter frozen experiments. Other-sport rows without a recent, supported
moneyline YES quote and independently supplied fair probability are not
eligible for allocation.
"""
from __future__ import annotations

import csv
import json
import math
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import requests
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse

ROOT = Path(__file__).resolve().parents[1]
ET = ZoneInfo('America/New_York')
ONLINE = ROOT/'data'/'online_ev_plans'
MANUAL = ROOT/'data'/'portfolio_other_sports.csv'
SUPPORTED = {'NBA', 'NCAAB', 'NCAAF', 'NHL', 'NFL', 'SOCCER', 'MLB', 'WNBA', 'MLS', 'CFB', 'MBB'}
router = APIRouter()
# Schedules are only a discovery convenience; they do not imply a priced market.
ESPN_SPORTS = {
    'NHL': ['hockey/nhl'], 'NFL': ['football/nfl'],
    'SOCCER': ['soccer/eng.1', 'soccer/usa.1'],
    'MLB': ['baseball/mlb'], 'WNBA': ['basketball/wnba'],
}
SCHEDULE_CACHE = {}
SCHEDULE_LOCK = threading.Lock()


def number(value):
    if value is None or isinstance(value, bool):
        return None
    try:
        v = float(value)
        return v if math.isfinite(v) else None
    except (ValueError, TypeError):
        return None


def utc(value):
    try:
        d = datetime.fromisoformat(str(value).strip().replace('Z', '+00:00'))
        return d.astimezone(timezone.utc) if d.tzinfo else None
    except (ValueError, AttributeError, OverflowError, TypeError):
        return None


def day_for(now, offset):
    return (now.astimezone(ET).date()+timedelta(days=offset)).isoformat()


def within_day(start, now, offset):
    return start.astimezone(ET).date().isoformat() == day_for(now, offset)


def normalized_online(row, now, offset):
    """Only Owls' explicit Polymarket YES / H2H contract records.

    This is a source-specific paper candidate, not an executable US order.
    quote_age_seconds is measured at source retrieval time and carried forward.
    """
    if not isinstance(row, dict):
        return None
    sport = str(row.get('sport') or '').strip().upper()
    if sport not in SUPPORTED or str(row.get('provider') or '') != 'Owls Insight':
        return None
    if str(row.get('venue') or '') != 'polymarket.com':
        return None
    start = utc(row.get('start_utc'))
    source_at = utc(row.get('source_as_of_utc'))
    quote_age = number(row.get('quote_age_seconds'))
    probability, ask = number(row.get('model_probability')), number(row.get('best_ask'))
    fee = number(row.get('fee_per_share'))
    if (start is None or not within_day(start, now, offset) or not now < start or
        not source_at or source_at > now or quote_age is None or quote_age < 0 or
        probability is None or not 0 < probability < 1 or ask is None or not 0 < ask < 1 or
        fee is None or not 0 <= fee < .25):
        return None
    # Distinguish source retrieval from underlying order-book age.
    observed = source_at-timedelta(seconds=quote_age)
    event_id = str(row.get('event_id') or '').strip()
    pick = str(row.get('pick') or '').strip()
    matchup = str(row.get('game') or '').strip()
    if not event_id or not pick or not matchup:
        return None
    return {
        'id': event_id, 'sport': sport, 'matchup': matchup,
        'start_utc': start.isoformat(), 'source': 'Owls Insight published Polymarket EV',
        'venue': 'polymarket.com (international; paper only)',
        'source_url': str(row.get('source_url') or ''), 'provenance': 'published_provider',
        'outcomes': [{
            'id': str(row.get('side') or pick), 'name': pick, 'probability': probability,
            'ask': ask, 'fee_per_share': fee,
            'observed_utc': observed.isoformat(),
            'shares': number(row.get('liquidity_usd')), 'depth_unit': 'USD',
        }],
        'alerts': ['No independently verified injury/roster coverage'],
        'warning': 'Provider quote and model are not independently verified. No US-executable order mapping.',
    }


def normalized_manual(row, now, offset):
    if not isinstance(row, dict):
        return None
    sport = str(row.get('sport') or '').strip().upper()
    if sport not in SUPPORTED:
        return None
    start = utc(row.get('start_utc'))
    stamp = utc(row.get('quote_observed_utc'))
    probability, ask = number(row.get('win_probability')), number(row.get('ask'))
    fee = number(row.get('fee_per_share') or 0)
    event_id = str(row.get('event_id') or '').strip()
    outcome = str(row.get('pick') or '').strip()
    away, home = str(row.get('away') or '').strip(), str(row.get('home') or '').strip()
    venue = str(row.get('venue') or '').strip().lower()
    if (not start or not within_day(start, now, offset) or start <= now or not stamp or
        not probability or not 0 < probability < 1 or not ask or not 0 < ask < 1 or
        fee is None or not 0 <= fee < .25 or not event_id or not away or not home or
        outcome not in (home, away, 'Draw') or venue not in ('polymarket.com', 'polymarket.us')):
        return None
    if str(row.get('market_type') or '').strip().lower() not in ('moneyline_yes','h2h_yes'):
        return None
    return {
        'id': 'manual:'+sport+':'+event_id, 'sport': sport,
        'matchup': away+' @ '+home, 'start_utc': start.isoformat(),
        'source': str(row.get('source') or 'User-imported independent forecast'),
        'venue': venue+' (manual quote; paper only)',
        'source_url': str(row.get('market_url') or ''), 'provenance': 'user_import',
        'outcomes': [{
            'id': outcome, 'name': outcome, 'probability': probability,
            'ask': ask, 'fee_per_share': fee,
            'observed_utc': stamp.isoformat(),
            'shares': number(row.get('depth_shares')), 'depth_unit': 'shares',
        }],
        'alerts': [str(row.get('injury_note') or 'Injury/roster coverage unverified')],
        'warning': 'Manual market identity, model and quote have not been verified; paper sizing only.',
    }


def merge_events(rows):
    """Group separate imported YES outcomes into one matchup, never combine conflicting games."""
    items = {}
    for e in rows:
        if not e:
            continue
        k = (e['id'], e['start_utc'], e['matchup'], e['venue'])
        if k not in items:
            items[k] = dict(e)
            items[k]['outcomes'] = []
        seen = {str(x['id']) for x in items[k]['outcomes']}
        for o in e['outcomes']:
            if str(o['id']) not in seen:
                items[k]['outcomes'].append(o)
                seen.add(str(o['id']))
    return sorted(items.values(), key=lambda r: (r['start_utc'], r['sport'], r['matchup']))



def espn_schedule_payload(payload, sport, now, offset):
    """Upcoming fixtures only. No invented model probabilities or contract odds."""
    rows=[]
    for event in (payload.get('events') or [])[:180]:
        if not isinstance(event, dict):
            continue
        start = utc(event.get('date'))
        if start is None or start <= now+timedelta(minutes=5) or not within_day(start,now,offset):
            continue
        if str(((event.get('status') or {}).get('type') or {}).get('state') or '') != 'pre':
            continue
        competitors = ((event.get('competitions') or [{}])[0].get('competitors') or [])
        sides={}
        for c in competitors:
            side=c.get('homeAway')
            team=c.get('team') or {}
            if side not in ('home','away'):
                continue
            name=str(team.get('displayName') or team.get('shortDisplayName') or '').strip()
            if not name:
                continue
            logos=team.get('logos') or []
            logo = (logos[0].get('href') if logos and isinstance(logos[0],dict) else None) or team.get('logo') or ''
            if not isinstance(logo,str) or not logo.startswith('https://a.espncdn.com/'):
                logo=''
            sides[side]={'name':name,'logo':logo}
        if set(sides)!={'home','away'}:
            continue
        eid=str(event.get('id') or '')
        if not eid:
            continue
        rows.append({
            'id':'espn-schedule:'+sport+':'+eid,
            'sport': sport,
            'matchup':sides['away']['name']+' @ '+sides['home']['name'],
            'start_utc':start.isoformat(),
            'source':'ESPN schedule (unpriced)', 'venue':'No market quote',
            'source_url':'', 'provenance':'schedule_only',
            'outcomes':[{'id':side,'name':sides[side]['name'],'probability':None,'ask':None,
                         'fee_per_share':0,'observed_utc':None,'shares':None,
                         'depth_unit':'shares','logo':sides[side]['logo']} for side in ('away','home')],
            'alerts':['Roster/injury status not verified'],
            'warning':'Schedule only. A model win probability and fresh YES ask are REQUIRED before allocating.',
        })
    return rows


def fetch_one_schedule(sport, path, now, offset):
    date=day_for(now,offset)
    url='https://site.api.espn.com/apis/site/v2/sports/'+path+'/scoreboard'
    response=requests.get(url,params={'dates':date.replace('-',''),'limit':200},timeout=(4,9),
                          headers={'User-Agent':'ParlayGuardResearch/1.0'})
    response.raise_for_status()
    payload=response.json()
    if not isinstance(payload,dict):
        raise ValueError('Scoreboard response was not a JSON object')
    return espn_schedule_payload(payload,sport,now,offset)


def discover_schedules(now, offset):
    key=day_for(now,offset)
    with SCHEDULE_LOCK:
        hit=SCHEDULE_CACHE.get(key)
        if hit and 0<=(now-hit[0]).total_seconds()<600:
            return hit[1],hit[2]
    entries,errors=[],[]
    jobs=[(sport,path) for sport,paths in ESPN_SPORTS.items() for path in paths]
    with ThreadPoolExecutor(max_workers=6) as pool:
        futures={pool.submit(fetch_one_schedule,sport,path,now,offset):(sport,path) for sport,path in jobs}
        for f in as_completed(futures):
            sport,path=futures[f]
            try:
                entries.extend(f.result())
            except (requests.RequestException,ValueError,TypeError) as exc:
                errors.append(sport+' schedule unavailable: '+str(exc)[:100])
    with SCHEDULE_LOCK:
        SCHEDULE_CACHE[key]=(now,entries,errors)
    return entries,errors


def combine_with_schedules(external,schedules):
    """External priced candidate wins over a matching ESPN schedule placeholder."""
    result=list(external)
    def words(v):
        return ''.join(c for c in str(v).casefold() if c.isalnum())
    for e in schedules:
        # Prevent duplicate game cards by comparing exact normalized school/team
        # names and start times. Never infer moneyline prices from schedules.
        matched=False
        for quoted in external:
            if e['sport']!=quoted['sport']:
                continue
            start,other=utc(e['start_utc']),utc(quoted['start_utc'])
            if start is None or other is None or abs((start-other).total_seconds())>3600:
                continue
            if words(e['matchup'])==words(quoted['matchup']):
                matched=True;break
        if not matched:
            result.append(e)
    return sorted(result,key=lambda x:(x['start_utc'],x['sport'],x['matchup']))

def get_external(now=None, offset=0):
    now = now or datetime.now(timezone.utc)
    date = day_for(now, offset)
    external, watch, warnings = [], [], []
    plan = ONLINE/(date+'.json')
    if plan.is_file():
        try:
            payload = json.loads(plan.read_text(encoding='utf-8'))
            if payload.get('date_et') != date:
                raise ValueError('report date mismatch')
            for item in (payload.get('allocation') or {}).get('bets') or []:
                e = normalized_online(item, now, offset)
                if e:
                    external.append(e)
            # Free demo sports BOOK lines are informational, not Polymarket asks.
            for item in payload.get('sportsbook_watchlist') or []:
                if isinstance(item, dict) and str(item.get('sport') or '').upper() in SUPPORTED:
                    start = utc(item.get('start_utc'))
                    if start and start>now and within_day(start,now,offset):
                        watch.append({
                            'sport':str(item.get('sport') or ''), 'game':str(item.get('game') or ''),
                            'pick':str(item.get('pick') or ''),
                            'provider_ev_pct':number(item.get('provider_ev_pct')),
                            'fair_probability':number(item.get('fair_probability')),
                            'bookmaker':str(item.get('bookmaker') or ''),
                            'start_utc':start.isoformat(),
                            'reason':'Sportsbook price only; NO matched Polymarket moneyline ask',
                        })
        except (OSError, ValueError, TypeError, AttributeError) as exc:
            warnings.append('Online EV report could not be loaded: '+str(exc)[:150])
    if MANUAL.is_file():
        try:
            with MANUAL.open(newline='', encoding='utf-8-sig') as f:
                for index, row in enumerate(csv.DictReader(f), start=2):
                    e = normalized_manual(row, now, offset)
                    if e:
                        external.append(e)
                    elif any((v or '').strip() for v in row.values() if isinstance(v,str)):
                        # Avoid leaking private content; provide line number only.
                        warnings.append('Manual sports row '+str(index)+' rejected (date, contract, source or probability/price checks)')
        except (OSError, UnicodeError, ValueError) as exc:
            warnings.append('Manual sports CSV unreadable: '+str(exc)[:140])
    priced=merge_events(external)
    schedule, schedule_warnings=discover_schedules(now,offset)
    warnings.extend(schedule_warnings[:5])
    return {'date_et':date, 'asof_utc':now.isoformat(), 'games':combine_with_schedules(priced,schedule),
            'sportsbook_watchlist':watch[:100], 'warnings':warnings,
            'info':'Other sports require independently modeled probabilities and timestamped Polymarket YES asks. Sportsbook EV is watchlist only. Paper simulation; no orders.'}


@router.get('/api/portfolio/external')
def portfolio_external(date_offset:int=Query(default=0,ge=0,le=2)):
    return JSONResponse(get_external(offset=date_offset),headers={'Cache-Control':'no-store,max-age=0'})


@router.get('/portfolio')
def portfolio_dashboard():
    page = ROOT/'static'/'portfolio.html'
    if not page.is_file():
        raise HTTPException(503,'Portfolio dashboard asset missing')
    return FileResponse(page, media_type='text/html', headers={'Cache-Control':'no-store'})


@router.get('/portfolio-core.js')
def portfolio_core():
    page = ROOT/'static'/'portfolio_core.js'
    if not page.is_file():
        raise HTTPException(503,'Portfolio allocation code missing')
    return FileResponse(page, media_type='application/javascript', headers={'Cache-Control':'no-store'})


@router.get('/portfolio-ui.js')
def portfolio_ui():
    page = ROOT/'static'/'portfolio_ui.js'
    if not page.is_file():
        raise HTTPException(503,'Portfolio dashboard script missing')
    return FileResponse(page, media_type='application/javascript', headers={'Cache-Control':'no-store'})
