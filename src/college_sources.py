"""Read-only college schedules, Polymarket moneyline asks, roster and injury evidence.

API coverage is incomplete by design; missing reports are NEVER an all-clear.
"""
from __future__ import annotations
import csv
import json
import math
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import requests

ROOT = Path(__file__).resolve().parents[1]
STORE = ROOT / 'data' / 'college_guard'
ET = ZoneInfo('America/New_York')
GAMMA = 'https://gamma-api.polymarket.com'
CLOB = 'https://clob.polymarket.com'
LEAGUES = {
 'mbb':{'label':"NCAA Men's Basketball",'espn':'basketball/mens-college-basketball','group':50,'series':'10470'},
 'cfb':{'label':'NCAA FBS Football','espn':'football/college-football','group':80,'series':'10210'},
}


def utc(value):
    try:
        x=datetime.fromisoformat(str(value).strip().replace('Z','+00:00'))
        return x.astimezone(timezone.utc) if x.tzinfo else None
    except (ValueError,TypeError,AttributeError):return None


def numeric(value):
    try:
        f=float(value)
        return f if math.isfinite(f) else None
    except (ValueError,TypeError):return None


def values(value):
    if isinstance(value,list):return value
    if isinstance(value,str):
        try:
            parsed=json.loads(value)
            if isinstance(parsed,list):return parsed
        except (TypeError,ValueError):pass
    return []


def canonical(value):
    return re.sub(r'[^a-z0-9]','',str(value).lower())


def _names(team):
    arr=[team.get('display_name'),team.get('short_name'),team.get('location'),
         team.get('location','')+' '+team.get('name','')]
    # Abbreviations only count when unique among the TWO opposing teams.
    # A shared acronym (e.g., OSU vs OSU) is ambiguous and rejected.
    names={canonical(n) for n in arr if len(canonical(n))>3}
    ab=canonical(team.get('abbr') or '')
    if len(ab)>=2:names.add(ab)
    return names


def outcome_team(label, home, away):
    name=canonical(label)
    matched=[team['id'] for team in (home,away) if name in _names(team)]
    return matched[0] if len(matched)==1 else None


def parse_espn(payload, sport, day, now=None, *, phase="pre"):
    now=now or datetime.now(timezone.utc)
    games=[]
    for ev in payload.get('events') or []:
        comp=(ev.get('competitions') or [{}])[0]
        teams={}
        for item in comp.get('competitors') or []:
            which=item.get('homeAway')
            if which not in ('home','away'):continue
            t=item.get('team') or {}
            if not t.get('id'):continue
            logo=(t.get('logos') or [{}])[0].get('href') or ''
            if not logo.startswith('https://a.espncdn.com/'):logo=''
            teams[which]={'id':str(t['id']),'display_name':str(t.get('displayName') or ''),
                          'short_name':str(t.get('shortDisplayName') or ''),
                          'location':str(t.get('location') or ''),'name':str(t.get('name') or ''),
                          'abbr':str(t.get('abbreviation') or ''),'logo':logo}
        if set(teams)!={'home','away'}:continue
        event_status=ev.get('status') or comp.get('status') or {}
        state=str((event_status.get('type') or {}).get('state') or '')
        if state != phase:continue
        when=utc(ev.get('date') or comp.get('date'))
        if when is None or (phase == "pre" and when <= now+timedelta(minutes=5)):continue
        if when.astimezone(ET).date().isoformat() != day:continue
        current = {}
        if phase == "in":
            scores = {item.get("homeAway"): item.get("score") for item in comp.get("competitors") or []}
            current = {"score_text": f"{scores.get('home')}-{scores.get('away')}",
                       "period_text": f"Q{event_status.get('period')}",
                       "clock_text": event_status.get("displayClock") or ""}
        games.append({**current,'sport':sport,'espn_id':str(ev.get('id') or ''),'home':teams['home'],
                      'away':teams['away'],'start_utc':when.isoformat(),
                      'neutral':bool(comp.get('neutralSite')),
                      'season':int((ev.get('season') or {}).get('year') or (when.year + (when.month>=10 and sport=='mbb'))),
                      'name':str(ev.get('name') or '')})
    return games


def fetch_scoreboard(session,sport,day,now=None):
    conf=LEAGUES[sport]
    url=f'https://site.api.espn.com/apis/site/v2/sports/{conf["espn"]}/scoreboard'
    r=session.get(url,params={'dates':day.replace('-',''),'groups':conf['group'],'limit':500},timeout=(6,18))
    r.raise_for_status()
    return parse_espn(r.json(),sport,day,now)


def match_markets(session,sport,games,now=None,pages=2):
    """Require unique BOTH-team name match AND a matched tipoff within 2h.
    Ambiguous/unsupported Gamma markets are skipped, never assigned by fuzzy substring.
    """
    now=now or datetime.now(timezone.utc)
    by_game={g['espn_id']:{} for g in games}
    tokens={};matched=set();errors=[]
    for page in range(pages):
        response=session.get(GAMMA+'/events',params={'series_id':LEAGUES[sport]['series'],
            'active':'true','closed':'false','limit':100,'offset':page*100},timeout=(5,20))
        response.raise_for_status()
        batch=response.json()
        if not isinstance(batch,list):raise ValueError('Polymarket events must be a list')
        for ev in batch:
            when=utc(ev.get('gameStartTime'))
            if when is None:
                for m in ev.get('markets') or []:
                    when=utc(m.get('gameStartTime'))
                    if when is not None:break
            if when is None or when <= now:continue
            for market in ev.get('markets') or []:
                if market.get('closed') or str(market.get('sportsMarketType') or '').lower()!='moneyline':continue
                outcomes=values(market.get('outcomes'));ids=values(market.get('clobTokenIds'))
                if len(outcomes)!=2 or len(ids)!=2:continue
                for game in games:
                    key=game['espn_id']
                    if key in matched or abs((utc(game['start_utc'])-when).total_seconds())>7200:continue
                    mapping=[outcome_team(x,game['home'],game['away']) for x in outcomes]
                    if len(set(mapping))!=2 or set(mapping)!={game['home']['id'],game['away']['id']}:continue
                    for team_id,token in zip(mapping,ids):tokens[str(token)]=(key,team_id)
                    matched.add(key)
                    break
        if len(batch)<100:break
    for start in range(0,len(tokens),100):
        ids=list(tokens)[start:start+100]
        r=session.post(CLOB+'/books',json=[{'token_id':s} for s in ids],timeout=(6,20))
        r.raise_for_status()
        response=r.json()
        if not isinstance(response,list):raise ValueError('Invalid order book response')
        for book in response:
            key=str(book.get('asset_id') or book.get('token_id') or '')
            if key not in tokens:continue
            asks=[]
            for row in book.get('asks') or []:
                p,q=numeric(row.get('price')),numeric(row.get('size'))
                if p is not None and q is not None and 0<p<1 and q>0:asks.append((p,q))
            if asks:
                ask,shares=min(asks)
                game_id,team_id=tokens[key]
                by_game[game_id][team_id]={'ask':ask,'shares':shares,'token_id':key,
                    'observed_utc':datetime.now(timezone.utc).isoformat(),
                    'source':'Polymarket.com CLOB observed best ask'}
    return by_game


def _roster_athletes(payload):
    found={}
    def add(item):
        if not isinstance(item,dict):return
        team_list=item.get('items') or item.get('athletes')
        if isinstance(team_list,list):
            for child in team_list:add(child)
        ath=item.get('athlete') or item
        if isinstance(ath,dict) and ath.get('id'):
            name=ath.get('displayName') or ath.get('fullName') or ath.get('name')
            if name:found[str(ath['id'])]=str(name)
    for item in payload.get('athletes') or []:add(item)
    return found


def _injury_items(payload):
    results=[]
    injuries=payload.get('injuries') or payload.get('items') or []
    if isinstance(injuries,dict):injuries=injuries.get('items') or []
    for row in injuries:
        if not isinstance(row,dict):continue
        ath=row.get('athlete') or {}
        name=str(ath.get('displayName') or ath.get('fullName') or row.get('name') or '')
        description=str(row.get('status') or row.get('type') or row.get('description') or 'reported injury')
        if isinstance(row.get('status'),dict):description=str(row['status'].get('description') or row['status'].get('name') or 'reported injury')
        if name:results.append({'player':name,'status':description[:100],
                                 'reported_date':str(row.get('date') or row.get('updated') or ''),
                                 'source':'ESPN injuries feed'})
    return results[:20]


def check_team(sport,team_id,*,client=None,root=ROOT,now=None):
    """Do not equate a missing team injury feed with a verified healthy lineup."""
    now=now or datetime.now(timezone.utc)
    conf=LEAGUES[sport];base=f'https://site.api.espn.com/apis/site/v2/sports/{conf["espn"]}/teams/{team_id}'
    client=client or requests
    evidence={'team_id':team_id,'checked_utc':now.isoformat(),
              'injury_status':'unverified','roster_status':'unverified',
              'injuries':[],'roster_changes':[],'source_url':base+'/injuries'}
    try:
        response=client.get(base+'/injuries',timeout=(4,9));response.raise_for_status()
        obj=response.json()
        evidence['injuries']=_injury_items(obj)
        evidence['injury_status']='reported' if evidence['injuries'] else 'no_reports_in_feed_not_verified_healthy'
    except (requests.RequestException,ValueError,TypeError,KeyError):pass
    try:
        response=client.get(base+'/roster',timeout=(4,9));response.raise_for_status()
        players=_roster_athletes(response.json())
        if not players:return evidence
        evidence['roster_status']='baseline_recorded'
        path=Path(root)/'data'/'college_guard'/'rosters'/f'{sport}_{team_id}.json'
        path.parent.mkdir(parents=True,exist_ok=True)
        previous=json.loads(path.read_text(encoding='utf8')) if path.exists() else {}
        # Compare within same season to avoid treating annual graduation as a fresh absence.
        local=now.astimezone(ET)
        season=int(local.year+1 if sport=='mbb' and local.month>=8 else
                   local.year-1 if sport=='cfb' and local.month<=2 else local.year)
        events=[]
        if previous.get('season')==season:
            for old in previous.get('recent_changes') or []:
                t=utc(old.get('observed_utc'))
                if t and timedelta(0)<=now-t<timedelta(days=7):events.append(old)
            if previous.get('players'):
                prior=previous['players']
                for key in sorted(set(prior)-set(players)):
                    events.append({'player':prior[key],'change':'No longer listed by ESPN','observed_utc':now.isoformat()})
                for key in sorted(set(players)-set(prior)):
                    events.append({'player':players[key],'change':'Newly listed by ESPN','observed_utc':now.isoformat()})
            evidence['roster_status']='changed' if events else 'unchanged_listing'
        evidence['roster_changes']=events[-30:]
        tmp=path.with_suffix('.tmp')
        tmp.write_text(json.dumps({'season':season,'checked_utc':now.isoformat(),
                                   'players':players,'recent_changes':events[-30:]}),encoding='utf8')
        tmp.replace(path)
    except (requests.RequestException,ValueError,TypeError,OSError,KeyError):pass
    return evidence


def manual_report(sport,team_id,root=ROOT,now=None):
    """Optional attributable observations. Require URL and <=36h recency."""
    path=Path(root)/'data'/'college_guard'/'roster_reports.csv'
    if not path.exists():return []
    now=now or datetime.now(timezone.utc);found=[]
    with path.open(encoding='utf-8-sig',newline='') as f:
        for row in csv.DictReader(f):
            if row.get('sport')!=sport or str(row.get('team_id'))!=str(team_id):continue
            t=utc(row.get('checked_at_utc'))
            if not t or t>now or now-t>timedelta(hours=36) or not str(row.get('source_url') or '').startswith('https://'):continue
            found.append({'player':str(row.get('player') or 'Unknown'),
                          'status':str(row.get('status') or 'Status to verify'),
                          'reported_date':t.isoformat(),'source':'User-supplied report',
                          'source_url':row['source_url']})
    return found[:20]


def injury_report(sport,games,*,root=ROOT,max_teams=28,now=None):
    now=now or datetime.now(timezone.utc)
    teams=[]
    for g in games:
        for side in ('home','away'):
            k=g[side]['id']
            if k not in teams:teams.append(k)
    # Avoid saturating ESPN on a very large college slate; mark other teams unreviewed.
    teams=teams[:max_teams]
    results={}
    with ThreadPoolExecutor(max_workers=7) as ex:
        jobs={ex.submit(check_team,sport,k,root=root,now=now):k for k in teams}
        for f in as_completed(jobs):
            k=jobs[f]
            try:results[k]=f.result()
            except Exception as e:results[k]={'team_id':k,'injury_status':'unverified','roster_status':'unverified',
                                                'injuries':[],'roster_changes':[], 'error':str(e)[:100]}
    for k in teams:
        extras=manual_report(sport,k,root=root,now=now)
        if extras:results[k]['injuries'].extend(extras);results[k]['injury_status']='reported'
    return results
