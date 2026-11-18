"""Read-only NBA points-prop research. No orders or claimed executable Combo quotes.

Uses the user's historical NBA player game CSV to estimate rough probabilities.
These are empirical baselines, not calibrated forecasts; benchmark before using EV.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
import re
import threading
import time
import unicodedata

import pandas as pd
import requests

from .live_polymarket import (
    GAMMA, CLOB, event_teams, event_game_date, find_market,
    get_json, parsed_list,
)

ROOT = Path(__file__).resolve().parents[1]
HISTORY_PATH = ROOT / 'data' / 'nba_player_prop_history.csv'
_CACHE = {}
_LOCK = threading.Lock()
CACHE_SECONDS = 60
MAX_PROPS_PER_GAME = 16  # limits CLOB calls; prioritized by Gamma midpoint


def clean_name(s):
    s = unicodedata.normalize('NFKD', str(s))
    return re.sub(r'[^a-z0-9]+', ' ', ''.join(c for c in s.lower() if not unicodedata.combining(c))).strip()


def match_name_in_question(player, question):
    hay = ' ' + clean_name(question) + ' '
    needle = ' ' + clean_name(player) + ' '
    return needle.strip() and needle in hay


@lru_cache(maxsize=4)
def _read_history(path, modification_time):
    df = pd.read_csv(path, dtype={'GAME_ID':str})
    needed = {'GAME_DATE','PLAYER_NAME','TEAM_ABBREVIATION','PTS','WL'}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError('Player history is missing: ' + ', '.join(sorted(missing)))
    df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'], errors='coerce')
    df['PTS'] = pd.to_numeric(df['PTS'], errors='coerce')
    df['PLAYER_NAME'] = df['PLAYER_NAME'].astype(str).str.strip()
    df['TEAM_ABBREVIATION'] = df['TEAM_ABBREVIATION'].astype(str).str.upper()
    df['WL'] = df['WL'].astype(str).str.upper()
    if 'MIN' in df:
        df['MIN'] = pd.to_numeric(df['MIN'].astype(str).str.replace(r':\d+$', '',regex=True), errors='coerce')
        df = df[df['MIN'].isna() | (df['MIN'] >= 10)]
    df = df[df['GAME_DATE'].notna() & df['PTS'].notna() & df['WL'].isin(['W','L'])]
    if 'GAME_ID' in df:
        df = df.drop_duplicates(['GAME_ID','PLAYER_NAME'])
    return df.sort_values('GAME_DATE',ascending=False)


def game_player_history(game, history_path=HISTORY_PATH):
    path = Path(history_path)
    if not path.is_file():
        return {}, f'No player history CSV at {path}. Run: python -m src.collect_prop_history'
    df = _read_history(str(path), path.stat().st_mtime_ns)
    cutoff = pd.Timestamp(game['date'])
    df = df[(df['GAME_DATE'] < cutoff) & (df['GAME_DATE'] >= cutoff - pd.Timedelta(days=430))]
    # Only include players whose latest recorded appearance was on one of these teams,
    # and played recently. This is not a confirmed pre-game injury/roster check.
    history = {}
    for name, group in df.groupby('PLAYER_NAME', sort=False):
        rows = group.head(45)
        if rows.empty or str(rows.iloc[0]['TEAM_ABBREVIATION']) not in (game['away'],game['home']):
            continue
        if (cutoff - rows.iloc[0]['GAME_DATE']).days > 35:
            continue
        if len(rows) < 8:
            continue
        history[name] = rows
    return history, None


def parse_points_line(question, market):
    """Only parse unambiguous full-game NBA points lines.

    A '25+' line is mathematically the same event as >24.5 points (integer stats).
    """
    match = re.search(
    r'\bpoints?\s+o/u\s+(\d+(?:\.5)?)\b',
    question,
    re.I
)

    if match:
        return float(match.group(1))
    
    q = str(question or '')
    low = q.lower()
    if not re.search(r'\bpoints?\b', low):
        return None
    if re.search(r'\b(first|1st|second|2nd|third|3rd|fourth|4th)\s*(?:half|quarter|period)\b|\bteam total\b|\bcombined\b|\bpoints\s*\+\s*rebounds\b',low):
        return None
    for key in ('line', 'lineValue'):
        raw = market.get(key)
        if raw is not None:
            try:
                val = float(raw)
                if 0 < val < 90 and val % 1 == 0.5:
                    return val
            except (ValueError, TypeError):
                pass
    patterns = [
        (r'\b(\d{1,2}(?:\.\d+)?)\s*\+\s*points\b', True),
        (r'\b(?:over|under|o/u|above|below)\s+(\d{1,2}\.5)\s*points\b', False),
        (r'\bpoints\s+(?:over|under|o/u)\s+(\d{1,2}\.5)\b', False),
        (r'\bpoints\s+(\d{1,2}\.5)\b', False),
    ]
    for expression, plus in patterns:
        m = re.search(expression, low)
        if m:
            value = float(m.group(1))
            if plus and value.is_integer():
                return value - .5
            if not plus and value % 1 == .5:
                return value
    return None


def extract_prop_candidates(event, histories):
    """Collect points Yes/No tokens only when player and line parse confidently."""
    candidates=[]
    for market in event.get('markets') or []:
        if market.get('closed') or market.get('active') is False or market.get('acceptingOrders') is False:
            continue
        q = str(market.get('question') or '')
        # YES must mean OVER. NBA prop markets frequently use '25+ points'.
        # A market asking about UNDER or a generic O/U line has different or
        # ambiguous YES/NO semantics and is deliberately not priced here.
        is_points_market = (
        re.search(r'\bpoints?\s+o/u\s+\d+(?:\.5)?\b', q, re.I)
        or re.search(r'\b(?:over|above)\s+\d+(?:\.5)?\b', q, re.I)
        or re.search(r'\b\d+(?:\.5)?\s*\+\s*points\b', q, re.I)
)
        if not is_points_market:
            continue
        market_type = str(market.get('sportsMarketType') or '').lower()
        if market_type and not ('player' in market_type or 'points' in market_type):
            continue
        line = parse_points_line(q, market)
        outcomes = [str(x).strip().lower() for x in parsed_list(market.get('outcomes'))]
        tokens = parsed_list(market.get('clobTokenIds'))
        if line is None or len(tokens)!=2 or set(outcomes)!={'yes','no'}:
            continue
        names=[n for n in histories if match_name_in_question(n,q)]
        if len(names)!=1:
            continue
        yes_index=outcomes.index('yes'); no_index=outcomes.index('no')
        raw_prices=parsed_list(market.get('outcomePrices'))
        try:
            middle=float(raw_prices[yes_index])
        except (ValueError,TypeError,IndexError):
            middle=.5
        candidates.append({'player': names[0], 'line':line, 'question':q,
            'yes_token':str(tokens[yes_index]), 'no_token':str(tokens[no_index]),
            'market_id':str(market.get('id') or ''), 'distance':abs(middle-.5)})
    candidates.sort(key=lambda c:c['distance'])
    # One line per player chosen near midpoint, to avoid multiple correlated alternates.
    seen=set(); selected=[]
    for c in candidates:
        if c['player'] in seen: continue
        seen.add(c['player']); selected.append(c)
        if len(selected)>=MAX_PROPS_PER_GAME: break
    return selected


def book_quote(token):
    r=requests.get(CLOB+'/book',params={'token_id':token},timeout=9,
                   headers={'User-Agent':'ParlayGuard-Research/1.1'})
    r.raise_for_status()
    data=r.json()
    def best(levels, reverse):
        values=[]
        for level in levels or []:
            try:
                p=float(level['price']); size=float(level['size'])
                if 0 < p < 1 and size > 0: values.append((p,size))
            except (KeyError,TypeError,ValueError):
                pass
        values.sort(key=lambda x:x[0], reverse=reverse)
        return values[0] if values else None
    bid=best(data.get('bids'),True)
    ask=best(data.get('asks'),False)
    return {'bid':bid[0] if bid else None, 'ask':ask[0] if ask else None,
            'ask_dollars':round(ask[0]*ask[1],2) if ask else 0.0,
            'spread':round(ask[0]-bid[0],5) if bid and ask else None}


def estimate(rows, line, team_to_win):
    # Weighted empirical rate; no projected minutes, availability, or opponent correction.
    points=rows['PTS'].tolist()
    wins=(rows['WL']=='W').tolist()
    n=len(points)
    over=[int(p>line) for p in points]
    # mild recency weighting; recent games receive up to 2x older games
    weights=[1+((n-1-i)/max(1,n-1)) for i in range(n)]
    base=sum(o*w for o,w in zip(over,weights))/sum(weights)
    condition=[(o,w) for o,w,win in zip(over,weights,wins) if win == team_to_win]
    count_cond=len(condition)
    cond=(sum(o*w for o,w in condition) + 8*base)/(sum(w for _,w in condition)+8)
    return {'over':round(base,5),'over_given_result':round(cond,5),
            'samples':n,'conditional_samples':count_cond}


def moneyline_recommendation(game, event):
    match=find_market(event, game['home'],game['away'])
    if not match:
        return None, {'reason':'No matching moneyline market'}
    _, mapping=match
    sides={}
    for team, token in mapping.items():
        try:
            book=book_quote(token)
        except requests.RequestException:
            book={'ask':None}
        p1=game['home_v1'] if team==game['home'] else 1-game['home_v1']
        p2=game['home_v2'] if team==game['home'] else 1-game['home_v2']
        ask=book['ask']
        roi=min(p1,p2)/ask-1 if ask and 0<ask<1 else None
        sides[team]={'team':team,'ask':ask,'model_v1':round(p1,4),
                     'model_v2':round(p2,4),'estimated_roi':round(roi,4) if roi is not None else None,
                     'probability_conservative':round(max(0,min(p1,p2)-.05),4)}
    home_avg=(game['home_v1']+game['home_v2'])/2
    favorite=game['home'] if home_avg>=.5 else game['away']
    eligible=[v for v in sides.values() if v['ask'] and
              min(v['model_v1'],v['model_v2'])>=.5 and
              v['probability_conservative']/v['ask']-1>=.05]
    best=max(eligible,key=lambda v:v['probability_conservative']/v['ask']-1) if eligible else None
    return best, {'predicted_winner':favorite,'sides':sides,'reason':None if best else 'No moneyline clears conservative value filter'}


def _event(game):
    slug='nba-'+game['away'].lower()+'-'+game['home'].lower()+'-'+game['date']
    s=requests.Session()
    s.headers.update({'User-Agent':'ParlayGuard-Research/1.1'})
    try:
        event=get_json(s,GAMMA,'/events/slug/'+slug)
    except requests.RequestException as e:
        return None, 'Event lookup failed: '+type(e).__name__
    if not isinstance(event,dict) or event_teams(event)!={game['away'],game['home']} or event_game_date(event)!=game['date']:
        return None, 'Gamma event missing or does not match teams/date'
    return event, None


def analyze_game(game):
    now=datetime.now(timezone.utc).isoformat()
    answer={'updated_at':now,'game':{k:game[k] for k in ('away','home','date')},
            'status':'unavailable', 'best_moneyline':None, 'moneyline':None,
            'best_prop':None,'other_props':[], 'combined':None,'notes':[]}
    event,error=_event(game)
    if error:
        answer['notes'].append(error); return answer
    best_ml, ml_data=moneyline_recommendation(game,event)
    answer['best_moneyline']=best_ml
    answer['moneyline']=ml_data
    history,reason=game_player_history(game)
    if reason:
        answer['notes'].append(reason)
        answer['status']='partial'
        return answer
    print("\n========== PLAYER PROP DEBUG ==========")
    print("Game:", game['away'], "@", game['home'])
    print("Total markets in event:", len(event.get('markets') or []))
    print("Players with usable history:", len(history))

    for market in event.get('markets') or []:
        question = str(market.get('question') or '')
        if 'point' in question.lower() or 'player' in question.lower():
            print({
                'question': question,
                'outcomes': market.get('outcomes'),
                'sportsMarketType': market.get('sportsMarketType'),
                'active': market.get('active'),
                'acceptingOrders': market.get('acceptingOrders')
            })

    print("=======================================\n")
    candidates=extract_prop_candidates(event,history)
    if not candidates:
        answer['notes'].append('No confidently parsed active points markets with matched player history')
        answer['status']='partial';return answer
    # Each player prop has two token books. Parallel IO is limited per opened card.
    tasks={}
    with ThreadPoolExecutor(max_workers=6) as pool:
        for cand in candidates:
            for direction in ('yes','no'):
                tasks[pool.submit(book_quote,cand[direction+'_token'])]=(cand,direction)
        books={}
        for task in as_completed(tasks):
            cand, direction=tasks[task]
            try: books[(cand['market_id'],direction)]=task.result()
            except (requests.RequestException,ValueError): pass
    props=[]
    for cand in candidates:
        rows=history[cand['player']]
        player_team=str(rows.iloc[0]['TEAM_ABBREVIATION'])
        for direction in ('yes','no'):
            book=books.get((cand['market_id'],direction))
            if not book or not book['ask'] or book['spread'] is None: continue
            ask=book['ask']
            over=estimate(rows,cand['line'], team_to_win=True)
            # A combined outcome conditional on the chosen team winning needs
            # the player's own team winning if same side, losing otherwise.
            conditional=estimate(rows,cand['line'], team_to_win=(best_ml is not None and best_ml['team']==player_team))
            p=over['over'] if direction=='yes' else 1-over['over']
            pc=conditional['over_given_result'] if direction=='yes' else 1-conditional['over_given_result']
            raw_roi=p/ask-1
            cautious_roi=max(0,p-.05)/ask-1
            qualifies=(.08<=ask<=.92 and book['spread']<=.12 and
                       book['bid'] is not None and book['bid']>=.05 and
                       book['ask_dollars']>=20 and cautious_roi>=.05)
            props.append({'player':cand['player'],'team':player_team,'stat':'points',
                'selection': 'Over' if direction=='yes' else 'Under','line':cand['line'],
                'market_question':cand['question'], 'market_id':cand['market_id'],
                'ask':round(ask,4),'bid':round(book['bid'],4) if book['bid'] else None,
                'spread':book['spread'],'ask_depth_usd':book['ask_dollars'],
                'estimated_probability':round(p,4),
                'estimated_roi':round(raw_roi,4),'conservative_roi':round(cautious_roi,4),
                'conditional_probability':round(pc,4) if conditional['conditional_samples']>=5 else None,
                'samples':over['samples'],'conditional_samples':conditional['conditional_samples'],
                'qualifies':qualifies,'market_url':'https://polymarket.com/event/'+str(event.get('slug'))})
    props.sort(key=lambda x:(x['qualifies'],x['conservative_roi']), reverse=True)
    qualified=[p for p in props if p['qualifies']]
    answer['best_prop']=qualified[0] if qualified else None
    answer['other_props']=props[:8] if not qualified else [x for x in props if x is not qualified[0]][:8]
    if not qualified:
        answer['notes'].append('No points prop passed minimum sample, price, spread, liquidity and value screens')
    if best_ml and answer['best_prop']:
        prop=answer['best_prop']; win_prob=min(best_ml['model_v1'],best_ml['model_v2'])
        joint=win_prob*prop['conditional_probability'] if prop['conditional_probability'] is not None else None
        answer['combined']={
            'moneyline_team':best_ml['team'],'player':prop['player'],
            'selection':prop['selection'],'line':prop['line'],
            'estimated_joint_probability':round(joint,4) if joint is not None else None,
            'maximum_break_even_combined_price':round(joint,4) if joint is not None else None,
            'actual_combo_quote':None,'combined_estimated_roi':None,
            'combined_payout_multiplier':None,
            'status':'Awaiting verified Combo availability and executable quote',
            'method':'Team win model × smoothed player prop frequency conditional on team result (exploratory; not calibrated)'}
    else:
        answer['combined']={'status':'Not evaluated: need qualifying moneyline and player prop',
                            'actual_combo_quote':None,'combined_estimated_roi':None,
                            'combined_payout_multiplier':None}
    answer['status']='ok'
    return answer


def cached_analyze_game(game):
    key=(game['away'],game['home'],game['date'],game['home_v1'],game['home_v2'])
    now=time.monotonic()
    with _LOCK:
        item=_CACHE.get(key)
        if item and now-item[0]<CACHE_SECONDS:
            return item[1]
    result=analyze_game(game)
    with _LOCK:
        _CACHE[key]=(time.monotonic(),result)
        if len(_CACHE)>250:
            _CACHE.clear()
    return result
