"""Read-only Parlay Guard Polymarket NBA price/EV scanner.

Usage (project root):
  python -u -m src.polymarket_nba_ev --predictions models/live_predictions_YYYY-MM-DD_TIMESTAMP.csv
  python -u -m src.polymarket_nba_ev --predictions models/live_predictions_YYYY-MM-DD_TIMESTAMP.csv --fee-rate 0.02

No trading, credentials, or order submission. Requires requests and pandas.
Only pre-tipoff forecasts should be used. Do not use retrospective backtest CSVs.
"""
from __future__ import annotations
import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo
import pandas as pd
import requests

GAMMA = 'https://gamma-api.polymarket.com'
CLOB = 'https://clob.polymarket.com'
ET = ZoneInfo('America/New_York')
ALIASES = {
 'ATL':['atlanta hawks','hawks'], 'BOS':['boston celtics','celtics'],
 'BKN':['brooklyn nets','nets'], 'CHA':['charlotte hornets','hornets'],
 'CHI':['chicago bulls','bulls'], 'CLE':['cleveland cavaliers','cavaliers','cavs'],
 'DAL':['dallas mavericks','mavericks','mavs'], 'DEN':['denver nuggets','nuggets'],
 'DET':['detroit pistons','pistons'], 'GSW':['golden state warriors','warriors'],
 'HOU':['houston rockets','rockets'], 'IND':['indiana pacers','pacers'],
 'LAC':['los angeles clippers','la clippers','clippers'],
 'LAL':['los angeles lakers','la lakers','lakers'],
 'MEM':['memphis grizzlies','grizzlies'], 'MIA':['miami heat','heat'],
 'MIL':['milwaukee bucks','bucks'], 'MIN':['minnesota timberwolves','timberwolves','wolves'],
 'NOP':['new orleans pelicans','pelicans'], 'NYK':['new york knicks','knicks'],
 'OKC':['oklahoma city thunder','thunder'], 'ORL':['orlando magic','magic'],
 'PHI':['philadelphia 76ers','76ers','sixers'], 'PHX':['phoenix suns','suns'],
 'POR':['portland trail blazers','trail blazers','blazers'],
 'SAC':['sacramento kings','kings'], 'SAS':['san antonio spurs','spurs'],
 'TOR':['toronto raptors','raptors'], 'UTA':['utah jazz','jazz'],
 'WAS':['washington wizards','wizards']
}

def get_json(session, base, path, params=None):
    r=session.get(base+path,params=params,timeout=18)
    r.raise_for_status()
    return r.json()

def parsed_list(value):
    if isinstance(value,list): return value
    if isinstance(value,str):
        try:
            result=json.loads(value)
            if isinstance(result,list): return result
        except ValueError: pass
    return []

def team_in(text, abbr):
    t=str(text).lower()
    return any(re.search(r'(?<![a-z])'+re.escape(name)+r'(?![a-z])',t)
               for name in ALIASES.get(abbr,[]))

def matched_teams(text):
    return {a for a in ALIASES if team_in(text,a)}

def date_et(value):
    if not value: return None
    try:
        d=datetime.fromisoformat(str(value).replace('Z','+00:00'))
        if d.tzinfo is None: return None
        return d.astimezone(ET).date().isoformat()
    except ValueError: return None

def discover_events(session, max_pages):
    """Scan NBA-tagged events; tag 745 is confirmed by the user's API response."""
    for page in range(max_pages):
        params={'active':'true','closed':'false','limit':100,'offset':page*100,'tag_id':745}
        items=get_json(session,GAMMA,'/events',params)
        if not isinstance(items,list):
            raise RuntimeError('Unexpected Gamma /events response format')
        for event in items: yield event
        if len(items)<100: break

def event_game_date(event):
    """Only use explicit game timing or an ISO date embedded in the event slug.

    startDate is typically market creation, NOT game start, so exclude it.
    """
    for field in ('gameStartTime','startTime'):
        d=date_et(event.get(field))
        if d: return d
    for market in event.get('markets') or []:
        for field in ('gameStartTime','startTime'):
            d=date_et(market.get(field))
            if d: return d
    slug=str(event.get('slug') or '')
    match=re.search(r'(20\d{2})-(\d{2})-(\d{2})(?!\d)',slug)
    if match:
        try:
            return datetime.strptime(match.group(0),'%Y-%m-%d').date().isoformat()
        except ValueError: pass
    return None

def event_teams(event):
    title=str(event.get('title') or event.get('question') or '')
    found=matched_teams(title)
    if len(found)==2: return found
    # NBA game slugs commonly use 3-letter abbreviations separated by hyphens.
    slug=str(event.get('slug') or '').lower()
    short={'atl':'ATL','bos':'BOS','bkn':'BKN','cha':'CHA','chi':'CHI',
           'cle':'CLE','dal':'DAL','den':'DEN','det':'DET','gsw':'GSW',
           'hou':'HOU','ind':'IND','lac':'LAC','lal':'LAL','mem':'MEM',
           'mia':'MIA','mil':'MIL','min':'MIN','nop':'NOP','nyk':'NYK',
           'okc':'OKC','orl':'ORL','phi':'PHI','phx':'PHX','por':'POR',
           'sac':'SAC','sas':'SAS','tor':'TOR','uta':'UTA','was':'WAS'}
    codes={short[x] for x in re.findall(r'(?<![a-z])(?:'+ '|'.join(short) +r')(?![a-z])',slug)}
    return codes if len(codes)==2 else found

def find_market(event, home, away):
    """Require an explicit two-team game and unambiguous outcome mapping."""
    title=str(event.get('title') or event.get('question') or '')
    if event_teams(event)!={home,away}: return None
    for market in event.get('markets',[]):
        if market.get('closed') or market.get('active') is False or market.get('acceptingOrders') is False:
            continue
        outcomes=parsed_list(market.get('outcomes'))
        tokens=parsed_list(market.get('clobTokenIds'))
        if len(outcomes)!=2 or len(tokens)!=2: continue
        names=[matched_teams(x) for x in outcomes]
        mapping={}
        if names[0]=={home} and names[1]=={away}:
            mapping={home:str(tokens[0]),away:str(tokens[1])}
        elif names[0]=={away} and names[1]=={home}:
            mapping={away:str(tokens[0]),home:str(tokens[1])}
        elif {str(x).lower() for x in outcomes}=={'yes','no'}:
            question=str(market.get('question') or '')
            identified=matched_teams(question)
            # Only 'Will TEAM win?' is a direct two-team moneyline.
            if len(identified)==1 and re.search(r'\b(win|beat|defeat)\b',question,re.I):
                named=next(iter(identified))
                if named in (home,away):
                    yes_idx=next(i for i,x in enumerate(outcomes) if str(x).lower()=='yes')
                    mapping={named:str(tokens[yes_idx]),
                             (away if named==home else home):str(tokens[1-yes_idx])}
        if mapping and not any(word in str(market.get('question','')).lower() for word in
                               ('spread','handicap','total points','over/under','first half','quarter')):
            return market,mapping
    return None

def best_ask(session, token, min_shares):
    book=get_json(session,CLOB,'/book',{'token_id':token})
    asks=[]
    for level in book.get('asks',[]):
        try:
            price=float(level['price']); size=float(level['size'])
            if 0<price<1 and size>0: asks.append((price,size))
        except (ValueError,TypeError,KeyError): continue
    asks.sort()
    if not asks: return None
    # Top-of-book is a quoted price, not a guarantee the desired size fills.
    p,qty=asks[0]
    return p,qty,sum(q for _,q in asks)

def evaluate(prob, price, fee_rate, slippage, stake):
    effective=price+slippage
    if not 0<effective<1: return None
    # Conservative fee assumption: proportional fee on amount spent.
    total_cost=stake*(1+fee_rate)
    shares=stake/effective
    win_profit=shares-total_cost
    loss_profit=-total_cost
    ev=prob*win_profit+(1-prob)*loss_profit
    return dict(effective_price=effective,break_even=effective*(1+fee_rate),
                expected_profit=ev,expected_roi=ev/total_cost,
                profit_if_win=win_profit,loss_if_lose=loss_profit,
                shares=shares)

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--predictions',required=True,type=Path)
    ap.add_argument('--fee-rate',type=float,default=0.,help='Estimated fees as fraction of spend, e.g. 0.02')
    ap.add_argument('--slippage',type=float,default=0.01,help='Additional per-share price buffer, default $0.01')
    ap.add_argument('--stake',type=float,default=10.)
    ap.add_argument('--min-edge',type=float,default=0.05,help='Minimum model probability minus break-even, default 0.05')
    ap.add_argument('--max-pages',type=int,default=8,help='Max 500-event Gamma pages to scan')
    args=ap.parse_args()
    if not 0<=args.fee_rate<1 or not 0<=args.slippage<0.5 or args.stake<=0 or not 0<=args.min_edge<1:
        ap.error('Invalid fee, slippage, stake, or edge')
    df=pd.read_csv(args.predictions,dtype={'GAME_ID':str})
    required={'GAME_DATE','HOME_TEAM','AWAY_TEAM','P_HOME_WIN_GB','TIPOFF_UTC','GENERATED_AT_UTC'}
    if not required.issubset(df.columns):
        raise ValueError(f'Expected pre-tipoff live prediction CSV, missing {sorted(required-set(df.columns))}')
    now=datetime.now(timezone.utc)
    tips=pd.to_datetime(df.TIPOFF_UTC,utc=True,errors='coerce')
    created=pd.to_datetime(df.GENERATED_AT_UTC,utc=True,errors='coerce')
    upcoming=df.loc[(tips>pd.Timestamp(now)) & (created<tips)].copy()
    if upcoming.empty:
        print('No future-tipoff games in prediction file. No live EV evaluation performed.')
        return
    session=requests.Session()
    session.headers.update({'User-Agent':'ParlayGuard-Research/1.0','Accept':'application/json'})
    target_dates=set(upcoming.GAME_DATE.astype(str))
    events=[]
    print('Fetching active Polymarket events...',flush=True)
    for event in discover_events(session,args.max_pages):
        title=str(event.get('title') or '')
        if len(event_teams(event))!=2: continue
        # Use gameStartTime/startTime or explicit date in slug; not creation date.
        event_date=event_game_date(event)
        if event_date in target_dates: events.append((event,event_date))
    print(f'Found {len(events)} two-team event candidates on forecast dates.',flush=True)
    rows=[]
    for game in upcoming.itertuples(index=False):
        home,away=str(game.HOME_TEAM),str(game.AWAY_TEAM)
        options=[]
        for event,event_date in events:
            if event_date!=str(game.GAME_DATE): continue
            result=find_market(event,home,away)
            if result: options.append((event,*result))
        if len(options)!=1:
            print(f'SKIP {away} @ {home}: {len(options)} unambiguous matching markets (need exactly 1)',flush=True)
            continue
        event,market,tokens=options[0]
        for team,prob in ((home,float(game.P_HOME_WIN_GB)),(away,1-float(game.P_HOME_WIN_GB))):
            try: ask=best_ask(session,tokens[team],args.stake)
            except requests.RequestException as exc:
                print(f'SKIP {team}: order book error: {exc}',flush=True); continue
            if ask is None:
                print(f'SKIP {team}: no executable asks',flush=True); continue
            price,top_size,total_size=ask
            calc=evaluate(prob,price,args.fee_rate,args.slippage,args.stake)
            if calc is None: continue
            enough_top=top_size*price>=args.stake
            edge=prob-calc['break_even']
            status=('POTENTIAL EDGE — REVIEW' if edge>=args.min_edge and enough_top
                    else 'LOW TOP-BOOK DEPTH' if not enough_top else 'PASS')
            rows.append({'checked_at_utc':now.isoformat(),'game_date':str(game.GAME_DATE),
                         'away':away,'home':home,'team':team,'model_win_prob':prob,
                         'best_ask':price,'top_ask_shares':top_size,'top_ask_depth_usd':top_size*price,
                         'fee_rate_assumed':args.fee_rate,'slippage_assumed':args.slippage,
                         'break_even_prob':calc['break_even'],'edge':edge,
                         'expected_roi':calc['expected_roi'],'expected_profit_usd':calc['expected_profit'],
                         'profit_if_win_usd':calc['profit_if_win'],'loss_if_lose_usd':calc['loss_if_lose'],
                         'status':status,'market_question':market.get('question',''),
                         'market_id':market.get('id'),'token_id':tokens[team],
                         'url':'https://polymarket.com/event/'+str(event.get('slug',''))})
    if not rows:
        print('No safely matched, tradable NBA prices found. No EV claims made.')
        print('Check market availability, API access, event timestamps, or increase --max-pages.')
        return
    result=pd.DataFrame(rows).sort_values(['game_date','away','home','team'])
    view=result[['away','home','team','model_win_prob','best_ask','break_even_prob','edge','expected_roi','status']].copy()
    for col in ['model_win_prob','break_even_prob','edge','expected_roi']:
        view[col]=view[col].map(lambda x:f'{x:+.1%}' if col in ('edge','expected_roi') else f'{x:.1%}')
    view['best_ask']=view.best_ask.map(lambda x:f'${x:.3f}')
    print('\n===== POLYMARKET NBA — ESTIMATED EV =====',flush=True)
    print(view.to_string(index=False),flush=True)
    output=Path('models')/f'polymarket_ev_{now.strftime("%Y%m%dT%H%M%SZ")}.csv'
    output.parent.mkdir(parents=True,exist_ok=True)
    result.to_csv(output,index=False)
    print(f'\nSaved {output}',flush=True)
    print('Research only. Estimated EV depends on model calibration and fee assumptions.',flush=True)
    print('Top-of-book liquidity may not support full stake; no trades were placed.',flush=True)

if __name__=='__main__': main()
