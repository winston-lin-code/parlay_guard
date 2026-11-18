"""Read-only Polymarket NBA moneyline quotes; no orders or authentication."""
import json
import re
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
import requests
GAMMA='https://gamma-api.polymarket.com'
CLOB='https://clob.polymarket.com'
ET=ZoneInfo('America/New_York')
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

def get_live_quotes(games):
    # Scan once for all matchups; never match a game without verified date and teams.
    now=datetime.now(timezone.utc).isoformat()
    wanted={(g['away'],g['home'],g['date']) for g in games}
    result={f'{a}|{h}|{d}':{'status':'unavailable','reason':'No matching active game market'} for a,h,d in wanted}
    session=requests.Session()
    session.headers.update({'User-Agent':'ParlayGuard-Research/1.0','Accept':'application/json'})
    candidates={}
    for event in discover_events(session,max_pages=8):
        d=event_game_date(event)
        if not d: continue
        teams=event_teams(event)
        if len(teams)!=2: continue
        for a,h,day in wanted:
            if day!=d or teams!={a,h}: continue
            key=f'{a}|{h}|{day}'
            match=find_market(event,h,a)
            if match: candidates.setdefault(key,[]).append(match)
    for key,matches in candidates.items():
        if len(matches)!=1:
            result[key]={'status':'unavailable','reason':'Ambiguous matching markets'}
            continue
        market,tokens=matches[0]
        sides={}
        prices=parsed_list(market.get('outcomePrices'))
        outcomes=parsed_list(market.get('outcomes'))
        for team,token in tokens.items():
            # Implied % is indicative outcome price, not wager share or bettor count.
            implied=None
            for idx,outcome in enumerate(outcomes):
                if matched_teams(outcome)=={team} and idx<len(prices):
                    try:
                        val=float(prices[idx])
                        if 0<=val<=1: implied=val
                    except (ValueError,TypeError): pass
            try:
                ask=best_ask(session,token,0)
            except requests.RequestException:
                ask=None
            sides[team]={'implied_probability':implied,'ask':ask[0] if ask else None}
        result[key]={'status':'ok','sides':sides}
    return {'updated_at':now,'games':result,'note':'Market-implied prices are not percentage of wagers. Best asks may have limited liquidity.'}
