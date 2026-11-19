"""Score recorded Parlay Guard strategy snapshots from completed NBA game logs.

Reads data/paper_trades/strategy_snapshot_*.json created with
`add_parlay_comparison.py --record-paper`.

Only a clear player box-score observation on the exact matchup/date settles a
player leg. Missing player, no-show, ambiguous date/team, postponed or void
markets remain PENDING. A verified two-team team-game record settles a
moneyline; it never guesses from prop results.

Reported parlay profits are HYPOTHETICAL: product of single-leg asking prices
is not an executable Combo quote. No win probability is inferred here.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import re
import pandas as pd

DEFAULT_PLAYER = Path('data/processed/player_training_data_v2.csv')
DEFAULT_TEAM = Path('data/nba_2026_27_latest_raw.csv')
DEFAULT_PAPER = Path('data/paper_trades')

TEAM_IDS = {
    'ATL':'1610612737', 'BOS':'1610612738', 'BKN':'1610612751', 'CHA':'1610612766',
    'CHI':'1610612741', 'CLE':'1610612739', 'DAL':'1610612742', 'DEN':'1610612743',
    'DET':'1610612765', 'GSW':'1610612744', 'HOU':'1610612745', 'IND':'1610612754',
    'LAC':'1610612746', 'LAL':'1610612747', 'MEM':'1610612763', 'MIA':'1610612748',
    'MIL':'1610612749', 'MIN':'1610612750', 'NOP':'1610612740', 'NYK':'1610612752',
    'OKC':'1610612760', 'ORL':'1610612753', 'PHI':'1610612755', 'PHX':'1610612756',
    'POR':'1610612757', 'SAC':'1610612758', 'SAS':'1610612759', 'TOR':'1610612761',
    'UTA':'1610612762', 'WAS':'1610612764',
}
TEAM_NAMES = {
    'ATL':('hawks','atlanta hawks'), 'BOS':('celtics','boston celtics'),
    'BKN':('nets','brooklyn nets'), 'CHA':('hornets','charlotte hornets'),
    'CHI':('bulls','chicago bulls'), 'CLE':('cavaliers','cleveland cavaliers'),
    'DAL':('mavericks','dallas mavericks'), 'DEN':('nuggets','denver nuggets'),
    'DET':('pistons','detroit pistons'), 'GSW':('warriors','golden state warriors'),
    'HOU':('rockets','houston rockets'), 'IND':('pacers','indiana pacers'),
    'LAC':('clippers','la clippers','los angeles clippers'),
    'LAL':('lakers','los angeles lakers'), 'MEM':('grizzlies','memphis grizzlies'),
    'MIA':('heat','miami heat'), 'MIL':('bucks','milwaukee bucks'),
    'MIN':('timberwolves','minnesota timberwolves'),
    'NOP':('pelicans','new orleans pelicans'), 'NYK':('knicks','new york knicks'),
    'OKC':('thunder','oklahoma city thunder'), 'ORL':('magic','orlando magic'),
    'PHI':('76ers','sixers','philadelphia 76ers'), 'PHX':('suns','phoenix suns'),
    'POR':('trail blazers','blazers','portland trail blazers'),
    'SAC':('kings','sacramento kings'), 'SAS':('spurs','san antonio spurs'),
    'TOR':('raptors','toronto raptors'), 'UTA':('jazz','utah jazz'),
    'WAS':('wizards','washington wizards'),
}


def clean(s):
    return re.sub('[^a-z0-9]', '', str(s).casefold())


def parse_slug(slug):
    found = re.fullmatch(r'nba-([a-z]{2,3})-([a-z]{2,3})-(\d{4}-\d{2}-\d{2})', str(slug))
    if not found:
        return None
    away,home,date = found.groups()
    away,home = away.upper(),home.upper()
    if away not in TEAM_IDS or home not in TEAM_IDS or away == home:
        return None
    return away,home,date


def map_team(name):
    key = clean(name)
    candidates = [abbr for abbr,aliases in TEAM_NAMES.items()
                  if key == clean(abbr) or key in [clean(x) for x in aliases]]
    return candidates[0] if len(candidates) == 1 else None


def load_player_games(path):
    if not path.is_file():
        print(f'Player game data not found: {path}; props remain pending')
        return None
    df = pd.read_csv(path, low_memory=False, dtype={'TEAM_ID':str})
    required = {'GAME_DATE','PLAYER_NAME','TEAM_ID','MIN','PTS','REB','AST','FG3M'}
    if not required.issubset(df.columns):
        raise ValueError(f'Player games CSV needs {sorted(required-set(df.columns))}')
    df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'],errors='coerce').dt.strftime('%Y-%m-%d')
    df['_NAME']=df['PLAYER_NAME'].map(clean)
    df['TEAM_ID']=df['TEAM_ID'].astype(str).str.replace(r'\.0$', '', regex=True)
    for key in ('MIN','PTS','REB','AST','FG3M'):
        df[key] = pd.to_numeric(df[key], errors='coerce')
    return df


def load_team_games(path):
    if not path.is_file():
        print(f'Team game data not found: {path}; moneylines remain pending')
        return None
    df = pd.read_csv(path, low_memory=False, dtype={'GAME_ID':str})
    required = {'GAME_DATE','TEAM_ABBREVIATION','WL','GAME_ID','PTS'}
    if not required.issubset(df.columns):
        raise ValueError(f'Team game CSV needs {sorted(required-set(df.columns))}')
    df['GAME_DATE']=pd.to_datetime(df['GAME_DATE'],errors='coerce').dt.strftime('%Y-%m-%d')
    df['TEAM_ABBREVIATION']=df['TEAM_ABBREVIATION'].astype(str).str.upper().str.strip()
    df['WL']=df['WL'].astype(str).str.upper().str.strip()
    df['PTS']=pd.to_numeric(df['PTS'], errors='coerce')
    return df


def settle_player(leg, player_games, snapshot_time):
    if player_games is None:
        return None
    parsed=parse_slug(leg['event_slug'])
    if not parsed:
        return None
    a,h,date=parsed
    tipoff=pd.to_datetime(leg['tipoff_utc'],utc=True,errors='coerce')
    if pd.isna(tipoff) or tipoff <= snapshot_time:
        return None  # not a confirmed pregame record
    candidate=player_games[(player_games['GAME_DATE']==date)
                           & (player_games['_NAME']==clean(leg['player']))
                           & player_games['TEAM_ID'].isin([TEAM_IDS[a],TEAM_IDS[h]])]
    if len(candidate)!=1:
        return None  # no-show/cancel/ambiguity: never assume loss or void
    row=candidate.iloc[0]
    value=row.get(leg['stat'])
    if pd.isna(value) or pd.isna(row['MIN']) or row['MIN']<=0:
        return None
    if leg['side']=='OVER':
        return bool(value > float(leg['line']))
    if leg['side']=='UNDER':
        return bool(value < float(leg['line']))
    return None


def game_winner(slug, team_games):
    if team_games is None:
        return None
    parsed=parse_slug(slug)
    if not parsed:
        return None
    a,h,date=parsed
    df=team_games[(team_games['GAME_DATE']==date)
                  & team_games['TEAM_ABBREVIATION'].isin([a,h])]
    # Require the exact paired game_id for both teams.
    matches=[]
    for gameid,g in df.groupby('GAME_ID'):
        if (len(g)==2 and set(g['TEAM_ABBREVIATION'])=={a,h}
                and set(g['WL'])=={'W','L'} and g['PTS'].notna().all()
                and len(set(g['PTS']))==2):
            byabbr=g.set_index('TEAM_ABBREVIATION')
            if bool(byabbr.loc[a,'PTS']>byabbr.loc[h,'PTS']) == bool(byabbr.loc[a,'WL']=='W'):
                matches.append(a if byabbr.loc[a,'WL']=='W' else h)
    return matches[0] if len(matches)==1 else None


def paper_session(snapshot, player_games, team_games, *, file=''):
    observed=pd.to_datetime(snapshot.get('observed_utc'), utc=True, errors='coerce')
    budget=float(snapshot.get('budget',0))
    if pd.isna(observed) or not 0<budget<=100000:
        raise ValueError('Invalid snapshot timestamp or budget')
    legs=snapshot.get('legs',[])
    outcomes={l['id']:settle_player(l,player_games,observed) for l in legs}
    moneylines=snapshot.get('moneylines',[])
    ml={}
    for game in moneylines:
        ix=game.get('chosen')
        if ix is None or not 0<=ix<len(game['sides']):
            continue
        side=game['sides'][ix]
        if side.get('ask') is None:
            continue
        winner=game_winner(game['slug'],team_games)
        abbr=map_team(side['team'])
        ml[game['slug']]=(None if winner is None or abbr is None else abbr==winner, float(side['ask']))
    rows=[]
    for strategy in ('singles','same_game','cross_game','moneyline'):
        if strategy=='moneyline':
            tickets=[{'price':price,'win':won} for won,price in ml.values()]
            # If any original games lack moneyline price, benchmark unavailable.
            if len(tickets)!=len(moneylines):
                tickets=[]
        else:
            tickets=[]
            for ticket in snapshot.get('tickets',{}).get(strategy,[]):
                results=[outcomes.get(id) for id in ticket['leg_ids']]
                tickets.append({'price':float(ticket['cost']),
                                'win':None if any(x is None for x in results) else all(results)})
        count=len(tickets)
        pending=sum(t['win'] is None for t in tickets)
        complete=count>0 and pending==0
        if complete:
            unit=budget/count
            gross=sum(unit/t['price'] for t in tickets if t['win'])
            pnl=gross-budget
        else:
            gross=pnl=None
        rows.append({'snapshot':str(file),'observed_utc':str(observed), 'strategy':strategy,
                     'budget':budget,'ticket_count':count,'pending_tickets':pending,
                     'complete':complete,'won_tickets':sum(t['win'] is True for t in tickets),
                     'gross_payout':gross,'profit':pnl,'roi':pnl/budget if complete else None})
    pending_legs=[{'snapshot':str(file),'event':l['event'],'player':l['player'],
                   'stat':l['stat'],'side':l['side'],'line':l['line'],'game_start_utc':l['tipoff_utc']}
                  for l in legs if outcomes.get(l['id']) is None]
    return rows,pending_legs


def summarize(df):
    # All three prop strategies must be evaluated on the SAME pregame snapshots;
    # otherwise missing sessions could make one strategy look spuriously better.
    prop = df[df['strategy'].isin(['singles','same_game','cross_game'])]
    shared = set()
    for key, group in prop.groupby('snapshot'):
        if set(group.loc[group['complete'], 'strategy']) == {'singles','same_game','cross_game'}:
            shared.add(key)
    out=[]
    for strategy in ('singles','same_game','cross_game','moneyline'):
        complete=df[(df['strategy']==strategy)&(df['complete']==True)&df['snapshot'].isin(shared)].copy()
        if complete.empty:
            out.append({'strategy':strategy,'complete_sessions':0,'roi':float('nan'),
                        'total_stake':0,'total_profit':0,'winning_sessions':0})
        else:
            st=complete['budget'].sum(); p=complete['profit'].sum()
            out.append({'strategy':strategy,'complete_sessions':len(complete),
                        'roi':p/st if st else float('nan'),'total_stake':st,
                        'total_profit':p,'winning_sessions':int((complete['profit']>0).sum())})
    return pd.DataFrame(out)


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--paper-dir',type=Path,default=DEFAULT_PAPER)
    ap.add_argument('--player-games',type=Path,default=DEFAULT_PLAYER)
    ap.add_argument('--team-games',type=Path,default=DEFAULT_TEAM)
    args=ap.parse_args(argv)
    files=sorted(args.paper_dir.glob('strategy_snapshot_*.json'))
    if not files:
        print(f'No paper snapshots in {args.paper_dir}. Run add_parlay_comparison.py --record-paper before games.')
        return
    pg=load_player_games(args.player_games)
    tg=load_team_games(args.team_games)
    all_rows, pending=[],[]
    for path in files:
        content=json.loads(path.read_text(encoding='utf-8'))
        rows,unresolved=paper_session(content,pg,tg,file=path.name)
        all_rows.extend(rows)
        pending.extend(unresolved)
    args.paper_dir.mkdir(parents=True,exist_ok=True)
    df=pd.DataFrame(all_rows)
    df.to_csv(args.paper_dir/'strategy_paper_sessions.csv',index=False)
    summary=summarize(df)
    summary.to_csv(args.paper_dir/'strategy_paper_summary.csv',index=False)
    pd.DataFrame(pending,columns=['snapshot','event','player','stat','side','line','game_start_utc']).to_csv(
        args.paper_dir/'pending_prop_results.csv',index=False)
    print('Snapshots:',len(files),'| Complete strategy-session rows:',int(df['complete'].sum()))
    print(summary.to_string(index=False,float_format=lambda x:f'{x:.3f}'))
    print('Saved:',args.paper_dir/'strategy_paper_summary.csv')
    print('Unresolved player legs:',len(pending),'(not automatically treated as losses)')
    print('WARNING: Profit/ROI is hypothetical using single-leg ask products, not a real Combo fill.')
    print('WARNING: Automatic stat-result inference is NOT official Polymarket settlement verification.')


if __name__=='__main__':
    main()
