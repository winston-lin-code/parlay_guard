"""Parlay Guard NBA/Polymarket retrospective backtest.

Inputs:
  elo_final_holdout_predictions.csv (supplied model holdout)
  epinnock/polymarket-nba data/markets.parquet
  epinnock/polymarket-nba data/prices/prices-2025-26.parquet

Install: pip install duckdb

Run:
  python parlay_guard_polymarket_full_backtest.py \
    --predictions elo_final_holdout_predictions.csv \
    --markets markets.parquet \
    --prices prices-2025-26.parquet

NOTE: This uses sampled historic Polymarket prices, not historic executable asks.
The alternate side is proxied as 1 - observed first-outcome price. Price
adjustments (+1c/+2c) are sensitivity analyses, not measured order-book spreads.
It does NOT reproduce a live v1-v2 GB consensus; GB/Elo are the two columns
in the historical file. The script skips unmatched/stale/unverifiable games.
"""
import argparse, csv
from collections import defaultdict
from pathlib import Path

NICKNAMES = {
 'ATL':'hawks','BOS':'celtics','BKN':'nets','CHA':'hornets','CHI':'bulls',
 'CLE':'cavaliers','DAL':'mavericks','DEN':'nuggets','DET':'pistons',
 'GSW':'warriors','HOU':'rockets','IND':'pacers','LAC':'clippers',
 'LAL':'lakers','MEM':'grizzlies','MIA':'heat','MIL':'bucks',
 'MIN':'timberwolves','NOP':'pelicans','NYK':'knicks','OKC':'thunder',
 'ORL':'magic','PHI':'76ers','PHX':'suns','POR':'trail blazers',
 'SAC':'kings','SAS':'spurs','TOR':'raptors','UTA':'jazz','WAS':'wizards'
}

def first_side(outcome, home, away):
    s=str(outcome or '').strip().lower()
    for abbr in (home, away):
        names={NICKNAMES[abbr],abbr.lower()}
        if any(s == n or s.endswith(' '+n) for n in names):
            return abbr
    return None

def load_quotes(markets_path, prices_path, target_slugs):
    try:
        import duckdb
    except ImportError as e:
        raise SystemExit('Requires duckdb: pip install duckdb') from e
    conn=duckdb.connect(database=':memory:')
    # Use only 2025-26 game moneylines and PRE-game price points.
    # A 30-minute lead before tipoff avoids accidental in-game signals.
    query="""
    WITH selected AS (
      SELECT m.condition_id,
             m.event_slug,
             m.outcomes[1] AS first_outcome,
             CAST(m.game_start_time AS TIMESTAMPTZ) AS tipoff,
             CAST(p.timestamp AS TIMESTAMPTZ) AS observed,
             CAST(p.price AS DOUBLE) AS first_price,
             p.fidelity
      FROM read_parquet(?) m
      JOIN read_parquet(?) p ON m.condition_id=p.condition_id
      WHERE lower(m.market_type) = 'moneyline'
        AND m.season='2025-26'
        AND p.price > 0 AND p.price < 1
        AND CAST(p.timestamp AS TIMESTAMPTZ)
            <= CAST(m.game_start_time AS TIMESTAMPTZ) - INTERVAL '30 minutes'
        AND CAST(p.timestamp AS TIMESTAMPTZ)
            >= CAST(m.game_start_time AS TIMESTAMPTZ) - INTERVAL '3 hours'
    )
    SELECT condition_id,event_slug,first_outcome,tipoff,observed,first_price,fidelity
    FROM selected
    QUALIFY row_number() OVER
        (PARTITION BY condition_id ORDER BY observed DESC,
         CASE WHEN fidelity='minute' THEN 1 ELSE 0 END DESC)=1
    """
    quoted=conn.execute(query, [str(markets_path), str(prices_path)]).fetchall()
    # If the same event has multiple moneyline conditions, do NOT pick one
    # opportunistically after seeing outcomes; mark the event ambiguous.
    events=defaultdict(list)
    for condition, slug, outcome, tipoff, observed, price, fidelity in quoted:
        if slug in target_slugs:
            events[slug].append({'condition_id':condition, 'first_outcome':outcome,
                'tipoff':tipoff, 'observed':observed,'price':price,'fidelity':fidelity})
    return {slug: values[0] for slug, values in events.items() if len(values)==1},\
           {slug:len(values) for slug, values in events.items() if len(values)>1}

def simulate(rows, price_map, ambiguous, stake=10.0):
    strategies=['GB: every pick','Elo: every pick','GB: EV > 0%',
       'Elo: EV > 0%','GB: EV > 5%','Elo: EV > 5%',
       'GB: 5pp cushion, EV > 5%','Elo: 5pp cushion, EV > 5%',
       'Consensus: 5pp cushion, EV > 5%']
    bets=[]; missing=defaultdict(int); coverage=[]
    for r in rows:
        home, away=r['HOME_TEAM'],r['AWAY_TEAM']; date=r['GAME_DATE']
        slug=f'nba-{away.lower()}-{home.lower()}-{date}'
        if slug in ambiguous:
            missing['ambiguous_moneyline_markets']+=1;continue
        q=price_map.get(slug)
        if not q:
            missing['no_fresh_pretipoff_price']+=1;continue
        abbr=first_side(q['first_outcome'],home,away)
        if abbr is None:
            missing['unknown_first_outcome']+=1;continue
        hprice=float(q['price']) if abbr==home else 1-float(q['price'])
        if not (0<hprice<1):
            missing['bad_price']+=1;continue
        actual_home=r['HOME_WIN'].strip().lower() in ('true','1')
        gb=float(r['p_home_win_selected_elo_gb'])
        elo=float(r['p_home_win_elo_formula'])
        sides={'HOME':home,'AWAY':away}
        coverage.append({'game_id':r['GAME_ID'],'date':date,'slug':slug,
         'price_home':hprice,'price_away':1-hprice,
         'price_timestamp_utc':str(q['observed']),'tipoff_utc':str(q['tipoff']),
         'source_condition_id':q['condition_id']})
        for premium in (0.00,0.01,0.02):
            for strat in strategies:
                gb_side='HOME' if gb>=.5 else 'AWAY'
                elo_side='HOME' if elo>=.5 else 'AWAY'
                if strat.startswith('Consensus'):
                    if gb_side!=elo_side: continue
                    side=gb_side
                    prob=min(gb,elo) if side=='HOME' else min(1-gb,1-elo)
                    prob=max(0,prob-.05)
                    margin=.05
                else:
                    is_gb=strat.startswith('GB:')
                    p=gb if is_gb else elo
                    side='HOME' if p>=.5 else 'AWAY'
                    prob=p if side=='HOME' else 1-p
                    if '5pp cushion' in strat: prob=max(0,prob-.05)
                    margin=.05 if 'EV > 5%' in strat else (0.0 if 'EV > 0%' in strat else None)
                reference=hprice if side=='HOME' else 1-hprice
                buy=reference+premium
                if not (0<buy<1): continue
                projected_roi=prob/buy-1
                if margin is not None and projected_roi <= margin: continue
                won=actual_home if side=='HOME' else (not actual_home)
                net=stake*(1/buy-1) if won else -stake
                bets.append({'scenario_buy_premium_cents':int(round(premium*100)),
                    'strategy':strat,'game_id':r['GAME_ID'],'date':date,
                    'event_slug':slug,'selected':sides[side],'model_p_used':prob,
                    'sampled_first_outcome':q['first_outcome'],'sampled_first_price':q['price'],
                    'reference_price':reference,'assumed_buy_price':buy,
                    'projected_roi':projected_roi,'won':int(won),'stake':stake,
                    'net_profit':net,'observed_utc':str(q['observed']),
                    'condition_id':q['condition_id']})
    return bets,coverage,dict(missing),strategies

def write_csv(filename, rows, cols=None):
    if not rows:
        Path(filename).write_text('No results\n');return
    cols=cols or list(rows[0]);
    with open(filename,'w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=cols);w.writeheader();w.writerows(rows)

def reports(bets, strategies, coverage, prefix):
    out=[]; monthly=[]
    for prem in (0,1,2):
        for name in strategies:
            ss=[b for b in bets if b['strategy']==name and b['scenario_buy_premium_cents']==prem]
            profit=sum(x['net_profit'] for x in ss);amount=sum(x['stake'] for x in ss)
            peak=0.;running=0.;max_drawdown=0.
            for row in sorted(ss,key=lambda x:(x['date'],x['game_id'])):
                running+=row['net_profit'];peak=max(peak,running)
                max_drawdown=max(max_drawdown,peak-running)
            out.append({'buy_price_premium_cents':prem,'strategy':name,'n_bets':len(ss),
                'wins':sum(x['won'] for x in ss),'staked':round(amount,2),
                'net_profit':round(profit,2),'roi_pct':round(100*profit/amount,2) if amount else None,
                'max_drawdown_usd':round(max_drawdown,2)})
            months=sorted(set(x['date'][:7] for x in ss))
            for month in months:
                xx=[x for x in ss if x['date'].startswith(month)]
                monthly.append({'buy_price_premium_cents':prem,'strategy':name,
                    'month':month,'bets':len(xx),
                    'net_profit':round(sum(x['net_profit'] for x in xx),2)})
    write_csv(prefix+'_bets.csv',bets)
    write_csv(prefix+'_summary.csv',out)
    write_csv(prefix+'_monthly.csv',monthly)
    write_csv(prefix+'_matched_quotes.csv',coverage)
    return out

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--predictions',type=Path,default=Path('elo_final_holdout_predictions.csv'))
    ap.add_argument('--markets',type=Path,required=True)
    ap.add_argument('--prices',type=Path,required=True)
    ap.add_argument('--output-prefix',default='parlay_guard_polymarket')
    args=ap.parse_args()
    with args.predictions.open(newline='',encoding='utf-8') as f:
        rows=list(csv.DictReader(f))
    slugs={f"nba-{r['AWAY_TEAM'].lower()}-{r['HOME_TEAM'].lower()}-{r['GAME_DATE']}" for r in rows}
    quotes,ambiguous=load_quotes(args.markets,args.prices,slugs)
    bets,coverage,missing,strategies=simulate(rows,quotes,ambiguous)
    summaries=reports(bets,strategies,coverage,args.output_prefix)
    print('Prediction games:',len(rows),'Matched valid markets:',len(coverage),'Unmatched:',missing)
    print('NB: Results use sampled historical price, not proven executable asks.')
    for r in summaries:
        if r['buy_price_premium_cents']==0:print(r)

if __name__=='__main__':main()
