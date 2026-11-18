"""Parlay Guard dual-model dashboard with conservative value screening."""
import argparse
import html
import json
from pathlib import Path
import pandas as pd

def esc(x): return html.escape(str(x), quote=True)
def num(x): return pd.to_numeric(x, errors='coerce')
def pct(x): return 'N/A' if pd.isna(x) else f'{x:.1%}'
def signed(x): return 'N/A' if pd.isna(x) else f'{x:+.1%}'

# NBA team IDs are stable identifiers; CDN logo images require an internet connection.
TEAMS = {
 'ATL':('Atlanta Hawks',1610612737),'BOS':('Boston Celtics',1610612738),
 'BKN':('Brooklyn Nets',1610612751),'CHA':('Charlotte Hornets',1610612766),
 'CHI':('Chicago Bulls',1610612741),'CLE':('Cleveland Cavaliers',1610612739),
 'DAL':('Dallas Mavericks',1610612742),'DEN':('Denver Nuggets',1610612743),
 'DET':('Detroit Pistons',1610612765),'GSW':('Golden State Warriors',1610612744),
 'HOU':('Houston Rockets',1610612745),'IND':('Indiana Pacers',1610612754),
 'LAC':('LA Clippers',1610612746),'LAL':('Los Angeles Lakers',1610612747),
 'MEM':('Memphis Grizzlies',1610612763),'MIA':('Miami Heat',1610612748),
 'MIL':('Milwaukee Bucks',1610612749),'MIN':('Minnesota Timberwolves',1610612750),
 'NOP':('New Orleans Pelicans',1610612740),'NYK':('New York Knicks',1610612752),
 'OKC':('Oklahoma City Thunder',1610612760),'ORL':('Orlando Magic',1610612753),
 'PHI':('Philadelphia 76ers',1610612755),'PHX':('Phoenix Suns',1610612756),
 'POR':('Portland Trail Blazers',1610612757),'SAC':('Sacramento Kings',1610612758),
 'SAS':('San Antonio Spurs',1610612759),'TOR':('Toronto Raptors',1610612761),
 'UTA':('Utah Jazz',1610612762),'WAS':('Washington Wizards',1610612764),
}

def team_name(abbr):
    return TEAMS.get(str(abbr).upper(), (str(abbr), None))[0]

def team_logo(abbr, size=42):
    abbr=str(abbr).upper()
    _, team_id=TEAMS.get(abbr,(abbr,None))
    if team_id is None:
        return f'<span class="logo-fallback">{esc(abbr)}</span>'
    url=f'https://cdn.nba.com/logos/nba/{team_id}/primary/L/logo.svg'
    return (f'<span class="logo-wrap"><img class="team-logo" width="{size}" height="{size}" '
            f'src="{esc(url)}" alt="{esc(team_name(abbr))} logo" loading="lazy" '
            f'onerror="this.style.display=\'none\';this.nextElementSibling.style.display=\'inline-flex\'">'
            f'<span class="logo-fallback" style="display:none">{esc(abbr)}</span></span>')

def team_identity(abbr):
    return (f'<span class="team-identity">{team_logo(abbr)}'
            f'<span><strong>{esc(team_name(abbr))}</strong><small>{esc(abbr)}</small></span></span>')

def forecasts(path):
    df = pd.read_csv(path, dtype={'GAME_ID': str})
    req = {'AWAY_TEAM','HOME_TEAM','P_HOME_WIN_GB'}
    if not req.issubset(df): raise ValueError(f'{path}: missing {req-set(df)}')
    if df.duplicated(['AWAY_TEAM','HOME_TEAM']).any(): raise ValueError(f'{path}: duplicate matchup')
    return {(r.AWAY_TEAM,r.HOME_TEAM):r for _,r in df.iterrows()}

def factors(a,b):
    fields=[('Last 5 win %','WIN_PCT_LAST_5',True),('Last 10 win %','WIN_PCT_LAST_10',True),('Days rest','DAYS_REST',False),('Last 10 point differential','POINT_DIFF_LAST_10',False)]
    rows=[]
    for label,key,is_pct in fields:
        cells=[]
        for r in (a,b):
            for side in ('AWAY','HOME'):
                x=num(r.get(f'{side}_{key}')) if r is not None else float('nan')
                cells.append(pct(x) if is_pct else ('N/A' if pd.isna(x) else f'{x:+.1f}'))
        rows.append('<tr><td>'+esc(label)+'</td>'+''.join('<td>'+esc(v)+'</td>' for v in cells)+'</tr>')
    return '<table><thead><tr><th>Pregame input</th><th>v1 away</th><th>v1 home</th><th>v2 away</th><th>v2 home</th></tr></thead><tbody>'+''.join(rows)+'</tbody></table>'



def load_season_records(path, forecast_date):
    # Build team results from complete two-row regular-season game logs.
    df = pd.read_csv(path, dtype={'GAME_ID': str, 'TEAM_ID': str})
    needed = {'GAME_ID','GAME_DATE','TEAM_ABBREVIATION','MATCHUP','WL','PTS'}
    if not needed.issubset(df):
        raise ValueError(f'Season CSV missing columns: {sorted(needed-set(df))}')
    df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'], errors='raise')
    df = df.loc[df['GAME_DATE'] < pd.Timestamp(forecast_date)].copy()
    df['TEAM_ABBREVIATION'] = df['TEAM_ABBREVIATION'].astype(str).str.upper().str.strip()
    df['PTS'] = pd.to_numeric(df['PTS'], errors='coerce')
    df['WL'] = df['WL'].astype(str).str.upper().str.strip()
    df['GAME_ID'] = df['GAME_ID'].str.zfill(10)
    # Restrict to NBA regular season, not preseason or cup-only nonleague games.
    df = df[df['GAME_ID'].str.startswith('002')].copy()
    df = df.drop_duplicates(['GAME_ID','TEAM_ABBREVIATION'])
    results = {}
    bad = []
    for game_id, group in df.groupby('GAME_ID'):
        if len(group) != 2 or group['TEAM_ABBREVIATION'].nunique() != 2 or group['PTS'].isna().any() or not set(group['WL']).issubset({'W','L'}) or sorted(group['WL'].tolist()) != ['L','W']:
            bad.append(game_id)
            continue
        a, b = group.iloc[0], group.iloc[1]
        # Matchup can be repeated as AWAY @ HOME on both rows.
        matchup = str(a['MATCHUP'])
        if ' @ ' in matchup:
            away, home = [x.strip().upper() for x in matchup.split(' @ ', 1)]
        elif ' vs. ' in matchup:
            home, away = [x.strip().upper() for x in matchup.split(' vs. ', 1)]
        else:
            matchup = str(b['MATCHUP'])
            if ' @ ' in matchup:
                away, home = [x.strip().upper() for x in matchup.split(' @ ', 1)]
            elif ' vs. ' in matchup:
                home, away = [x.strip().upper() for x in matchup.split(' vs. ', 1)]
            else:
                bad.append(game_id)
                continue
        if {away, home} != set(group['TEAM_ABBREVIATION']):
            bad.append(game_id)
            continue
        if (a['PTS'] > b['PTS']) != (a['WL'] == 'W') or a['PTS'] == b['PTS']:
            bad.append(game_id)
            continue
        for r, opponent in ((a,b),(b,a)):
            team = r['TEAM_ABBREVIATION']
            results.setdefault(team, []).append({
                'date':r['GAME_DATE'], 'id':game_id, 'opponent':opponent['TEAM_ABBREVIATION'],
                'venue':'Home' if team == home else 'Away', 'win':r['WL']=='W',
                'pts':int(r['PTS']), 'opp_pts':int(opponent['PTS']),
            })
    if bad:
        raise ValueError(f'Invalid current-season team logs: {len(bad)} game(s); first IDs: {bad[:5]}')
    for rows in results.values():
        rows.sort(key=lambda x:(x['date'],x['id']), reverse=True)
    return results


def team_record_html(team, rows):
    if not rows:
        return f'<div class="teamrecord"><h4>{team_identity(team)}</h4><p>Season results unavailable in the supplied CSV.</p></div>'
    wins=sum(r['win'] for r in rows)
    losses=len(rows)-wins
    home=[r for r in rows if r['venue']=='Home']
    away=[r for r in rows if r['venue']=='Away']
    rec=lambda rs:f"{sum(r['win'] for r in rs)}-{sum(not r['win'] for r in rs)}"
    last=rows[:10]
    sequence=' '.join('<span class="win">W</span>' if r['win'] else '<span class="loss">L</span>' for r in last)
    table=[]
    for r in last:
        margin=r['pts']-r['opp_pts']
        table.append('<tr><td>'+esc(r['date'].strftime('%b %d'))+'</td><td>'+esc(r['venue'])+' vs '+esc(team_name(r['opponent']))+'</td><td class="'+('win' if r['win'] else 'loss')+'">'+('W' if r['win'] else 'L')+'</td><td>'+f"{r['pts']}-{r['opp_pts']}"+'</td><td>'+f'{margin:+d}'+'</td></tr>')
    return f'''<div class="teamrecord"><h4>{team_identity(team)}</h4><div class="recordgrid"><div><small>Season record</small><b>{wins}-{losses} ({wins/len(rows):.1%})</b></div><div><small>Home / Away</small><b>{rec(home)} / {rec(away)}</b></div><div><small>Last {len(last)}</small><b>{rec(last)}</b></div><div><small>Points for / against</small><b>{sum(r['pts'] for r in rows)/len(rows):.1f} / {sum(r['opp_pts'] for r in rows)/len(rows):.1f}</b></div></div><p class="sequence">Recent results (newest first): {sequence}</p><div class="scroll"><table><thead><tr><th>Date</th><th>Opponent</th><th>Result</th><th>Score</th><th>Margin</th></tr></thead><tbody>{''.join(table)}</tbody></table></div></div>'''

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for arg in ('ev','v1','v2'): ap.add_argument('--'+arg,required=True,type=Path)
    ap.add_argument('--output',type=Path,default=Path('models/parlay_guard_v1_vs_v2.html'))
    ap.add_argument('--season-csv',type=Path,required=True,help='2026-27 raw team-game logs')
    ap.add_argument('--forecast-date',default=None,help='YYYY-MM-DD; defaults to forecast CSV date')
    ap.add_argument('--min-roi',type=float,default=.10,help='Minimum estimated ROI for both models (default 0.10)')
    ap.add_argument('--min-edge',type=float,default=.05,help='Minimum probability minus ask price for both models (default 0.05)')
    ap.add_argument('--max-disagreement',type=float,default=.10,help='Maximum absolute probability difference between models (default 0.10)')
    args=ap.parse_args()
    ev=pd.read_csv(args.ev)
    required={'away','home','team','best_ask'}
    if not required.issubset(ev): raise ValueError(f'EV CSV missing {required-set(ev)}')
    ev['best_ask']=num(ev['best_ask'])
    v1,v2=forecasts(args.v1),forecasts(args.v2)
    forecast_dates=[]
    for path in (args.v1,args.v2):
        header=pd.read_csv(path,nrows=1)
        for key in ('GAME_DATE','game_date','FORECAST_DATE','forecast_date'):
            if key in header and len(header): forecast_dates.append(str(pd.Timestamp(header.iloc[0][key]).date()))
    if args.forecast_date:
        forecast_date=args.forecast_date
    elif forecast_dates and len(set(forecast_dates))==1:
        forecast_date=forecast_dates[0]
    else:
        raise ValueError('Pass --forecast-date YYYY-MM-DD to ensure season stats exclude game day')
    season_records=load_season_records(args.season_csv,forecast_date)
    items=[]; matched=0
    for (away,home),group in ev.groupby(['away','home'],sort=True):
        a,b=v1.get((away,home)),v2.get((away,home))
        if a is not None and b is not None: matched+=1
        p1=num(a.get('P_HOME_WIN_GB')) if a is not None else float('nan')
        p2=num(b.get('P_HOME_WIN_GB')) if b is not None else float('nan')
        for name,p in [('v1',p1),('v2',p2)]:
            if pd.notna(p) and not 0<=p<=1: raise ValueError(f'{name} invalid probability: {away} @ {home}')
        options=[]
        for _,r in group.iterrows():
            team=str(r['team']); price=r['best_ask']
            if team not in (home,away) or pd.isna(price) or not 0<price<1: continue
            q1=p1 if team==home else 1-p1
            q2=p2 if team==home else 1-p2
            roi1=q1/price-1 if pd.notna(q1) else float('nan')
            roi2=q2/price-1 if pd.notna(q2) else float('nan')
            edge1=q1-price if pd.notna(q1) else float('nan')
            edge2=q2-price if pd.notna(q2) else float('nan')
            valid=all(pd.notna(x) for x in (q1,q2,roi1,roi2))
            agreement=valid and q1>=.5 and q2>=.5
            disagreement=abs(q1-q2) if valid else float('nan')
            qualifies=(valid and agreement and min(roi1,roi2)>=args.min_roi and min(edge1,edge2)>=args.min_edge and disagreement<=args.max_disagreement)
            positive=valid and roi1>0 and roi2>0
            if qualifies: status='HIGH MODEL-ESTIMATED VALUE'; rank=0
            elif positive: status='POSITIVE IN BOTH · REVIEW'; rank=1
            elif valid and (roi1>0 or roi2>0): status='MODELS DISAGREE ON VALUE'; rank=2
            else: status='NO CLEAR VALUE SIGNAL'; rank=3
            options.append(dict(team=team,price=price,q1=q1,q2=q2,roi1=roi1,roi2=roi2,edge1=edge1,edge2=edge2,status=status,rank=rank,qualifies=qualifies,disagreement=disagreement))
        if not options: continue
        options.sort(key=lambda x:(x['rank'],-min(x['roi1'],x['roi2']) if pd.notna(x['roi1']) and pd.notna(x['roi2']) else 1e6))
        top=options[0]
        items.append(dict(away=away,home=home,a=a,b=b,p1=p1,p2=p2,options=options,top=top))
    items.sort(key=lambda x:(x['top']['rank'],-min(x['top']['roi1'],x['top']['roi2']) if pd.notna(x['top']['roi1']) and pd.notna(x['top']['roi2']) else 1e6))
    cards=[]
    for item in items:
        away,home=item['away'],item['home']; top=item['top']
        opts=[]
        for o in item['options']:
            cls='strong' if o['qualifies'] else ('review' if o['rank']<=2 else 'muted')
            opts.append(f'<tr><td>{team_identity(o["team"])}</td><td>${o["price"]:.3f}</td><td>{pct(o["q1"])}</td><td>{pct(o["q2"])}</td><td class="{cls}">{signed(o["roi1"])}</td><td class="{cls}">{signed(o["roi2"])}</td><td>{signed(o["edge1"])}</td><td>{signed(o["edge2"])}</td><td>{esc(o["status"])}</td></tr>')
        pick=lambda p: ('N/A' if pd.isna(p) else f'{team_name(home)} {pct(p)}' if p>=.5 else f'{team_name(away)} {pct(1-p)}')
        label=top['status']; cl='strong' if top['qualifies'] else ('review' if top['rank']<=2 else 'muted')
        cards.append(f'''<details class="game {cl}"><summary><span class="match">{team_identity(away)}<span class="versus">@</span>{team_identity(home)}</span><span class="pill {cl}">{esc(label)}</span><span>{esc(top['team'])} @ ${top['price']:.3f}</span><span>v1 {signed(top['roi1'])} · v2 {signed(top['roi2'])}</span></summary><section><div class="stats"><div><small>v1 predicted winner</small><b>{esc(pick(item['p1']))}</b></div><div><small>v2 predicted winner</small><b>{esc(pick(item['p2']))}</b></div><div><small>v1/v2 difference (home)</small><b>{signed(item['p2']-item['p1'])}</b></div></div><h3>All available wager sides</h3><div class="scroll"><table><thead><tr><th>Wager</th><th>Ask</th><th>v1 win %</th><th>v2 win %</th><th>v1 ROI</th><th>v2 ROI</th><th>v1 edge</th><th>v2 edge</th><th>Screening result</th></tr></thead><tbody>{''.join(opts)}</tbody></table></div><h3>2026–27 season record and previous 10 games</h3><p class="note">Completed regular-season games before {esc(forecast_date)}. Most recent first.</p><div class="teamgrid">{team_record_html(away,season_records.get(away,[]))}{team_record_html(home,season_records.get(home,[]))}</div><h3>Pregame inputs</h3><div class="scroll">{factors(item['a'],item['b'])}</div><p class="note">Inputs are not causal feature attributions. EV assumes a winning YES share pays $1; ROI = probability / best ask − 1. Excludes fees, slippage, execution and settlement risks.</p></section></details>''')
    # One eligible wager per matchup. Use the lower model probability, with a
    # 5-percentage-point haircut to reflect unvalidated calibration uncertainty.
    # Only include strictly positive adjusted edge; never force allocation.
    candidates=[]
    for item in items:
        eligible=[o for o in item['options'] if o['qualifies']]
        if not eligible:
            continue
        eligible.sort(key=lambda o:min(o['roi1'],o['roi2']),reverse=True)
        o=eligible[0]
        adjusted=max(0.0,min(o['q1'],o['q2'])-0.05)
        if adjusted<=o['price']:
            continue
        candidates.append({
            'matchup':f"{team_name(item['away'])} @ {team_name(item['home'])}",
            'team':team_name(o['team']), 'abbr':o['team'],
            'price':round(float(o['price']),6),
            'p1':round(float(o['q1']),6),'p2':round(float(o['q2']),6),
            'adjusted':round(adjusted,6),
            'min_roi':round(float(min(o['roi1'],o['roi2'])),6)
        })
    # Escape closing script tags in serialized data, even if the CSV is untrusted.
    candidate_json=json.dumps(candidates,ensure_ascii=True).replace('<','\\u003c')
    strong=sum(x['top']['qualifies'] for x in items)
    date=esc(str(ev['game_date'].iloc[0])) if 'game_date' in ev and len(ev) else 'NBA'
    style='''body{background:#0c1220;color:#e8eefb;font:15px system-ui,sans-serif;max-width:1200px;margin:35px auto;padding:0 18px}h1{margin-bottom:4px}p,.note{color:#a8b7cc;line-height:1.5}.summary{display:flex;gap:12px;flex-wrap:wrap;margin:22px 0}.metric{background:#182338;border:1px solid #33445d;border-radius:12px;padding:16px;min-width:180px}.metric b{display:block;font-size:25px}.metric small,.stats small{display:block;color:#a8b7cc}.game{background:#182338;border:1px solid #34435b;border-radius:12px;margin:10px 0;overflow:hidden}.game.strong{border-left:5px solid #62d99a}.game.review{border-left:5px solid #e3b65d}summary{cursor:pointer;display:grid;grid-template-columns:1fr 1.4fr .9fr 1.2fr;gap:12px;align-items:center;padding:18px}summary:hover{background:#24314a}.match{font-weight:750}.pill{font-size:11px;font-weight:700;letter-spacing:.025em;padding:7px 9px;border-radius:6px;display:inline-block;width:fit-content}.pill.strong{background:#184d37;color:#8df0b4}.pill.review{background:#564221;color:#ffdb8b}.pill.muted{background:#2c374a;color:#b7c3d5}section{padding:12px 18px 22px;border-top:1px solid #34435b}.stats{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}.stats>div{background:#101a2c;padding:12px;border-radius:9px}.stats b{display:block;margin-top:5px}.scroll{overflow-x:auto}table{width:100%;border-collapse:collapse;margin:12px 0}th,td{text-align:left;padding:10px;border-bottom:1px solid #34435b;white-space:nowrap}th{font-size:12px;color:#a8b7cc}.strong{color:#86e7ae}.review{color:#f2c77b}.muted{color:#b5bfd0}.note{font-size:12px}.warning{padding:14px;border:1px solid #66533b;border-radius:9px;background:#2a241e}@media(max-width:850px){summary{grid-template-columns:1fr 1fr}.stats{grid-template-columns:1fr}}'''
    style+='''.teamgrid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:14px}.teamrecord{background:#101a2c;border:1px solid #34435b;border-radius:10px;padding:14px;min-width:0}.teamrecord h4{margin:0 0 12px}.recordgrid{display:grid;grid-template-columns:1fr 1fr;gap:10px}.recordgrid small{display:block;color:#a8b7cc}.recordgrid b{display:block;margin-top:3px}.sequence{font-size:13px}.win{color:#86e7ae;font-weight:700}.loss{color:#f0a4a4;font-weight:700}@media(max-width:850px){.teamgrid{grid-template-columns:1fr}}'''
    style += '.team-identity{display:inline-flex;align-items:center;gap:9px}.team-identity strong{display:block}.team-identity small{display:block;color:#a8b7cc;font-size:11px}.team-logo,.logo-wrap,.logo-fallback{width:42px;height:42px}.team-logo{object-fit:contain}.logo-wrap{display:inline-flex;align-items:center;justify-content:center}.logo-fallback{border-radius:50%;background:#34435b;color:white;align-items:center;justify-content:center;font-size:11px}.match{display:flex;align-items:center;gap:8px;flex-wrap:wrap}.versus{color:#a8b7cc}'
    style += '.allocator{background:#182338;border:1px solid #40516a;border-radius:14px;padding:20px;margin:22px 0}.allocator-controls{display:flex;gap:18px;flex-wrap:wrap;align-items:end}.allocator label{display:grid;gap:6px;color:#c6d4e8;font-size:13px}.allocator input,.allocator select{background:#0c1220;color:#e8eefb;border:1px solid #62728b;border-radius:7px;padding:10px;font-size:16px;min-width:130px}.allocator .alloc-metrics{display:flex;gap:12px;flex-wrap:wrap;margin:14px 0}.allocator .alloc-metrics>div{background:#101a2c;border-radius:8px;padding:12px;min-width:150px}.allocator .alloc-metrics strong{display:block;font-size:22px;margin-top:4px}.allocator .alloc-note{color:#a8b7cc;font-size:12px;line-height:1.5}.allocator .empty{padding:12px;background:#101a2c;border-radius:8px}#alloc-table td,#alloc-table th{white-space:normal}'
    allocator_html='''<section class="allocator" id="bankroll-allocator">
      <h2>Bankroll Allocator <small style="font-size:12px;color:#a8b7cc">Research simulation</small></h2>
      <p>Enter a budget to simulate how much of it would be allocated to qualifying wagers. The remaining balance stays unallocated.</p>
      <div class="allocator-controls">
        <label>Budget ($)<input id="alloc-budget" type="number" min="0" max="1000000000" step="0.01" value="100" inputmode="decimal"></label>
        <label>Risk preset<select id="alloc-risk"><option value="cautious">Cautious (5% total cap)</option><option value="balanced">Higher exposure (10% total cap)</option><option value="full">Full budget (100% simulation)</option></select></label>
      </div>
      <div class="alloc-metrics"><div><small>Allocated</small><strong id="alloc-total">$0.00</strong></div><div><small>Unallocated</small><strong id="alloc-left">$100.00</strong></div><div><small>Eligible wagers</small><strong id="alloc-count">0</strong></div><div><small>Expected ending balance*</small><strong id="alloc-expected">$100.00</strong></div><div><small>Expected net profit/loss*</small><strong id="alloc-profit">$0.00</strong></div><div><small>All wagers lose</small><strong id="alloc-worst">$100.00</strong></div><div><small>All wagers win</small><strong id="alloc-best">$100.00</strong></div></div>
      <div id="alloc-message" class="alloc-note" aria-live="polite"></div>
      <div class="scroll"><table id="alloc-table"><thead><tr><th>Matchup</th><th>Wager</th><th>Ask</th><th>v1 / v2 probability</th><th>Conservative adjusted probability</th><th>Allocated</th><th>Budget %</th><th>Expected payout*</th><th>Expected net*</th></tr></thead><tbody id="alloc-rows"></tbody></table></div>
      <p class="alloc-note">*Projected figures are mathematical estimates using the lower model probability minus 5 percentage points, not a guaranteed or calibrated forecast. Expected ending balance is the probability-weighted average across outcomes, including unallocated cash. “All win” and “all lose” are scenarios, not forecasts. Full-budget mode deliberately overrides stake limits and can lose the entire budget; it is not recommended. All calculations assume $1 payout per winning YES share and ignore fees, slippage and market liquidity.</p><p class="alloc-note">Rules: one wager per game; both models must pass the existing value screen. Adjusted probability = lower model estimate minus 5 percentage points. Fractional Kelly uses 1/8 Kelly (cautious) or 1/4 Kelly (higher exposure), with a 2% per-game cap and 5%/10% total cap. Fractions are heuristics, not statistically calibrated. Allocations are illustrative, not a recommendation to place bets. No fees, slippage, liquidity or correlated-outcome adjustments are included. Quotes are historical snapshots.</p>
    </section>'''
    allocator_js='''<script>
    const candidates = __CANDIDATES__;
    const money = x => '$' + x.toLocaleString('en-US',{minimumFractionDigits:2,maximumFractionDigits:2});
    const percent = x => (100*x).toFixed(1) + '%';
    const budgetInput = document.getElementById('alloc-budget');
    const riskInput = document.getElementById('alloc-risk');
    function renderAllocation() {
      const raw = Number(budgetInput.value);
      const valid = budgetInput.value.trim() !== '' && Number.isFinite(raw) && raw >= 0 && raw <= 1e9;
      const budget = valid ? raw : 0;
      const full = riskInput.value === 'full';
      const balanced = riskInput.value === 'balanced';
      const totalCap = full ? 1 : balanced ? 0.10 : 0.05;
      const fraction = balanced ? 0.25 : 0.125;
      const rows = candidates.map(c => {
        // For a YES share costing price, full Kelly f=(p-price)/(1-price).
        const fullKelly = Math.max(0, (c.adjusted-c.price)/(1-c.price));
        return {...c, weight:fullKelly, share:Math.min(0.02, fullKelly*fraction)};
      }).filter(c=>c.weight>0).sort((a,b)=>b.weight-a.weight || b.min_roi-a.min_roi);
      const sum = rows.reduce((t,c)=>t+(full?c.weight:c.share),0);
      const scale = full ? (sum>0 ? 1/sum : 0) : (sum>totalCap ? totalCap/sum : 1);
      // Allocate cents deterministically; full-budget mode distributes leftover cents
      // to the largest fractional remainders so the total equals the entered budget.
      const budgetCents=Math.round(budget*100);
      const targetCents=full && rows.length ? budgetCents : null;
      const shares=rows.map(c=>full?c.weight*scale:c.share*scale);
      const rawCents=shares.map(share=>budgetCents*share);
      const centsList=rawCents.map(x=>Math.floor(x+1e-9));
      if(full && rows.length){
        let remaining=targetCents-centsList.reduce((a,b)=>a+b,0);
        const order=rows.map((_,i)=>i).sort((a,b)=>(rawCents[b]-Math.floor(rawCents[b]))-(rawCents[a]-Math.floor(rawCents[a])));
        for(let i=0;i<remaining;i++) centsList[order[i%order.length]]++;
      }
      let allocatedCents = 0, expectedPayout = 0, bestPayout = 0;
      const body = document.getElementById('alloc-rows');
      body.replaceChildren();
      rows.forEach((c,i)=>{
        const cents=centsList[i];
        if(cents<=0) return;
        allocatedCents += cents;
        const stake=cents/100;
        const payout=stake/c.price;
        const expected=payout*c.adjusted;
        expectedPayout+=expected;
        bestPayout+=payout;
        const tr=document.createElement('tr');
        const vals=[c.matchup,c.team,'$'+c.price.toFixed(3),percent(c.p1)+' / '+percent(c.p2),percent(c.adjusted),money(stake),percent(stake/budget),money(expected),money(expected-stake)];
        vals.forEach(v=>{const td=document.createElement('td');td.textContent=v;tr.appendChild(td)});
        body.appendChild(tr);
      });
      const allocated=allocatedCents/100;
      const unallocated=Math.max(0,budget-allocated);
      const expectedEnding=unallocated+expectedPayout;
      const net=expectedEnding-budget;
      document.getElementById('alloc-total').textContent=money(allocated);
      document.getElementById('alloc-left').textContent=money(unallocated);
      document.getElementById('alloc-count').textContent=String(body.children.length);
      document.getElementById('alloc-expected').textContent=money(expectedEnding);
      document.getElementById('alloc-profit').textContent=(net<0?'-':'')+money(Math.abs(net));
      document.getElementById('alloc-worst').textContent=money(unallocated);
      document.getElementById('alloc-best').textContent=money(unallocated+bestPayout);
      document.getElementById('alloc-message').textContent=!valid ? 'Enter a valid, nonnegative budget (up to $1 billion).' : !rows.length ? 'No wagers pass both the value screen and the extra uncertainty adjustment. No allocation is suggested, including in full-budget mode.' : budget===0 ? 'Enter a positive budget to see dollar allocations.' : full ? 'FULL BUDGET SCENARIO: all available funds are at risk. This overrides the cautious allocation limits and is not recommended.' : 'These are model-based hypothetical amounts, not verified profitable opportunities.';
    }
    budgetInput.addEventListener('input',renderAllocation);
    riskInput.addEventListener('change',renderAllocation);
    renderAllocation();
    </script>'''.replace('__CANDIDATES__',candidate_json)
    page=f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Parlay Guard — Value Screen</title><style>{style}</style></head><body><h1>Parlay Guard · v1 vs v2</h1><p>{date} · Saved market-price snapshot · Ranked by agreement and conservative estimated ROI.</p><div class="summary"><div class="metric"><small>Matchups</small><b>{len(items)}</b></div><div class="metric"><small>Both models available</small><b>{matched}</b></div><div class="metric"><small>High model-estimated value screens</small><b>{strong}</b></div></div><p class="warning"><strong>Important:</strong> “High model-estimated value” is a screening label, not a recommendation to bet or proof of positive expected returns. v2 is not independently validated or calibrated. Large estimated edges may reflect model error, stale prices, missing injuries, low liquidity, or data issues. Verify current prices and risk before any decision.</p><p><strong>Screening rules:</strong> both models favor the selected team (win probability ≥50%), each estimates ROI ≥{args.min_roi:.0%}, each estimates probability edge ≥{args.min_edge:.0%}, and their probabilities differ by ≤{args.max_disagreement:.0%}. These thresholds are heuristics, not statistically validated confidence levels.</p><p>Click a matchup to see both sides, both model estimates, and pregame factors.</p>{allocator_html}{''.join(cards)}{allocator_js}<p class="note">This report uses saved historical odds, not live executable quotes. For research and model evaluation only.</p></body></html>'''
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(page,encoding='utf-8')
    print(f'Saved {args.output.resolve()} | {strong} high-value screens | {matched}/{len(items)} matchups matched')

if __name__=='__main__': main()
