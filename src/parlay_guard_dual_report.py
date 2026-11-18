"""Side-by-side v1/v2 NBA forecasts with snapshot Polymarket EV.
Run: python -m src.parlay_guard_dual_report --ev models/polymarket_ev_....csv --v1 models/live_predictions_....csv --v2 models/live_predictions_v2_....csv
"""
import argparse, html
from pathlib import Path
import pandas as pd

def e(x): return html.escape(str(x),quote=True)
def pc(x): return 'N/A' if pd.isna(x) else f'{float(x):.1%}'
def signed(x): return 'N/A' if pd.isna(x) else f'{float(x):+.1%}'
def n(x): return pd.to_numeric(x,errors='coerce')

def forecasts(path):
    df=pd.read_csv(path,dtype={'GAME_ID':str})
    required={'AWAY_TEAM','HOME_TEAM','P_HOME_WIN_GB'}
    if not required.issubset(df): raise ValueError(f'{path}: missing {required-set(df)}')
    if df.duplicated(['AWAY_TEAM','HOME_TEAM']).any(): raise ValueError(f'{path}: duplicate matchup')
    return {(r.AWAY_TEAM,r.HOME_TEAM):r for _,r in df.iterrows()}

def factor_table(a,b):
    fields=[('Last 5 win %','WIN_PCT_LAST_5','pct'),('Last 10 win %','WIN_PCT_LAST_10','pct'),('Days rest','DAYS_REST','number'),('Last 10 point diff','POINT_DIFF_LAST_10','number')]
    lines=[]
    for label,feature,fmt in fields:
        values=[]
        for r in (a,b):
            for side in ('AWAY','HOME'):
                x=n(r.get(f'{side}_{feature}')) if r is not None else float('nan')
                values.append(pc(x) if fmt=='pct' else ('N/A' if pd.isna(x) else f'{x:+.1f}'))
        lines.append('<tr><td>'+e(label)+'</td>'+''.join('<td>'+e(x)+'</td>' for x in values)+'</tr>')
    return '<table><thead><tr><th>Pregame feature</th><th>v1 away</th><th>v1 home</th><th>v2 away</th><th>v2 home</th></tr></thead><tbody>'+''.join(lines)+'</tbody></table>'

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for arg in ('ev','v1','v2'): p.add_argument('--'+arg,required=True,type=Path)
    p.add_argument('--output',type=Path,default=Path('models/parlay_guard_v1_vs_v2.html'))
    args=p.parse_args()
    ev=pd.read_csv(args.ev)
    required={'away','home','team','best_ask','model_win_prob'}
    if not required.issubset(ev): raise ValueError(f'EV CSV missing {required-set(ev)}')
    v1,v2=forecasts(args.v1),forecasts(args.v2)
    for c in ('best_ask','model_win_prob'): ev[c]=pd.to_numeric(ev[c],errors='coerce')
    rows=[]; matched=0
    for (away,home),g in ev.groupby(['away','home'],sort=True):
        key=(away,home); a=v1.get(key); b=v2.get(key)
        if a is None or b is None: print(f'WARNING: {away} @ {home}: missing '+('v1 ' if a is None else '')+('v2' if b is None else ''))
        if a is not None and b is not None: matched+=1
        p1=n(a.get('P_HOME_WIN_GB')) if a is not None else float('nan')
        p2=n(b.get('P_HOME_WIN_GB')) if b is not None else float('nan')
        if pd.notna(p1) and not 0<=p1<=1: raise ValueError(f'Invalid v1 probability {key}')
        if pd.notna(p2) and not 0<=p2<=1: raise ValueError(f'Invalid v2 probability {key}')
        def pick(prob): return ('N/A','N/A') if pd.isna(prob) else ((home,pc(prob)) if prob>=.5 else (away,pc(1-prob)))
        pick1,pick2=pick(p1),pick(p2)
        detail=[]
        for _,r in g.iterrows():
            price=r.best_ask; team=r.team
            if pd.notna(price) and not 0<price<1: continue
            prob1=p1 if team==home else 1-p1 if team==away else float('nan')
            prob2=p2 if team==home else 1-p2 if team==away else float('nan')
            roi1=prob1/price-1 if pd.notna(prob1) and pd.notna(price) else float('nan')
            roi2=prob2/price-1 if pd.notna(prob2) and pd.notna(price) else float('nan')
            detail.append(f'<tr><td>{e(team)}</td><td>${price:.3f}</td><td>{pc(prob1)}</td><td class="{("pos" if roi1>0 else "neg")}">{signed(roi1)}</td><td>{pc(prob2)}</td><td class="{("pos" if roi2>0 else "neg")}">{signed(roi2)}</td></tr>')
        best=g.sort_values('model_win_prob',ascending=False).iloc[0]
        delta=p2-p1 if pd.notna(p1) and pd.notna(p2) else float('nan')
        idx=len(rows)
        rows.append(f'''<details><summary><span class="match">{e(away)} @ {e(home)}</span><span>{e(pick1[0])} {e(pick1[1])}</span><span>{e(pick2[0])} {e(pick2[1])}</span><span>{signed(delta)} home</span></summary><section><h3>Model probabilities</h3><p>Home ({e(home)}): v1 {pc(p1)} · v2 {pc(p2)}. Away ({e(away)}): v1 {pc(1-p1)} · v2 {pc(1-p2)}.</p><h3>Pregame model inputs</h3>{factor_table(a,b)}<p class="note">These are input values, not causal feature attributions.</p><h3>Polymarket snapshot — both models</h3><table><thead><tr><th>Side</th><th>Best ask</th><th>v1 probability</th><th>v1 est. ROI</th><th>v2 probability</th><th>v2 est. ROI</th></tr></thead><tbody>{''.join(detail)}</tbody></table><p class="note">ROI = model probability / ask price − 1, before fees, slippage, and liquidity effects. v2 is not yet independently validated. Prices are historical snapshots.</p></section></details>''')
    date=e(str(ev['game_date'].iloc[0])) if 'game_date' in ev and len(ev) else 'NBA'
    page=f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Parlay Guard — v1 vs v2</title><style>body{{background:#0d1320;color:#edf2fc;font:15px system-ui,sans-serif;max-width:1150px;margin:35px auto;padding:0 20px}}h1{{margin-bottom:5px}}p,.note{{color:#aab8cf;line-height:1.55}}.head,summary{{display:grid;grid-template-columns:1.3fr 1fr 1fr 1fr;gap:12px;align-items:center}}.head{{color:#aab8cf;font-size:12px;text-transform:uppercase;padding:16px}}details{{background:#182338;border:1px solid #34435b;border-radius:10px;margin:9px 0;overflow:hidden}}summary{{cursor:pointer;padding:18px 16px}}summary:hover{{background:#24314a}}.match{{font-weight:700}}section{{padding:5px 18px 24px;border-top:1px solid #34435b;overflow-x:auto}}table{{border-collapse:collapse;width:100%;margin:12px 0}}th,td{{text-align:left;border-bottom:1px solid #34435b;padding:11px;white-space:nowrap}}th{{color:#aab8cf;font-size:12px}}.pos{{color:#78e1a5}}.neg{{color:#ffaaaa}}.note{{font-size:12px}}@media(max-width:650px){{.head,summary{{grid-template-columns:1fr 1fr;font-size:13px}}}}</style></head><body><h1>Parlay Guard · v1 vs v2</h1><p>{date} · {len(rows)} matchups · {matched} matched in both forecast files. Click any matchup to expand. v1 = frozen baseline; v2 = retrained through Nov 17, 2026.</p><div class="head"><span>Matchup</span><span>v1 predicted winner</span><span>v2 predicted winner</span><span>v2 − v1 (home win probability)</span></div>{''.join(rows)}<p class="note">For research only. No injury or lineup adjustments; probability calibration and v2 out-of-sample accuracy have not been established.</p></body></html>'''
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(page,encoding='utf-8')
    print(f'Saved {args.output.resolve()} | matched {matched}/{len(rows)} matchups')

if __name__=='__main__': main()
