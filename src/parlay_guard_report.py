"""Offline Parlay Guard HTML report with expandable model input breakdown.

Run: python -m src.parlay_guard_report --ev models/polymarket_ev_....csv --predictions models/live_predictions_....csv
"""
import argparse
import html
from pathlib import Path
import pandas as pd


def esc(v):
    return html.escape(str(v), quote=True)


def pct(v, signed=False):
    return f'{float(v):+,.1%}' if signed else f'{float(v):.1%}'


def num(v, digits=1):
    return f'{float(v):+,.{digits}f}'


def factor_breakdown(pred):
    if pred is None:
        return '<p class="muted">No matching prediction features found. Pass <code>--predictions</code> with the frozen forecast CSV.</p>'
    factors = [
        ('Last 5 games: win rate', 'HOME_WIN_PCT_LAST_5', 'AWAY_WIN_PCT_LAST_5', 'percent', 'Recent form (short window)'),
        ('Last 10 games: win rate', 'HOME_WIN_PCT_LAST_10', 'AWAY_WIN_PCT_LAST_10', 'percent', 'Recent form (longer window)'),
        ('Last 10 games: point differential', 'HOME_POINT_DIFF_LAST_10', 'AWAY_POINT_DIFF_LAST_10', 'signed', 'Recent scoring margin per game'),
        ('Days of rest', 'HOME_DAYS_REST', 'AWAY_DAYS_REST', 'int', 'Time since last game'),
    ]
    home, away = str(pred['HOME_TEAM']), str(pred['AWAY_TEAM'])
    lines=[]
    for label,hcol,acol,fmt,meaning in factors:
        hv, av = pd.to_numeric(pred.get(hcol),errors='coerce'),pd.to_numeric(pred.get(acol),errors='coerce')
        if pd.isna(hv) or pd.isna(av):
            continue
        if fmt=='percent': display=lambda v: pct(v)
        elif fmt=='int': display=lambda v: f'{v:g}'
        else: display=lambda v: num(v)
        advantage = home if hv>av else away if av>hv else 'Even'
        lines.append(f'<tr><td>{esc(label)}</td><td>{display(av)}</td><td>{display(hv)}</td><td>{esc(advantage)}</td><td class="muted">{esc(meaning)}</td></tr>')
    elo_h = pd.to_numeric(pred.get('HOME_ELO'),errors='coerce')
    elo_a = pd.to_numeric(pred.get('AWAY_ELO'),errors='coerce')
    if pd.notna(elo_h) and pd.notna(elo_a):
        adv = home if elo_h>elo_a else away if elo_a>elo_h else 'Even'
        lines.append(f'<tr><td>Elo rating</td><td>{elo_a:.0f}</td><td>{elo_h:.0f}</td><td>{esc(adv)}</td><td class="muted">Pre-game team strength estimate</td></tr>')
    if not lines:
        return '<p class="muted">The prediction CSV does not contain the expected model features.</p>'
    pgb = pd.to_numeric(pred.get('P_HOME_WIN_GB'),errors='coerce')
    pelo = pd.to_numeric(pred.get('P_HOME_WIN_ELO'),errors='coerce')
    summary = ''
    if pd.notna(pgb):
        summary += f'<strong>Gradient Boosting:</strong> {esc(home)} {pct(pgb)}, {esc(away)} {pct(1-pgb)}'
    if pd.notna(pelo):
        summary += f' &nbsp; · &nbsp; <strong>Elo baseline:</strong> {esc(home)} {pct(pelo)}, {esc(away)} {pct(1-pelo)}'
    return f'''<h3>Factors used by the prediction model</h3><p>{summary}</p><div class="scroll"><table class="inner"><thead><tr><th>Feature</th><th>{esc(away)} (away)</th><th>{esc(home)} (home)</th><th>Higher value</th><th>What it measures</th></tr></thead><tbody>{''.join(lines)}</tbody></table></div><p class="note">These are the actual pre-game inputs to the model, not measured causal contributions. A higher value is not always a positive effect (for example, extra rest can have nonlinear effects). This report does not calculate SHAP values or quantify how much each feature changed the prediction. Injuries, confirmed starters, and travel are not model inputs.</p>'''


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ev',type=Path,help='Polymarket EV CSV (default: latest models/polymarket_ev_*.csv)')
    parser.add_argument('--predictions',type=Path,help='Matching immutable live_predictions CSV (default: latest same-date file)')
    parser.add_argument('--output',type=Path,help='Output HTML path')
    args=parser.parse_args()
    source=args.ev
    if source is None:
        files=sorted(Path('models').glob('polymarket_ev_*.csv'),key=lambda f:f.stat().st_mtime)
        if not files: parser.error('No Polymarket EV CSV found')
        source=files[-1]
    df=pd.read_csv(source)
    needed={'away','home','team','model_win_prob','best_ask','edge','expected_roi','status'}
    if not needed.issubset(df.columns):parser.error('Missing EV columns: '+', '.join(sorted(needed-set(df.columns))))
    if df.empty:parser.error('EV CSV is empty')
    for c in ('model_win_prob','best_ask','edge','expected_roi'):
        df[c]=pd.to_numeric(df[c],errors='coerce')
    df=df.dropna(subset=['model_win_prob','best_ask','edge','expected_roi'])
    if df.empty:parser.error('No valid EV rows')
    game_date=str(df['game_date'].iloc[0]) if 'game_date' in df else None
    checked=str(df['checked_at_utc'].iloc[0]) if 'checked_at_utc' in df else 'Unknown'
    predictions=args.predictions
    if predictions is None:
        pattern=f'live_predictions_{game_date}_*.csv' if game_date and game_date!='nan' else 'live_predictions_*.csv'
        candidates=sorted(Path('models').glob(pattern),key=lambda f:f.stat().st_mtime)
        if len(candidates)==1: predictions=candidates[0]
        elif len(candidates)>1:
            print('Multiple prediction snapshots found; pass --predictions to ensure exact matching. Feature breakdown omitted.')
    lookup={}
    if predictions:
        pdf=pd.read_csv(predictions,dtype={'GAME_ID':str})
        for col in ('AWAY_TEAM','HOME_TEAM'):
            if col not in pdf:parser.error(f'Prediction CSV missing {col}')
        for _,r in pdf.iterrows():
            key=(str(r.AWAY_TEAM),str(r.HOME_TEAM))
            if key in lookup:parser.error(f'Duplicate forecast matchup: {key}')
            lookup[key]=r
        print(f'Loaded model factors from {predictions}')
    rows=[];flagged=0
    for (away,home),group in df.groupby(['away','home'],sort=True):
        group=group.sort_values('expected_roi',ascending=False)
        top=group.iloc[0];winner=group.sort_values('model_win_prob',ascending=False).iloc[0]
        flag=group['status'].astype(str).str.contains('POTENTIAL EDGE',case=False).any()
        flagged+=int(flag)
        detail=[]
        for _,side in group.iterrows():
            url=str(side.get('url',''))
            link=f'<a href="{esc(url)}" target="_blank" rel="noopener noreferrer">Market</a>' if url.startswith('https://polymarket.com/') else ''
            depth=pd.to_numeric(side.get('top_ask_depth_usd'),errors='coerce')
            depth_text=f'${depth:,.2f}' if pd.notna(depth) else 'N/A'
            detail.append(f'<tr><td>{esc(side.team)}</td><td>{pct(side.model_win_prob)}</td><td>${side.best_ask:.3f}</td><td>{pct(side.edge,True)}</td><td class="{"positive" if side.expected_roi>0 else "negative"}">{pct(side.expected_roi,True)}</td><td>{esc(side.status)}</td><td>{depth_text}</td><td>{link}</td></tr>')
        factors=factor_breakdown(lookup.get((str(away),str(home))))
        status='<span class="pill">Review edge</span>' if flag else '<span class="muted">No flagged edge</span>'
        roi_class='positive' if top.expected_roi>0 else 'negative'
        index=len(rows)
        rows.append(f'''<tr class="main-row"><td><strong>{esc(away)} @ {esc(home)}</strong></td><td>{esc(winner.team)}</td><td>{pct(winner.model_win_prob)}</td><td>{esc(top.team)}</td><td class="{roi_class}">{pct(top.expected_roi,True)}</td><td>{status}</td><td><button aria-expanded="false" aria-controls="d{index}" onclick="toggle(this,'d{index}')">Show factors</button></td></tr><tr id="d{index}" class="details" hidden><td colspan="7"><div class="detail-panel">{factors}<h3>Polymarket prices and EV</h3><div class="scroll"><table class="inner"><thead><tr><th>Side</th><th>Win chance</th><th>Ask</th><th>Edge</th><th>Est. ROI</th><th>Status</th><th>Top ask depth</th><th>Link</th></tr></thead><tbody>{''.join(detail)}</tbody></table></div></div></td></tr>''')
    output=args.output or source.with_name(source.stem+'_report.html')
    output.parent.mkdir(parents=True,exist_ok=True)
    content=f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Parlay Guard | {esc(game_date or 'NBA')}</title><style>
:root{{font-family:system-ui,-apple-system,Segoe UI,sans-serif;color:#e9edf5;background:#10151f}}body{{max-width:1240px;margin:40px auto;padding:0 20px}}h1{{margin:0;font-size:30px}}h3{{font-size:17px;margin:18px 0 10px}}.muted,.sub{{color:#a7b1c2}}.sub{{margin:8px 0 26px}}.cards{{display:flex;gap:14px;flex-wrap:wrap;margin:20px 0}}.card{{background:#1c2534;padding:17px 24px;border-radius:12px;min-width:140px}}.card strong{{font-size:26px;display:block}}.card span{{font-size:12px;color:#a7b1c2}}.scroll{{overflow-x:auto}}table{{border-collapse:collapse;width:100%;text-align:left}}th,td{{padding:13px 11px;border-bottom:1px solid #30394a;white-space:nowrap}}th{{color:#a7b1c2;font-size:12px;text-transform:uppercase}}.main-row:hover{{background:#1b2533}}.positive{{color:#65dfa7;font-weight:700}}.negative{{color:#f6a6a6}}.pill{{color:#65dfa7;border:1px solid #368769;border-radius:100px;padding:4px 9px;font-size:12px}}button{{background:#293a54;color:white;border:1px solid #46536a;border-radius:7px;padding:7px 11px;cursor:pointer}}.details td{{background:#192231}}.inner th,.inner td{{padding:9px}}.detail-panel{{padding:12px 8px 20px}}.detail-panel p{{white-space:normal;line-height:1.6}}a{{color:#93c5fd}}.note{{line-height:1.6;margin-top:22px;color:#a7b1c2;font-size:13px}}code{{color:#bcd6f6}}
</style></head><body><h1>Parlay Guard</h1><p class="sub">NBA predictions and Polymarket estimated value · {esc(game_date or 'NBA games')}<br>Market prices checked: {esc(checked)} UTC</p><div class="cards"><div class="card"><strong>{len(rows)}</strong><span>Matched games</span></div><div class="card"><strong>{flagged}</strong><span>Games with flagged sides</span></div></div><div class="scroll"><table><thead><tr><th>Matchup</th><th>Predicted winner</th><th>Win probability</th><th>Best EV side</th><th>Estimated ROI</th><th>Review status</th><th>Explore</th></tr></thead><tbody>{''.join(rows)}</tbody></table></div><p class="note">Model features are pre-game inputs, not proof of causal influence. Estimated EV relies on model calibration, prices, fees and liquidity. Check injuries and starting lineups separately. Prices are a snapshot, not live quotes. Research only; no bet recommended.</p><script>function toggle(button,id){{const e=document.getElementById(id);e.hidden=!e.hidden;button.textContent=e.hidden?'Show factors':'Hide factors';button.setAttribute('aria-expanded',String(!e.hidden));}}</script></body></html>'''
    output.write_text(content,encoding='utf-8')
    print(f'Created {output.resolve()}')
    print(f'Games: {len(rows)} | Games with flagged sides: {flagged} | Feature matches: {sum(k in lookup for k in df.groupby(["away","home"]).groups)}')

if __name__=='__main__':main()
