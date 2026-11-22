"""Install the read-only historical NBA market-line audit in Parlay Guard.

Requires the previously installed line calibration module; never overwrites
trained models, datasets, or paper-trading snapshots. Backups precede edits.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
import shutil

ROOT=Path.cwd()
SOURCE=Path(__file__).resolve().parent
MARKER='PARLAY_GUARD_MARKET_LINE_AUDIT_V1'
NEW_FILES=['src/market_line_validation.py','tests/test_market_line_validation.py',
           'docs/MARKET_LINE_VALIDATION.md']

API_BLOCK='''# PARLAY_GUARD_MARKET_LINE_AUDIT_V1
@app.get('/api/market-line-validation')
def market_line_validation_report():
    """Local historical BOOKMAKER quote audit; separate from live Polymarket EV."""
    import json
    path=ROOT/'models/market_line_validation_v1.json'
    if not path.exists():
        return JSONResponse({'status':'no_historical_market_lines',
            'message':'Import timestamped historical NBA spread/total offers and run validation. '
                      'Synthetic-threshold calibration is not bookmaker calibration.'},
            headers={'Cache-Control':'no-store'})
    try:
        data=json.loads(path.read_text(encoding='utf-8'))
        if data.get('validation_scope')!='timestamped_historical_bookmaker_reference_lines':
            raise ValueError('Unexpected audit schema')
        return JSONResponse(data,headers={'Cache-Control':'no-store'})
    except (OSError,ValueError,TypeError):
        return JSONResponse({'status':'invalid_historical_audit',
            'message':'Historical market-line audit is invalid; rerun the validator.'},
            headers={'Cache-Control':'no-store'})

'''
HTML_BLOCK='''<!-- PARLAY_GUARD_MARKET_LINE_AUDIT_V1 -->
<section id="market-line-audit">
<h2>Historical market-line audit</h2>
<p class="muted">Observed pregame bookmaker spreads and totals, one fixed book and a chronological test period. This is separate from Polymarket executable prices.</p>
<div class="note" id="market-audit-summary">No real historical market-line audit loaded yet.</div>
</section>
'''
JS_BLOCK='''// PARLAY_GUARD_MARKET_LINE_AUDIT_V1
async function renderMarketLineAudit(){
  const root=$('market-audit-summary');if(!root)return;
  try{
    const response=await fetch('/api/market-line-validation',{cache:'no-store'});
    if(!response.ok)throw new Error('HTTP '+response.status);
    const report=await response.json();root.replaceChildren();
    if(report.status){root.append(el('div','muted',report.message||report.status));return;}
    const policy=report.quote_policy||{};
    root.append(el('div','muted small','Reference bookmaker: '+(policy.bookmaker||'not set')+
      ' · minimum '+(policy.lead_minutes??'—')+' minutes before tipoff'+
      ' · live Polymarket prices NOT validated'));
    const targets=report.targets||{};
    for(const [key,name] of [['home_margin','Spread'],['total','Total']]){
      const t=targets[key]||{},b=t.test_baseline||{},c=t.test_calibrated_on_support||{};
      const line=el('div','ticket');line.append(el('b','',name+' · '+(t.status||'not evaluated')));
      line.append(el('div','muted small','Validation quotes '+(t.validation_quotes_resolved??0)+
        ' · test quotes '+(t.test_quotes_resolved??0)+
        ' · Brier (raw) '+(b.brier==null?'—':Number(b.brier).toFixed(4))+
        ' · ECE '+(b.ece_10_bins==null?'—':pct(b.ece_10_bins))));
      line.append(el('div','muted small','Calibrated Brier on supported test quotes: '+
        (c.brier==null?'unavailable':Number(c.brier).toFixed(4))+
        ' · research gate '+(t.research_gate_passed?'passed':'not passed')));
      root.append(line);
    }
    root.append(el('div','gold small','Reference bookmaker offers only; no guaranteed fills, validated Polymarket EV or automatic live-model replacement.'));
  }catch(e){root.textContent='Historical audit unavailable: '+e.message;}
}
'''


def _once(src: str,old: str,new: str,desc: str) -> str:
    count=src.count(old)
    if count!=1:
        raise ValueError(f'{desc}: expected one anchor, found {count}. No files changed.')
    return src.replace(old,new,1)


def patch_api(src: str) -> str:
    if MARKER in src:return src
    src=_once(src,"@app.get('/api/health')",API_BLOCK+"@app.get('/api/health')",'API health route')
    compile(src,'src/api.py','exec')
    return src


def patch_html(src: str) -> str:
    if MARKER in src:return src
    src=_once(src,'<section><h2>Data health</h2>',
              HTML_BLOCK+'\n<section><h2>Data health</h2>','data health section')
    src=_once(src,'async function poll(){',JS_BLOCK+'\nasync function poll(){','poll function')
    src=_once(src,'lastData=await r.json();render()',
              'lastData=await r.json();render();renderMarketLineAudit()', 'live fetch render')
    return src


def main():
    api=ROOT/'src/api.py';html=ROOT/'static/live.html'
    if not api.exists() or not html.exists() or not (ROOT/'src/calibrate_game_lines.py').exists():
        raise SystemExit('Run from the Parlay Guard repo root with the earlier calibration upgrade installed.')
    if any(not (SOURCE/p).exists() for p in NEW_FILES):
        raise SystemExit('Extract the ENTIRE ZIP in the Parlay Guard repo root first.')
    old_api=api.read_text(encoding='utf-8');old_html=html.read_text(encoding='utf-8')
    if MARKER in old_api and MARKER in old_html and all((ROOT/p).exists() for p in NEW_FILES):
        print('Historical market-line audit already installed. No changes.');return
    if MARKER in old_api or MARKER in old_html:
        raise SystemExit('Partial installation. Refusing to change working source automatically.')
    for path in NEW_FILES:
        if (ROOT/path).exists() and (ROOT/path).read_bytes() != (SOURCE/path).read_bytes():
            raise SystemExit(f'Existing local file differs; refusing overwrite: {path}')
    new_api=patch_api(old_api);new_html=patch_html(old_html)
    backup=Path.home()/'.parlay_guard_backups'/datetime.now().strftime('%Y%m%d_%H%M%S_market_line_audit')
    for target in (api,html):
        destination=backup/target.relative_to(ROOT);destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(target,destination)
    for name in NEW_FILES:
        target=ROOT/name;target.parent.mkdir(parents=True,exist_ok=True)
        if (SOURCE/name).resolve() != target.resolve():
            shutil.copy2(SOURCE/name,target)
    api.write_text(new_api,encoding='utf-8')
    html.write_text(new_html,encoding='utf-8')
    print('Installed historical market-line audit (read-only; no betting actions).')
    print('Backups:',backup)
    print('Run: python -m pytest tests/ -q')
    print('Run: python -m src.market_line_validation report')
    print('Restart Uvicorn to expose /api/market-line-validation and dashboard panel.')

if __name__=='__main__':main()
