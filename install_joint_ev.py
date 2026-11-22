"""Install Parlay Guard's read-only joint-probability research add-on.

Usage from inside the Parlay Guard repository:
    python install_joint_ev.py

Run only after reviewing the bundled source. No betting/order actions.
"""
from __future__ import annotations

import re
import shutil
from datetime import datetime
from pathlib import Path

ROOT = Path.cwd()
SOURCE = Path(__file__).resolve().parent / 'patch_files'
MARKER = 'PARLAY_GUARD_JOINT_RESEARCH_V1'

HTML_SECTION = '''<!-- PARLAY_GUARD_JOINT_RESEARCH_V1 -->
<section id="joint-lab"><h2>Joint probability lab · research only</h2>
<p class="muted">Singles, independent cross-game prop pairs, and same-game player prop + spread/total pairs.
Combined market prices are not verified. Estimates do not establish profitable bets.</p>
<div class="strategies" id="joint-research"><div class="note">Waiting for research snapshot…</div></div>
<p class="small muted">Historical same-game correlations are shrunken estimates, not proof of a future relationship.
Game-line cover probabilities are uncalibrated approximations from holdout error. Executable Combo EV is unavailable until an actual combo quote is verified.</p>
</section>
'''

HTML_JAVASCRIPT = r'''// PARLAY_GUARD_JOINT_RESEARCH_V1
function renderJointResearch(){
  const root=$('joint-research');if(!root)return;root.replaceChildren();
  const d=lastData.joint_research;
  if(!d){root.append(el('div','note','Joint research has not been calculated for this snapshot.'));return;}
  if(d.error){root.append(el('div','notice small','Joint analysis unavailable: '+d.error));return;}
  const items=[
    ['Singles · stressed estimates',d.singles||[]],
    ['Cross-game props · independence',d.cross_game||[]],
    ['Cross-game prop + line',d.cross_game_line||[]],
    ['Same-game prop + line',d.same_game||[]]
  ];
  for(let i=0;i<items.length;i++){
    const [name, rows]=items[i],card=el('div','strategy');card.append(el('h3','',name));
    if(!rows.length){card.append(el('p','muted small','No eligible research candidates.'));root.append(card);continue;}
    const shown=i===0?[...rows].sort((a,b)=>(b.stressed_single_roi??-99)-(a.stressed_single_roi??-99)).slice(0,6):rows.slice(0,6);
    for(const r of shown){
      const item=el('div','ticket');
      if(i===0){
        item.append(el('b','',r.player+' · '+r.stat+' '+r.side+' '+r.line),
          el('div','muted','Stressed P '+pct(r.stress_probability)+' · ask '+usd(r.individual_ask)+'/share'),
          el('div','gold','Modeled stressed single ROI '+pct(r.stressed_single_roi)+' · unverified'));
      }else if(i===1){
        const first=r.legs?.[0],second=r.legs?.[1];
        item.append(el('b','',(first?.player||'?')+' × '+(second?.player||'?')),
          el('div','muted','Independence joint P '+pct(r.joint_probability)),
          el('div','gold','Synthetic product price '+usd(r.hypothetical_product_ask)+' · NOT executable'),
          el('div','muted','Executable parlay EV: unavailable'));
      }else if(i===2){
        item.append(el('b','',r.player+' '+r.prop+' × '+r.game_line),
          el('div','muted','Different games · joint P '+pct(r.joint_probability)),
          el('div','gold','Line probability uncalibrated · quote unavailable'),
          el('div','muted','Executable parlay EV: unavailable'));
      }else{
        const sample=r.correlation?.n;
        item.append(el('b','',r.player+' '+r.prop+' × '+r.game_line),
          el('div','muted','Joint P '+pct(r.joint_probability)+' · line P '+pct(r.game_line_probability)),
          el('div','muted',sample?'Historical paired games: '+sample+' · shrunken correlation '+r.correlation.shrunk_rho:'Not enough verified historical pairs'),
          el('div','gold','Uncalibrated · executable parlay EV unavailable'));
      }
      card.append(item);
    }
    root.append(card);
  }
  if((d.warnings||[]).length){const warn=el('div','note');for(const w of d.warnings)warn.append(el('p','small',w));root.append(warn);}
}
'''


def backup(path: Path, folder: Path):
    dst = folder/path.relative_to(ROOT)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, dst)


def update_auto_live(text: str) -> str:
    if MARKER in text:
        return text
    pattern = re.compile(
        r'(?m)^(?P<indent>[ \t]*)return assemble_data\(events,preds,models,hist,metrics,props,money,\s*\n'
        r'[ \t]*datetime\.now\(timezone\.utc\),warnings\)'
    )
    matches = list(pattern.finditer(text))
    if len(matches) != 1:
        raise RuntimeError('Could not safely find the final assemble_data return statement exactly once. No changes applied.')
    m = matches[0]
    indent = m.group('indent')
    lines = [
        'snapshot=assemble_data(events,preds,models,hist,metrics,props,money,',
        '                       datetime.now(timezone.utc),warnings)',
        f'# {MARKER}',
        'try:',
        '    from .joint_ev import build_joint_research',
        '    snapshot[\'joint_research\']=build_joint_research(snapshot,ROOT)',
        'except Exception as exc:',
        '    warnings.append(\'Joint probability research unavailable: \'+str(exc)[:170])',
        '    snapshot[\'warnings\']=warnings[-12:]',
        '    snapshot[\'joint_research\']={\'error\':str(exc)[:170],\'singles\':[],\'cross_game\':[],\'cross_game_line\':[],\'same_game\':[]}',
        'return snapshot',
    ]
    replacement='\n'.join(indent+line if line and line != '                       datetime.now(timezone.utc),warnings)' else indent+'                       datetime.now(timezone.utc),warnings)' for line in lines)
    changed = text[:m.start()] + replacement + text[m.end():]
    # Also repair the known historic subprocess typo, if still present.
    changed = changed.replace("'-m','src.predict_current_props.py'", "'-m','src.predict_current_props'")
    changed = changed.replace("'-m','src/predict_current_props.py'", "'-m','src.predict_current_props'")
    compile(changed, 'src/auto_live.py', 'exec')
    return changed


def update_html(text: str) -> str:
    if MARKER in text:
        return text
    section_anchor = '<section><h2>Data health</h2>'
    function_anchor = 'function render(){'
    call_anchor = '  renderStrategies();\n  const warnings=lastData.warnings||[]'
    if text.count(section_anchor) != 1 or text.count(function_anchor) != 1 or text.count(call_anchor) != 1:
        raise RuntimeError('Dashboard layout is different from supported version; no changes applied.')
    text=text.replace(section_anchor, HTML_SECTION+'\n'+section_anchor, 1)
    text=text.replace(function_anchor, HTML_JAVASCRIPT+'\n'+function_anchor,1)
    text=text.replace(call_anchor, '  renderStrategies();\n  renderJointResearch();\n  const warnings=lastData.warnings||[]',1)
    return text


def main():
    auto=ROOT/'src/auto_live.py'
    live=ROOT/'static/live.html'
    if not auto.is_file() or not live.is_file() or not SOURCE.joinpath('src/joint_ev.py').exists():
        raise SystemExit('Run from the Parlay Guard project root after extracting the ZIP here.')
    original_auto = auto.read_text(encoding='utf-8')
    original_live = live.read_text(encoding='utf-8')
    # Prepare and validate both patches BEFORE touching files.
    new_auto = update_auto_live(original_auto)
    new_live = update_html(original_live)
    targets=['src/joint_ev.py','tests/test_joint_ev.py','docs/JOINT_PROBABILITY_LAB.md']
    # Local timestamped backup in user home, not inside repository.
    folder=Path.home()/'.parlay_guard_backups'/datetime.now().strftime('%Y%m%d_%H%M%S_joint_ev')
    paths=[auto,live]+[ROOT/p for p in targets if (ROOT/p).exists()]
    for path in paths:
        backup(path,folder)
    for relative in targets:
        src=SOURCE/relative;dst=ROOT/relative
        dst.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(src,dst)
    auto.write_text(new_auto,encoding='utf-8')
    live.write_text(new_live,encoding='utf-8')
    print('Installed read-only joint probability research lab.')
    print('Updated: src/auto_live.py, static/live.html')
    print('Added: src/joint_ev.py, tests/test_joint_ev.py, docs/JOINT_PROBABILITY_LAB.md')
    print('Backups:',folder)
    print('Run: python -m pytest tests/ -v --tb=short')
    print('Then restart Uvicorn and refresh the dashboard.')


if __name__ == '__main__':
    main()
