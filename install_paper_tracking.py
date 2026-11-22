"""Install Parlay Guard's immutable paper-strategy outcome tracker.

From the existing Parlay Guard project root, after extracting the archive:
    python install_paper_tracking.py

No wagers or network requests are made by this installer. Existing models,
historical data, and paper records are neither replaced nor deleted.
"""
from __future__ import annotations

import re
import shutil
from datetime import datetime
from pathlib import Path

ROOT = Path.cwd()
SOURCE = Path(__file__).resolve().parent / 'patch_files'
MARKER = 'PARLAY_GUARD_PAPER_TRACKING_V1'

HTML_SECTION = '''<!-- PARLAY_GUARD_PAPER_TRACKING_V1 -->
<section id="paper-tracking">
<h2>Paper strategy outcomes · longitudinal research</h2>
<p class="muted">One immutable pre-tipoff slate per NBA date. Completed game and player records settle each ticket only when verified; partial and unverified outcomes remain pending. No real orders or fills.</p>
<div class="strategies" id="paper-results"><div class="note">Awaiting paper-trading records…</div></div>
<p class="small muted">Single reference returns assume top-ask fills that were NOT confirmed. Cross-game prop returns use synthetic products that are NOT executable Combo quotes. Same-game and prop+line pairs have hit rates only because actual Combo prices are missing. These figures are not trading profit.</p>
</section>
'''

HTML_JS = '''// PARLAY_GUARD_PAPER_TRACKING_V1
function renderPaperTracking(){
  const root=$('paper-results');if(!root)return;root.replaceChildren();
  const report=lastData.paper_tracking;
  if(!report){root.append(el('div','note','No paper tracker output yet. A new slate is recorded only when eligible pregame props and player IDs are available.'));return;}
  if(report.error){root.append(el('div','notice','Paper tracking unavailable: '+report.error));return;}
  const names={single_prop:'Single props',cross_game_props:'Cross-game prop pairs',
    same_game_prop_line:'Same-game prop + line',cross_game_prop_line:'Cross-game prop + line'};
  const header=el('div','note',String(report.recorded_slates||0)+' immutable pre-tipoff slate(s) recorded. '+
    ((report.newly_captured||[]).length?'New captures: '+report.newly_captured.join(', ')+'. ':'')+
    'Results update as complete NBA team and player records become available.');
  root.append(header);
  for(const row of report.strategies||[]){
    const card=el('div','strategy');card.append(el('h3','',names[row.strategy]||row.strategy));
    card.append(el('div','muted small','Tickets '+row.tickets+' · resolved '+(row.wins+row.losses)+
      ' · pending '+row.pending+' · pushes '+row.pushes));
    card.append(el('div','big',row.hit_rate==null?'—':pct(row.hit_rate)));
    card.append(el('div','muted small','Verified ticket hit rate · '+row.fully_settled_slates+
      ' fully resolved slate(s)'));
    card.append(el('div','sep'));
    if(row.mean_paper_reference_roi==null){
      card.append(el('div','muted small','Paper reference ROI: unavailable'));
    }else{
      const label=row.strategy==='cross_game_props'?'Synthetic return, NOT executable':'Paper top-ask scenario return, NO fills';
      const roi=el('div','gold','Equal-budget mean reference ROI '+pct(row.mean_paper_reference_roi));
      card.append(roi,el('div','muted small',label));
    }
    card.append(el('div','muted small','Mean Brier score (uncalibrated): '+
      (row.mean_brier==null?'—':Number(row.mean_brier).toFixed(3))));
    root.append(card);
  }
  if((report.pending_dates||[]).length){root.append(el('div','note',
    'Slates with unresolved results: '+report.pending_dates.slice(-8).join(', ')));}
}
'''

AUTO_BLOCK = '''# PARLAY_GUARD_PAPER_TRACKING_V1
try:
    from .paper_track_joint import capture_pregame_snapshots, settle_pending, performance_summary
    new_slates = capture_pregame_snapshots(snapshot, ROOT)
    settle_pending(ROOT)
    snapshot['paper_tracking'] = performance_summary(ROOT)
    snapshot['paper_tracking']['newly_captured'] = new_slates
except Exception as exc:
    warnings.append('Paper tracker unavailable: '+str(exc)[:170])
    snapshot['warnings'] = warnings[-12:]
    snapshot['paper_tracking'] = {'error':str(exc)[:170], 'recorded_slates':0, 'strategies':[]}
'''


def patch_auto_live(content):
    if MARKER in content:
        return content
    if 'PARLAY_GUARD_JOINT_RESEARCH_V1' not in content:
        raise ValueError('Install the previous Joint Probability Lab first.')
    matches = list(re.finditer(r'(?m)^([ \t]*)return snapshot\s*$', content))
    if len(matches) != 1:
        raise ValueError('Could not identify the final joint-lab return snapshot line uniquely. No files changed.')
    found = matches[0]
    leading = found.group(1)
    block = '\n'.join(leading + line if line else '' for line in AUTO_BLOCK.rstrip().split('\n'))
    result = content[:found.start()] + block + '\n' + content[found.start():]
    compile(result, 'src/auto_live.py', 'exec')
    return result


def patch_live_html(content):
    if MARKER in content:
        return content
    if 'PARLAY_GUARD_JOINT_RESEARCH_V1' not in content:
        raise ValueError('The live dashboard has not been upgraded with the Joint Probability Lab.')
    section = '<section><h2>Data health</h2>'
    function = 'function render(){'
    call = '  renderJointResearch();\n'
    if any(content.count(x) != 1 for x in (section, function, call)):
        raise ValueError('Unexpected dashboard structure; declining to modify files automatically.')
    content = content.replace(section, HTML_SECTION+'\n'+section, 1)
    content = content.replace(function, HTML_JS+'\n'+function, 1)
    content = content.replace(call, call+'  renderPaperTracking();\n', 1)
    return content


def main():
    auto = ROOT/'src/auto_live.py'
    live = ROOT/'static/live.html'
    extras = ['src/paper_track_joint.py', 'tests/test_paper_track_joint.py',
              'docs/PAPER_TRACKING.md']
    if not auto.is_file() or not live.is_file() or not all((SOURCE/x).is_file() for x in extras):
        raise SystemExit('Extract the ZIP in your Parlay Guard repository root and retry.')
    old_auto, old_html = auto.read_text(encoding='utf-8'), live.read_text(encoding='utf-8')
    if MARKER in old_auto and MARKER in old_html and (ROOT/extras[0]).exists():
        print('Paper tracking is already installed. No files changed.')
        return
    if MARKER in old_auto or MARKER in old_html:
        raise SystemExit('Partial installation detected. Resolve manually instead of overwriting code.')
    new_auto = patch_auto_live(old_auto)
    new_html = patch_live_html(old_html)
    # Validate first; then back up originals outside git and away from data.
    backup_dir = Path.home()/'.parlay_guard_backups'/datetime.now().strftime('%Y%m%d_%H%M%S_paper_tracker')
    for target in [auto, live] + [ROOT/x for x in extras if (ROOT/x).exists()]:
        destination = backup_dir/target.relative_to(ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(target, destination)
    for relative in extras:
        target = ROOT/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(SOURCE/relative, target)
    auto.write_text(new_auto, encoding='utf-8')
    live.write_text(new_html, encoding='utf-8')
    print('Installed immutable paper snapshot capture and verified-only settlement.')
    print('Updated: src/auto_live.py, static/live.html')
    print('Added: '+', '.join(extras))
    print('Backups:', backup_dir)
    print('Run: python -m pytest tests/ -v --tb=short')
    print('Restart Uvicorn to enable automatic capture, settlement and dashboard metrics.')


if __name__ == '__main__':
    main()
