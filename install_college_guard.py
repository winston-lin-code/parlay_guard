"""Idempotently attach college endpoints and UI to the user's existing FastAPI app.
Backs up modified files; never rewrites data, models, or frozen tracker configs.
"""
from __future__ import annotations
import ast
import re
import shutil
from datetime import datetime,timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parent
API=ROOT/'src'/'api.py'
LIVE=ROOT/'static'/'live.html'
IMPORT_MARK='# PARLAY_GUARD_COLLEGE_GUARD_V1'
LINK_MARK='id="college-overview"'


def update_api(original):
    if IMPORT_MARK in original:return original,False
    anchor=re.search(r'(?m)^@app\.get\([\'\"]/',original)
    if not anchor:raise RuntimeError('Could not find homepage route in src/api.py. Installation cancelled safely.')
    insertion=(f'\n{IMPORT_MARK}\nfrom .college_api import router as college_guard_router\n'
               'app.include_router(college_guard_router)\n\n')
    patched=original[:anchor.start()]+insertion+original[anchor.start():]
    ast.parse(patched)
    return patched,True


def update_live(original):
    if LINK_MARK in original:return original,False
    match=re.search(r'</header\s*>',original,re.I)
    if not match:raise RuntimeError('Missing </header> in static/live.html. Installation cancelled safely.')
    html='''
<section id="college-overview" style="border:1px solid #345072;border-radius:13px;padding:16px;margin-top:18px;background:#15243a">
  <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:12px">
    <div><strong style="font-size:20px">College basketball &amp; football moneylines</strong>
      <p style="color:#a4b4cd">Separate trained models · live moneyline EV · injuries and roster changes</p></div>
    <a href="/college">Open college game cards and EV →</a>
  </div>
  <div id="college-overview-content" style="font-size:13px;color:#adbed5">Loading college data…</div>
</section>'''
    patched=original[:match.end()]+html+original[match.end():]
    if not re.search(r'</body\s*>',patched,re.I):raise RuntimeError('Missing </body> in static/live.html')
    patched=re.sub(r'</body\s*>','<script src="/college-preview.js" defer></script>\n</body>',patched,count=1,flags=re.I)
    return patched,True


def main():
    if not API.is_file() or not LIVE.is_file():raise SystemExit('Run from the original Parlay Guard project folder; expected src/api.py and static/live.html')
    a0=API.read_text(encoding='utf8');h0=LIVE.read_text(encoding='utf8')
    a1,ac=update_api(a0);h1,hc=update_live(h0)
    if not ac and not hc:
        print('College Guard API and UI already installed. No existing tracker/model files changed.');return
    backup=ROOT/'backups'/('college_guard_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    backup.mkdir(parents=True,exist_ok=True)
    shutil.copy2(API,backup/'api.py');shutil.copy2(LIVE,backup/'live.html')
    if ac:
        temp=API.with_suffix('.college.tmp');temp.write_text(a1,encoding='utf8');temp.replace(API)
    if hc:
        temp=LIVE.with_suffix('.college.tmp');temp.write_text(h1,encoding='utf8');temp.replace(LIVE)
    print('College Guard installed. Backup:',backup)
    print('Routes: /college, /api/college/live, /api/college/refresh')
    print('Models are trained separately; installer does not create fake predictions.')
    print('Frozen NBA and same-game experiments remain unchanged.')


if __name__=='__main__':main()
