"""Idempotent, non-destructive installer for Parlay Guard's /portfolio dashboard.

Only modifies src/api.py to register one new APIRouter and optionally adds a
navigation link to static/live.html. Copies both originals into timestamped
backups first. Does NOT alter data, predictions, frozen paper trades or other
previous upgrade scripts.
"""
from __future__ import annotations
import ast
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
API = ROOT/'src'/'api.py'
LIVE = ROOT/'static'/'live.html'
MARK = '# PG_MULTI_SPORT_PORTFOLIO_V1'
NAV_MARK = 'data-pg-portfolio-link="1"'


def patch_api(source):
    if MARK in source:
        return source, False
    match = re.search(r'(?m)^\s*@app\.get\(\s*([\'\"])/', source)
    if not match:
        raise RuntimeError('Could not find the FastAPI homepage @app.get("/") anchor. No files changed.')
    insertion = ('\n'+MARK+'\nfrom .portfolio_api import router as portfolio_router\n'
                 'app.include_router(portfolio_router)\n\n')
    updated = source[:match.start()]+insertion+source[match.start():]
    ast.parse(updated)
    return updated, True


def patch_live(source):
    if NAV_MARK in source:
        return source, False
    match = re.search(r'</header\s*>',source,re.I)
    if not match:
        # Older dashboard assets may have no header. Route still works.
        return source, False
    anchor=('<a data-pg-portfolio-link="1" href="/portfolio" '
            'style="display:inline-block;margin:10px 6px;padding:8px 14px;'
            'border:1px solid #3e8168;border-radius:9px;color:#9effcb;text-decoration:none;'
            'font-weight:700">Multi-Sport EV Portfolio</a>\n')
    return source[:match.end()]+ '\n'+anchor + source[match.end():], True


def write_atomic(path,text):
    target=path.with_name(path.name+'.portfolio.tmp')
    target.write_text(text,encoding='utf-8')
    target.replace(path)


def main():
    needed=[ROOT/'src'/'portfolio_api.py', ROOT/'static'/'portfolio.html',
            ROOT/'static'/'portfolio_ui.js', ROOT/'static'/'portfolio_core.js']
    if not all(p.is_file() for p in needed):
        raise SystemExit('Missing portfolio upgrade files: extract full ZIP into project root first.')
    if not API.is_file():
        raise SystemExit('Expected src/api.py. Run the installer in the existing Parlay Guard project root.')
    original=API.read_text(encoding='utf-8'); updated,change_api=patch_api(original)
    old_live=LIVE.read_text(encoding='utf-8') if LIVE.is_file() else None
    new_live,change_live=patch_live(old_live) if old_live is not None else (None,False)
    if not change_api and not change_live:
        print('Portfolio route already registered; no existing files changed.')
        return
    backup=ROOT/'backups'/('portfolio_'+datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%fZ'))
    backup.mkdir(parents=True,exist_ok=True)
    if change_api:
        shutil.copy2(API,backup/'api.py')
    if change_live:
        shutil.copy2(LIVE,backup/'live.html')
    if change_api:
        write_atomic(API,updated)
    if change_live:
        write_atomic(LIVE,new_live)
    print('Installed the read-only multi-sport portfolio dashboard.')
    print('Backup:', backup)
    print('Dashboard: http://127.0.0.1:8000/portfolio')
    print('Feeds: NBA live + college endpoint (if installed) + external online reports/CSV.')
    print('Frozen moneyline and same-game tracker files were not changed.')
    if old_live is None or not change_live:
        print('Navigation link not added (already present or no compatible </header>); open /portfolio directly.')

if __name__=='__main__':
    main()
