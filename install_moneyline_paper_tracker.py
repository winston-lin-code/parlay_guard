"""Safely add prospective moneyline paper tracking to the existing Parlay Guard API.

Run from Parlay Guard project root AFTER extracting this ZIP:
  python install_moneyline_paper_tracker.py
No orders or wagers are sent. Existing historical data/models are not modified.
"""
from __future__ import annotations

import ast
import shutil
from datetime import datetime, timezone
from pathlib import Path

ROOT=Path.cwd()
SOURCE=Path(__file__).resolve().parent
MARKER='PARLAY_GUARD_MONEYLINE_PAPER_TRACKER_V1'

HOOK = '''                # PARLAY_GUARD_MONEYLINE_PAPER_TRACKER_V1
                # Prospective PAPER-ONLY capture, never a trade/real money order.
                try:
                    from .moneyline_paper_track import on_live_refresh
                    output['moneyline_tracking'] = on_live_refresh(output, ROOT)
                except Exception as paper_exc:
                    output['moneyline_tracking'] = {'status': 'ERROR', 'paper_only': True,
                                                    'error': f'{type(paper_exc).__name__}: {paper_exc}'[:350]}
                    output.setdefault('warnings', []).append('Moneyline paper tracking: '+str(paper_exc)[:140])
'''

ROUTES = '''
# PARLAY_GUARD_MONEYLINE_PAPER_TRACKER_V1 — read-only tracker routes
@app.get('/api/moneyline-tracker')
def moneyline_paper_tracker_report():
    from .moneyline_paper_track import report
    return report(ROOT)

@app.get('/moneyline-tracker', response_class=HTMLResponse)
def moneyline_paper_tracker_page():
    page = ROOT / 'static/moneyline_tracker.html'
    if not page.is_file():
        raise HTTPException(503, 'Moneyline tracker page missing')
    return FileResponse(page, media_type='text/html', headers={'Cache-Control': 'no-store'})
'''


def update_api(original: str):
    if MARKER in original:
        if original.count(MARKER) != 2:
            raise RuntimeError('Existing moneyline tracking markers are incomplete; refusing to edit')
        return original, False
    anchor='                # Only publish fully assembled snapshots.'
    if original.count(anchor)!=1:
        raise RuntimeError('Could not safely locate LiveState._loop publishing hook in src/api.py')
    code=original.replace(anchor, HOOK+anchor, 1).rstrip()+'\n'+ROUTES
    ast.parse(code, filename='src/api.py')
    return code, True


def main():
    api=ROOT/'src/api.py'
    if not api.is_file():
        raise SystemExit('Open PowerShell in your Parlay Guard root (src/api.py must exist)')
    for relative in ('src/moneyline_paper_track.py', 'static/moneyline_tracker.html',
                     'tests/test_moneyline_paper_track.py', 'docs/MONEYLINE_PAPER_TRACKING.md'):
        target=ROOT/relative
        source=SOURCE/relative
        if not source.is_file() or not target.is_file():
            raise SystemExit(f'Missing {relative}. Extract the full ZIP in your repository root first.')
        if target.read_bytes()!=source.read_bytes():
            # Files may be extracted into root itself (SOURCE == ROOT).
            # In that case they are the same exact file, therefore no mismatch.
            raise SystemExit(f'{relative} differs from upgrade file. Refusing to install mismatched code.')
    old=api.read_text(encoding='utf-8')
    updated,modified=update_api(old)
    if modified:
        backup=ROOT/'backups'/('moneyline_tracker_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
        backup.mkdir(parents=True, exist_ok=True)
        shutil.copy2(api,backup/'api.py')
        api.write_text(updated,encoding='utf-8')
        print('Patched src/api.py; original saved to',backup/'api.py')
    else:
        print('API integration already installed (no duplicate patch).')
    # Safe to rerun. Configuration cannot be overwritten after first freeze.
    from src.moneyline_paper_track import init
    cfg=init(ROOT)
    print('Moneyline tracking strategy frozen at:', cfg['frozen_at_utc'])
    print('Frozen strategy checksum:',cfg['strategy_hash'])
    print('Paper only: no exchange orders and no real money stakes.')
    print('Open http://127.0.0.1:8000/moneyline-tracker after restarting Uvicorn.')
    print('To include all potential NBA matchups, launch with PG_MAX_GAMES=20 (see README).')

if __name__=='__main__':
    main()
