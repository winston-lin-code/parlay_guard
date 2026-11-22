"""Safe, repeatable installer for independent same-game paper comparisons.

Run from the Parlay Guard root after extracting the ZIP.
Does not alter existing frozen moneyline paper picks, hashes, or settlements.
"""
from __future__ import annotations

import ast
import shutil
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path.cwd()
MARKER = 'PARLAY_GUARD_SAME_GAME_COMPARISON_V1'
LIVE_HOOK = '''                # PARLAY_GUARD_SAME_GAME_COMPARISON_V1
                # Side-by-side paper experiment, no trade placement.
                try:
                    from .same_game_paper_track import on_live_refresh as same_game_refresh
                    output['same_game_comparison'] = same_game_refresh(output, ROOT)
                except Exception as comparison_exc:
                    output['same_game_comparison'] = {
                        'status': 'ERROR', 'paper_only': True,
                        'error': f'{type(comparison_exc).__name__}: {comparison_exc}'[:350],
                    }
                    output.setdefault('warnings', []).append(
                        'Same-game paper comparison: '+str(comparison_exc)[:150])
'''
ROUTES = '''
# PARLAY_GUARD_SAME_GAME_COMPARISON_V1 — read-only comparison pages
@app.get('/api/same-game-comparison')
def same_game_comparison_report():
    from .same_game_paper_track import report
    return report(ROOT)

@app.get('/same-game-comparison', response_class=HTMLResponse)
def same_game_comparison_page():
    page = ROOT / 'static/same_game_comparison.html'
    if not page.is_file():
        raise HTTPException(503, 'Comparison dashboard page missing')
    return FileResponse(page, media_type='text/html', headers={'Cache-Control':'no-store'})
'''


def update_api(source):
    if MARKER in source:
        if source.count(MARKER) != 2:
            raise RuntimeError('Incomplete previous same-game integration; refusing to edit')
        return source, False
    if 'PARLAY_GUARD_MONEYLINE_PAPER_TRACKER_V1' not in source:
        raise RuntimeError('Please install the existing moneyline tracker first; this upgrade preserves it')
    anchor = '                # Only publish fully assembled snapshots.'
    if source.count(anchor) != 1:
        raise RuntimeError('Could not find safe automatic-refresh anchor')
    updated = source.replace(anchor, LIVE_HOOK + anchor, 1).rstrip()+'\n'+ROUTES
    ast.parse(updated, filename='src/api.py')
    return updated, True


def update_live(source):
    if MARKER in source:
        if source.count(MARKER) != 2:
            raise RuntimeError('Incomplete previous auto_live integration; refusing to edit')
        return source, False
    # Preserve the existing predicted-winning-team display; add an independent
    # list of eligible props from either team for the comparison tracker.
    anchor = "            'moneylines':moneyline,'selected_props':chosen,"
    if source.count(anchor) != 1:
        raise RuntimeError('Unexpected src/auto_live.py game snapshot structure')
    source = source.replace(anchor,
        "            'moneylines':moneyline,'selected_props':chosen,\n"
        "            # PARLAY_GUARD_SAME_GAME_COMPARISON_V1\n"
        "            'same_game_prop_candidates':candidates,", 1)
    # Preserve original price and EV signals; add the quote timestamp.
    anchor2 = "              'top_ask_shares':as_num(row.get('ask_size_shares'))}"
    if source.count(anchor2) != 1:
        raise RuntimeError('Cannot safely locate live prop source timestamps')
    source = source.replace(anchor2,
        "              'top_ask_shares':as_num(row.get('ask_size_shares')),\n"
        "              'observed_utc':row['observed_utc'].isoformat(),\n"
        "              'game_start_utc':row['game_start_utc'].isoformat()}\n"
        "        # PARLAY_GUARD_SAME_GAME_COMPARISON_V1: timestamps retained for audit", 1)
    ast.parse(source, filename='src/auto_live.py')
    return source, True


def main():
    if not (ROOT / 'src/api.py').exists() or not (ROOT / 'src/auto_live.py').exists():
        raise SystemExit('Run from your Parlay Guard repository root containing src/api.py')
    for relative in ['src/same_game_paper_track.py', 'static/same_game_comparison.html',
                     'tests/test_same_game_paper_track.py', 'docs/SAME_GAME_COMPARISON.md']:
        if not (ROOT / relative).exists():
            raise SystemExit('Extract the complete ZIP first. Missing '+relative)
    api_path = ROOT / 'src/api.py'; live_path = ROOT / 'src/auto_live.py'
    api, api_changed = update_api(api_path.read_text(encoding='utf-8'))
    live, live_changed = update_live(live_path.read_text(encoding='utf-8'))
    if api_changed or live_changed:
        backup = ROOT / 'backups' / ('same_game_comparison_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
        backup.mkdir(parents=True,exist_ok=True)
        if api_changed:
            shutil.copy2(api_path, backup / 'api.py')
            api_path.write_text(api, encoding='utf-8')
        if live_changed:
            shutil.copy2(live_path,backup / 'auto_live.py')
            live_path.write_text(live, encoding='utf-8')
        print('Patched API/live prop feed; backups:',backup)
    else:
        print('API and prop feed integration already installed.')
    from src.same_game_paper_track import init
    cfg = init(ROOT)
    print('Frozen same-game paper experiment:',cfg['frozen_at_utc'])
    print('Config checksum:',cfg['config_hash'][:16])
    print('Existing frozen moneyline experiment was not modified.')
    print('Dashboard: http://127.0.0.1:8000/same-game-comparison')
    print('Research only: no orders and no live wagers.')


if __name__ == '__main__':
    main()
