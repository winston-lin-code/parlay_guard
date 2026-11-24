"""Frozen prospective *paper-only* moneyline strategy comparison for Parlay Guard.

No exchange orders, no real stakes, no retroactive predictions, no invented quotes.
The benchmark records one pregame selected team/observed moneyline ask per game,
then applies the same immutable selection to four independently tracked policies.

CLI: python -m src.moneyline_paper_track init|report|settle
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import tempfile
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any

from .common import ROOT

SCHEMA = 1
STORE = Path('data/paper_trades/moneyline_tracker')
CONFIG_NAME = 'frozen_strategy_v1.json'
CAPTURE_WINDOW_HOURS = 3
QUOTE_MAX_AGE_MINUTES = 20
STRATEGIES = {
    'every_game': {'type': 'all'},
    'ev_5pct': {'type': 'min_ev', 'minimum': 0.05},
    'ev_10pct': {'type': 'min_ev', 'minimum': 0.10},
    'win_prob_70pct': {'type': 'min_probability', 'minimum': 0.70},
}


def _utc(x):
    if isinstance(x, datetime):
        d = x
    elif isinstance(x, str) and x.strip():
        d = datetime.fromisoformat(x.strip().replace('Z', '+00:00'))
    else:
        raise ValueError('Missing/invalid timestamp')
    if d.tzinfo is None:
        raise ValueError('Timezone-aware UTC timestamp required')
    return d.astimezone(timezone.utc)


def _num(x):
    try:
        n = float(x)
        return n if math.isfinite(n) else None
    except (TypeError, ValueError):
        return None


def _canonical(obj):
    return json.dumps(obj, ensure_ascii=True, sort_keys=True, separators=(',', ':'), allow_nan=False)


def _hash(obj):
    return hashlib.sha256(_canonical(obj).encode()).hexdigest()


def _read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def _freeze(path: Path, obj: dict) -> bool:
    """Atomic exclusive create, never overwrite a pregame record."""
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open('x', encoding='utf-8') as f:
            f.write(json.dumps(obj, indent=2, sort_keys=True, allow_nan=False) + '\n')
        return True
    except FileExistsError:
        return False


def _atomic(path: Path, obj: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix='.pending_', suffix='.json', dir=path.parent)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as f:
            f.write(json.dumps(obj, indent=2, sort_keys=True, allow_nan=False) + '\n')
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def init(root=ROOT, *, at=None):
    """Lock rules from installation time onwards. Does not reset existing rules."""
    root = Path(root)
    config_path = root / STORE / CONFIG_NAME
    frozen = {
        'schema': SCHEMA,
        'frozen_at_utc': _utc(at or datetime.now(timezone.utc)).isoformat(),
        'bet_type': 'PAPER_ONLY_ONE_MONEYLINE_SIDE_PER_GAME',
        'stake_units_per_paper_selection': 1.0,
        'capture_window_hours_before_tipoff': CAPTURE_WINDOW_HOURS,
        'quote_max_age_minutes': QUOTE_MAX_AGE_MINUTES,
        'source': 'Parlay Guard live snapshot / Polymarket winner-contract best ask',
        'probability': 'Frozen Elo home-win probability, uncalibrated for executable market EV',
        'strategies': STRATEGIES,
        'policy': ('All-game baseline never filters on estimated EV or confidence; the other '
                   'three strategies select subsets of the IDENTICAL recorded game/pick/quote. '
                   'No orders or fills. Missing quotes/predictions are not fabricated.'),
    }
    frozen['strategy_hash'] = _hash({k: v for k, v in frozen.items() if k != 'strategy_hash'})
    if _freeze(config_path, frozen):
        return frozen
    existing = _read(config_path)
    if existing.get('schema') != SCHEMA or existing.get('strategies') != STRATEGIES:
        raise ValueError('Frozen strategy file differs from this upgrade; refusing to modify it')
    if existing.get('strategy_hash') != _hash({k: v for k, v in existing.items() if k != 'strategy_hash'}):
        raise ValueError('Frozen strategy file checksum mismatch; refusing to record picks')
    return existing


def _game_key(game):
    day, home, away = str(game.get('game_date') or ''), str(game.get('home') or '').upper(), str(game.get('away') or '').upper()
    if not (re.fullmatch(r'\d{4}-\d{2}-\d{2}', day) and re.fullmatch(r'[A-Z]{2,4}',home)
            and re.fullmatch(r'[A-Z]{2,4}',away) and home != away):
        raise ValueError('Invalid game identity')
    datetime.fromisoformat(day)  # Strictly validate the calendar date, not just its shape.
    return f'{day}_{away}_{home}'


def _selections(prob, ask):
    ev = prob / ask - 1.0
    return {
        'every_game': True,
        'ev_5pct': ev >= STRATEGIES['ev_5pct']['minimum'],
        'ev_10pct': ev >= STRATEGIES['ev_10pct']['minimum'],
        'win_prob_70pct': prob >= STRATEGIES['win_prob_70pct']['minimum'],
    }


def _candidate(game: dict, observed: datetime, source_time: datetime, config: dict):
    try:
        tip = _utc(game.get('tipoff_utc'))
        if not source_time <= observed < tip:
            return None, 'not_before_tipoff'
        if tip - observed > timedelta(hours=CAPTURE_WINDOW_HOURS):
            return None, 'outside_capture_window'
        win = game.get('elo_winner_forecast', game.get('winner_forecast')) or {}
        generated = _utc(win.get('forecast_generated_at_utc'))
        if not generated <= observed or observed - generated > timedelta(hours=6):
            return None, 'missing_or_stale_winner_forecast'
        home = str(game.get('home') or '').upper().strip()
        away = str(game.get('away') or '').upper().strip()
        side = str(win.get('predicted_winner') or '').upper().strip()
        hp = _num(win.get('home_probability'))
        if side not in (home, away) or hp is None or not (0 < hp < 1):
            return None, 'invalid_winner_forecast'
        expected = home if hp >= 0.5 else away
        if side != expected:
            return None, 'inconsistent_winner_side'
        p = hp if side == home else 1-hp
        quotes = game.get('moneylines') or {}
        quote = quotes.get(side) or {}
        ask = _num(quote.get('ask'))
        if ask is None or not (0 < ask < 1):
            return None, 'missing_valid_ask'
        shares = _num(quote.get('shares'))
        # Not a depth-complete fill quote: reject an explicitly empty best ask.
        if shares is not None and shares <= 0:
            return None, 'no_ask_liquidity'
        record = {
            'schema': SCHEMA,
            'game_key': _game_key(game),
            'game_date': game['game_date'],
            'home': home, 'away': away,
            'source_event_slug': str(game.get('slug') or ''),
            'source_game_id': str(win.get('nba_game_id') or ''),
            'tipoff_utc': tip.isoformat(),
            'captured_at_utc': observed.isoformat(),
            'source_updated_at_utc': source_time.isoformat(),
            'forecast_generated_at_utc': generated.isoformat(),
            'picked_team': side,
            'model_probability': round(p, 8),
            'home_win_probability': round(hp, 8),
            'moneyline_yes_ask': ask,
            'ask_shares_observed': shares,
            'clob_token_id': str(quote.get('token_id') or ''),
            'estimated_ev_uncalibrated': round(p/ask-1, 8),
            'strategy_selected': _selections(p, ask),
            'strategy_hash': config['strategy_hash'],
            'price_basis': 'OBSERVED_BEST_ASK_NOT_CONFIRMED_FILL',
            'paper_stake_units': 1.0,
            'is_real_bet': False,
        }
        record['record_hash'] = _hash(record)
        return record, None
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        return None, 'incomplete_game_data'


def capture(snapshot: dict, root=ROOT, *, at=None):
    """Freeze exactly one valid quote per game when within 3 hours of tipoff."""
    root = Path(root)
    config = init(root, at=at)
    now = _utc(at or datetime.now(timezone.utc))
    if now < _utc(config['frozen_at_utc']):
        raise ValueError('Cannot backfill paper picks before strategy freeze')
    if (snapshot.get('refresh_status') or {}).get('stale'):
        return {'created': [], 'unavailable': {'snapshot': 'stale'}}
    try:
        source = _utc(snapshot.get('updated_at_utc'))
    except (ValueError, TypeError):
        return {'created': [], 'unavailable': {'snapshot': 'missing_source_time'}}
    if not source <= now <= source+timedelta(minutes=QUOTE_MAX_AGE_MINUTES):
        return {'created': [], 'unavailable': {'snapshot': 'stale_source'}}
    created, unavailable = [], {}
    seen_keys = set()
    for game in snapshot.get('games') or []:
        try:
            key = _game_key(game)
        except ValueError:
            continue
        if key in seen_keys:
            unavailable[key] = 'duplicate_live_game'
            continue
        seen_keys.add(key)
        dest = root / STORE / 'picks' / f'{key}.json'
        if dest.exists():
            continue
        record, reason = _candidate(game, now, source, config)
        if record is None:
            unavailable[key] = reason
            continue
        if _freeze(dest, record):
            created.append(key)
    return {'created': created, 'unavailable': unavailable}


def _season(game_date: str):
    y, m = int(game_date[:4]), int(game_date[5:7])
    return f'{y if m>=7 else y-1}_{(y+1 if m>=7 else y) % 100:02d}'


def _load_results(path: Path):
    """Read only completed, paired NBA team logs from the user's local cache."""
    import pandas as pd
    if not path.exists():
        return {}
    d = pd.read_csv(path, dtype={'GAME_ID': str, 'TEAM_ID': str}, low_memory=False)
    needed = {'GAME_ID', 'GAME_DATE', 'TEAM_ABBREVIATION', 'PTS', 'WL'}
    if not needed.issubset(d.columns):
        raise ValueError(f'{path}: missing result columns {sorted(needed-set(d.columns))}')
    if 'IS_HOME' not in d.columns and 'MATCHUP' not in d.columns:
        raise ValueError(f'{path}: need MATCHUP or IS_HOME')
    dates = pd.to_datetime(d.GAME_DATE, errors='coerce')
    d['DAY'] = dates.dt.strftime('%Y-%m-%d')
    d['TEAM_ABBREVIATION'] = d['TEAM_ABBREVIATION'].astype(str).str.strip().str.upper()
    d['GAME_ID'] = d['GAME_ID'].astype(str).str.strip().str.removesuffix('.0').str.zfill(10)
    if 'IS_HOME' in d.columns:
        d['HOME'] = d.IS_HOME.astype(str).str.strip().str.lower().isin(['true','1'])
    else:
        match = d['MATCHUP'].astype(str).str.upper().str.strip()
        d['HOME'] = match.str.contains(' VS.', regex=False)
        has_at = match.str.contains(' @ ', regex=False)
        d.loc[has_at, 'HOME'] = d.loc[has_at, 'TEAM_ABBREVIATION'] == match[has_at].str.split(' @ ').str[-1].str.strip()
    d['PTS'] = pd.to_numeric(d['PTS'], errors='coerce')
    result = {}
    for ident, rows in d.groupby('GAME_ID'):
        if len(rows) != 2 or rows['DAY'].isna().any() or rows['DAY'].nunique()!=1:
            continue
        if int(rows['HOME'].sum()) != 1 or set(rows.WL) != {'W','L'} or rows.PTS.isna().any():
            continue
        h = rows.loc[rows.HOME].iloc[0]; a = rows.loc[~rows.HOME].iloc[0]
        if h.TEAM_ABBREVIATION == a.TEAM_ABBREVIATION or h.PTS == a.PTS:
            continue
        if (h.PTS > a.PTS) != (h.WL == 'W') or (a.PTS > h.PTS) != (a.WL == 'W'):
            continue
        k = f'{h.DAY}_{a.TEAM_ABBREVIATION}_{h.TEAM_ABBREVIATION}'
        result.setdefault(k, []).append({'nba_game_id': str(ident), 'home_pts': float(h.PTS),
                                          'away_pts': float(a.PTS),
                                          'winner': h.TEAM_ABBREVIATION if h.PTS>a.PTS else a.TEAM_ABBREVIATION})
    return result


def settle(root=ROOT, *, team_csv=None):
    root = Path(root)
    files = sorted((root / STORE / 'picks').glob('*.json'))
    if not files:
        return {'updated': 0, 'pending': 0}
    caches = {}
    updated, pending = 0, 0
    for path in files:
        record = _read(path)
        if record.get('record_hash') != _hash({k:v for k,v in record.items() if k!='record_hash'}):
            raise ValueError(f'Tampered frozen pick: {path}')
        if record.get('strategy_hash') != init(root)['strategy_hash']:
            raise ValueError('Pick does not match frozen strategy rules')
        source = Path(team_csv) if team_csv else root/'data'/f"nba_{_season(record['game_date'])}_latest_raw.csv"
        if source not in caches:
            caches[source] = _load_results(source)
        rows = caches[source].get(record['game_key'], [])
        # ESPN IDs are not official NBA IDs. Match by exact date + both teams + one verified game.
        gid = str(record.get('source_game_id') or '')
        if gid.isdecimal():
            rows = [r for r in rows if r['nba_game_id']==gid.zfill(10)]
        if len(rows)!=1:
            pending += 1
            continue
        item = rows[0]
        result = 'WIN' if item['winner']==record['picked_team'] else 'LOSS'
        settlement = {
            'schema': SCHEMA, 'game_key': record['game_key'], 'record_hash': record['record_hash'],
            'nba_game_id_verified': item['nba_game_id'],
            'result': result, 'home_pts': item['home_pts'], 'away_pts': item['away_pts'],
            'settled_at_utc': datetime.now(timezone.utc).isoformat(),
        }
        out = root / STORE / 'settlements' / path.name
        if out.exists():
            old = _read(out)
            if any(old.get(k)!=settlement[k] for k in ('game_key','record_hash','nba_game_id_verified','result','home_pts','away_pts')):
                raise ValueError(f'Conflict with prior settlement {out}: refusing to revise silently')
            continue
        if _freeze(out, settlement):
            updated += 1
    return {'updated': updated, 'pending': pending}


def report(root=ROOT):
    root=Path(root)
    configpath = root / STORE / CONFIG_NAME
    if not configpath.exists():
        return {'status':'NOT_INITIALIZED','paper_only':True}
    config=init(root)
    picks=sorted((root / STORE / 'picks').glob('*.json'))
    totals = {k:{'strategy':k,'selected':0,'settled':0,'wins':0,'losses':0,
                 'pending':0,'paper_profit_units':0.0,'paper_roi':None,
                 'win_rate':None,'max_drawdown_units':None} for k in STRATEGIES}
    outcomes={k:[] for k in STRATEGIES}
    recent_picks=[]
    for path in picks:
        rec=_read(path)
        if rec.get('record_hash') != _hash({k:v for k,v in rec.items() if k!='record_hash'}):
            raise ValueError(f'Tampered frozen pick: {path}')
        if rec.get('strategy_hash') != config['strategy_hash']:
            raise ValueError('Pick config differs from frozen strategy')
        settled_path=root/STORE/'settlements'/path.name
        settled = _read(settled_path) if settled_path.exists() else {}
        if settled and settled.get('record_hash') != rec['record_hash']:
            raise ValueError(f'Settlement checksum mismatch: {settled_path}')
        outcome = settled.get('result', 'PENDING')
        recent_picks.append({'game_date': rec['game_date'], 'matchup': f"{rec['away']} @ {rec['home']}",
                             'picked_team': rec['picked_team'],
                             'model_probability': rec['model_probability'],
                             'moneyline_yes_ask': rec['moneyline_yes_ask'],
                             'estimated_ev_uncalibrated': rec['estimated_ev_uncalibrated'],
                             'captured_at_utc': rec['captured_at_utc'],
                             'result': outcome, 'strategy_selected': rec['strategy_selected']})
        for name, selected in rec['strategy_selected'].items():
            if name not in totals or not selected:
                continue
            row=totals[name]; row['selected']+=1
            if outcome in ('WIN','LOSS'):
                row['settled']+=1
                row['wins' if outcome=='WIN' else 'losses']+=1
                # $1 cost at observed binary ask; payout $1/ask only if won.
                ret=(1.0/rec['moneyline_yes_ask']-1.0) if outcome=='WIN' else -1.0
                row['paper_profit_units']+=ret
                outcomes[name].append((rec['tipoff_utc'],rec['game_key'],ret))
            else:
                row['pending']+=1
    for name,row in totals.items():
        n=row['settled']
        if n:
            row['win_rate']=round(row['wins']/n,6)
            row['paper_profit_units']=round(row['paper_profit_units'],6)
            row['paper_roi']=round(row['paper_profit_units']/n,6)
            peak=0.; equity=0.; drawdown=0.
            for _,_,v in sorted(outcomes[name]):
                equity+=v
                peak=max(peak,equity)
                drawdown=max(drawdown,peak-equity)
            row['max_drawdown_units']=round(drawdown,6)
        else:
            row['paper_profit_units']=0.0
    return {
        'schema': SCHEMA,
        'paper_only': True,
        'frozen_at_utc': config['frozen_at_utc'],
        'strategy_hash': config['strategy_hash'],
        'captured_games': len(picks),
        'strategies': list(totals.values()),
        'recent_picks': sorted(recent_picks, key=lambda r:(r['game_date'],r['matchup']), reverse=True)[:40],
        'status': 'TRACKING' if picks else 'WAITING_FOR_ELIGIBLE_PREGAME_QUOTES',
        'notice': ('One baseline pick for each eligible observed game. No wager is created when a live game '
                   'or valid pretipoff quote is missing. ROI uses $1 hypothetical binary-contract ask cost '
                   'and observed best asks, not confirmed fills. No Polymarket order placement; '
                   'no fees, slippage, or live execution. Probability/EV calibration remains unverified.'),
    }


def on_live_refresh(snapshot: dict, root=ROOT):
    captured=capture(snapshot,root)
    settled=settle(root)
    result=report(root)
    result['newly_captured']=captured['created']
    result['unavailable_live_games']=captured['unavailable']
    result['newly_settled']=settled['updated']
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('init','report','settle','refresh-results'), nargs='?', default='report')
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--team-csv', type=Path, default=None)
    args=parser.parse_args(argv)
    if args.action=='init':
        cfg=init(args.root)
        print(json.dumps({'frozen_at_utc':cfg['frozen_at_utc'], 'strategy_hash':cfg['strategy_hash'],
                          'strategies':cfg['strategies'], 'paper_only':True},indent=2))
    elif args.action=='refresh-results':
        # Explicit optional network action. No connection is made during 'report'.
        from .predict_today import fetch_current_season
        dates=sorted({ _read(p)['game_date'] for p in (Path(args.root)/STORE/'picks').glob('*.json') })
        seasons=sorted({f"{y}-{int(z):02d}" for y,z in (_season(day).split('_') for day in dates)})
        for season in seasons:
            print(f'Refreshing completed NBA team-game cache for {season} ...', flush=True)
            fetch_current_season(season, None)
        if not seasons:
            print('No frozen games yet; no historical data to refresh.')
        print(json.dumps(settle(args.root,team_csv=args.team_csv),indent=2))
        print(json.dumps(report(args.root),indent=2))
    elif args.action=='settle':
        print(json.dumps(settle(args.root, team_csv=args.team_csv), indent=2))
        print(json.dumps(report(args.root),indent=2))
    else:
        print(json.dumps(report(args.root),indent=2))
    return 0


if __name__=='__main__':
    raise SystemExit(main())
