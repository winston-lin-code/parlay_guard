"""Parlay Guard: prospective, frozen MONEYLINE vs PROP single-contract paper experiment.

Independent from the existing frozen moneyline experiment. No wagers or orders.
The same observed pre-tipoff snapshot supplies moneyline and prop asks.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from .common import ROOT
from . import moneyline_paper_track as ml

SCHEMA = 1
STORE = Path('data/paper_trades/same_game_v1')
CONFIG = 'frozen_comparison.json'
CAPTURE_HOURS = 1
QUOTE_AGE_MIN = 20
STRATEGIES = {
    'moneyline_only': {'policy': 'always_one_elo_winner', 'stake': 1.0},
    'prop_only': {'policy': 'best_available_stressed_ev_prop_either_team', 'stake': 1.0},
    'best_positive_ev': {'policy': 'best_raw_estimated_ev_of_ml_or_eligible_prop_if_positive', 'stake': 1.0},
    'both_singles': {'policy': 'one_ml_and_one_eligible_prop', 'ml_stake': .5, 'prop_stake': .5},
}
STATS = {'PTS', 'REB', 'AST', 'FG3M'}


def utc(x):
    return ml._utc(x)


def number(x):
    return ml._num(x)


def digest(x):
    return ml._hash(x)


def read(path):
    return ml._read(path)


def freeze(path, obj):
    return ml._freeze(path, obj)


def init(root=ROOT, *, at=None):
    root = Path(root)
    path = root / STORE / CONFIG
    obj = {
        'schema': SCHEMA,
        'frozen_at_utc': utc(at or datetime.now(timezone.utc)).isoformat(),
        'decision_window': 'first successful refresh within 60 minutes before tipoff',
        'capture_hours': CAPTURE_HOURS,
        'max_quote_age_minutes': QUOTE_AGE_MIN,
        'paper_budget_units_per_game_per_strategy': 1.0,
        'model': 'frozen_elo_moneyline_and_existing_screened_player_props',
        'strategies': STRATEGIES,
        'same_game_fairness': 'one simultaneous live snapshot of both observed asks',
        'prop_universe': 'positive stressed-EV existing shortlist, both NBA teams',
        'price_policy': 'observed YES best ask, not a verified executable fill; no fees or slippage',
        'no_real_orders': True,
    }
    obj['config_hash'] = digest({k: v for k, v in obj.items() if k != 'config_hash'})
    if freeze(path, obj):
        return obj
    saved = read(path)
    if (saved.get('schema') != SCHEMA or saved.get('strategies') != STRATEGIES
            or saved.get('config_hash') != digest({k: v for k, v in saved.items() if k != 'config_hash'})):
        raise ValueError('Existing same-game experiment configuration differs or is tampered; refusing to reset')
    return saved


def player_ids(root):
    """Load current player identities only from this project's actual feature file."""
    import pandas as pd
    path = Path(root) / 'data/current_player_features.csv'
    if not path.exists():
        return {}
    try:
        df = pd.read_csv(path, usecols=['event_slug', 'player_name', 'PLAYER_ID'], low_memory=False)
    except ValueError:
        return {}
    ids = {}
    ambiguous = set()
    for row in df.itertuples(index=False):
        key = (str(row.event_slug), re.sub(r'[^a-z0-9]', '', str(row.player_name).casefold()))
        n = number(row.PLAYER_ID)
        if n is None or n < 1 or not n.is_integer():
            continue
        pid = str(int(n))
        if key in ids and ids[key] != pid:
            ambiguous.add(key)
        ids[key] = pid
    for key in ambiguous:
        ids.pop(key, None)
    return ids


def candidate(game, observed, source, cfg, ids):
    """Build exactly one immutable game decision; no invention of market prices."""
    try:
        key = ml._game_key(game)
        tip = utc(game['tipoff_utc'])
        if not source <= observed < tip:
            return None, 'not_pretipoff'
        if tip - observed > timedelta(hours=CAPTURE_HOURS):
            return None, 'outside_shared_capture_window'
        win = game.get('elo_winner_forecast', game.get('winner_forecast')) or {}
        if not isinstance(win, dict):
            return None, 'missing_forecast'
        generated = utc(win['forecast_generated_at_utc'])
        if generated > observed or observed - generated > timedelta(hours=6):
            return None, 'stale_forecast'
        home = str(game['home']).upper().strip()
        away = str(game['away']).upper().strip()
        hp = number(win.get('home_probability'))
        if not (hp is not None and 0 < hp < 1):
            return None, 'missing_probability'
        side = home if hp >= .5 else away
        if str(win.get('predicted_winner')).upper() != side:
            return None, 'inconsistent_winner_forecast'
        quote = (game.get('moneylines') or {}).get(side) or {}
        ask = number(quote.get('ask'))
        shares = number(quote.get('shares'))
        if ask is None or not 0 < ask < 1:
            return None, 'missing_moneyline_ask'
        if shares is not None and shares <= 0:
            return None, 'moneyline_no_liquidity'
        p = hp if side == home else 1-hp
        leg_ml = {'kind': 'moneyline', 'team': side, 'model_probability': round(p, 8),
                  'yes_ask': ask, 'estimated_ev': round(p / ask - 1, 8),
                  'token_id': str(quote.get('token_id') or ''), 'ask_shares': shares}
        valid = []
        dropped = {}
        for r in game.get('same_game_prop_candidates') or []:
            try:
                stat = str(r.get('stat', '')).upper()
                direction = str(r.get('side', '')).upper()
                team = str(r.get('team', '')).upper()
                line = number(r.get('line'))
                pa = number(r.get('ask'))
                pp = number(r.get('model_probability'))
                stress = number(r.get('stressed_ev_roi'))
                liquidity = number(r.get('top_ask_shares'))
                player = str(r.get('player') or '')
                pk = (str(game.get('slug') or ''), re.sub(r'[^a-z0-9]', '', player.casefold()))
                pid = ids.get(pk)
                quoted = utc(r.get('observed_utc'))
                if not (stat in STATS and direction in ('OVER', 'UNDER') and team in (home, away)
                        and line is not None and line % 1 == .5 and pa is not None and 0 < pa < 1
                        and pp is not None and 0 < pp < 1 and stress is not None and stress > 0
                        and pid is not None and quoted <= observed < quoted+timedelta(minutes=QUOTE_AGE_MIN)
                        and quoted < tip and (liquidity is None or liquidity > 0)):
                    dropped['invalid_or_stale_prop_or_missing_player_id'] = dropped.get('invalid_or_stale_prop_or_missing_player_id', 0) + 1
                    continue
                valid.append({'kind': 'prop', 'player': player, 'player_id': pid, 'team': team,
                              'stat': stat, 'side': direction, 'line': line,
                              'model_probability': round(pp, 8), 'yes_ask': pa,
                              'estimated_ev': round(pp / pa - 1, 8),
                              'stressed_ev_roi': stress, 'observed_at_utc': quoted.isoformat(),
                              'ask_shares': liquidity, 'market_url': str(r.get('market_url') or '')})
            except (TypeError, ValueError, KeyError, OverflowError):
                dropped['incomplete_prop'] = dropped.get('incomplete_prop', 0) + 1
        # Stable tie break; choose one prop, not two legs of the same statistic.
        valid.sort(key=lambda x: (-x['stressed_ev_roi'], -x['estimated_ev'], x['player_id'], x['stat'], x['side'], x['line']))
        prop = valid[0] if valid else None
        choices = {
            'moneyline_only': [{'kind': 'moneyline', 'stake': 1.0}],
            'prop_only': [{'kind': 'prop', 'stake': 1.0}] if prop else [],
            'best_positive_ev': [],
            'both_singles': [{'kind': 'moneyline', 'stake': .5}, {'kind': 'prop', 'stake': .5}] if prop else [],
        }
        # Compare raw p/ask-1 on both contracts. Screened props also retain their stressed EV.
        available = [leg_ml] + ([prop] if prop else [])
        best = max(available, key=lambda x: (x['estimated_ev'], x['kind'] == 'moneyline'))
        if best['estimated_ev'] > 0:
            choices['best_positive_ev'] = [{'kind': best['kind'], 'stake': 1.0}]
        record = {
            'schema': SCHEMA, 'config_hash': cfg['config_hash'], 'game_key': key,
            'game_date': game['game_date'], 'home': home, 'away': away,
            'event_slug': str(game.get('slug') or ''),
            'source_game_id': str(win.get('nba_game_id') or ''),
            'tipoff_utc': tip.isoformat(), 'captured_at_utc': observed.isoformat(),
            'snapshot_updated_at_utc': source.isoformat(),
            'forecast_generated_at_utc': generated.isoformat(),
            'moneyline': leg_ml, 'prop': prop,
            'available_valid_prop_count': len(valid), 'dropped_prop_reasons': dropped,
            'prop_status': 'VALID' if prop else 'NO_ELIGIBLE_PROP_AT_SHARED_CUTOFF',
            'strategies': choices, 'paper_only': True,
        }
        record['record_hash'] = digest(record)
        return record, None
    except (ValueError, KeyError, TypeError, OverflowError):
        return None, 'incomplete_game_data'


def capture(snapshot, root=ROOT, *, at=None):
    root = Path(root)
    cfg = init(root, at=at)
    now = utc(at or datetime.now(timezone.utc))
    if now < utc(cfg['frozen_at_utc']):
        raise ValueError('Cannot backdate the experiment')
    if (snapshot.get('refresh_status') or {}).get('stale'):
        return {'new': [], 'unavailable': {'snapshot': 'stale'}}
    try:
        source = utc(snapshot.get('updated_at_utc'))
    except (ValueError, TypeError):
        return {'new': [], 'unavailable': {'snapshot': 'missing_timestamp'}}
    if not (source <= now < source + timedelta(minutes=QUOTE_AGE_MIN)):
        return {'new': [], 'unavailable': {'snapshot': 'stale_source'}}
    ids = player_ids(root)
    created, unavailable, seen = [], {}, set()
    for game in snapshot.get('games') or []:
        try:
            key = ml._game_key(game)
        except ValueError:
            continue
        if key in seen:
            unavailable[key] = 'duplicate_snapshot_game'
            continue
        seen.add(key)
        path = root / STORE / 'picks' / (key+'.json')
        if path.exists():
            continue
        rec, why = candidate(game, now, source, cfg, ids)
        if rec is None:
            unavailable[key] = why
        elif freeze(path, rec):
            created.append(key)
    return {'new': created, 'unavailable': unavailable}


def result_for_prop(rec, nba_game, player_data):
    """Exact game ID+player ID. Missing/no-show is pending, never an automatic win."""
    if rec['prop'] is None:
        return None
    if player_data is None:
        return 'PENDING'
    try:
        from .player_prop_common import TEAM_ID_TO_ABBR
    except ImportError:
        return 'PENDING'
    p = rec['prop']
    df = player_data
    rows = df.loc[(df['GAME_ID'] == nba_game['nba_game_id']) & (df['PLAYER_ID'] == p['player_id'])]
    if len(rows) != 1:
        return 'PENDING'
    row = rows.iloc[0]
    try:
        team_id = int(float(row['TEAM_ID']))
        team = TEAM_ID_TO_ABBR.get(team_id)
        if str(team or '').upper() != p['team']:
            return 'PENDING'
        minutes = number(row['MIN']); actual = number(row[p['stat']])
        if minutes is None or minutes <= 0 or actual is None:
            return 'PENDING'
        diff = actual-p['line']
        if abs(diff) < 1e-9:
            return 'PUSH'
        return 'WIN' if (diff > 0 if p['side'] == 'OVER' else diff < 0) else 'LOSS'
    except (TypeError, KeyError, ValueError):
        return 'PENDING'


def load_player_results(root):
    import pandas as pd
    paths = [Path(root)/'data/live_player_training_data_v2.csv', Path(root)/'data/processed/player_training_data_v2.csv']
    if not paths[0].exists() and not paths[1].exists():
        return None
    path = next(x for x in paths if x.exists())
    df = pd.read_csv(path, dtype={'GAME_ID': str, 'PLAYER_ID': str, 'TEAM_ID': str}, low_memory=False)
    required = {'GAME_ID','PLAYER_ID','TEAM_ID','MIN','PTS','REB','AST','FG3M'}
    if not required.issubset(df):
        raise ValueError('Incomplete player results: '+str(sorted(required-set(df))))
    df['GAME_ID'] = df.GAME_ID.astype(str).str.strip().str.removesuffix('.0').str.zfill(10)
    df['PLAYER_ID'] = df.PLAYER_ID.astype(str).str.strip().str.removesuffix('.0')
    return df


def settlement_for(rec, root, team_cache, player_data):
    path = Path(root)/'data'/f"nba_{ml._season(rec['game_date'])}_latest_raw.csv"
    if path not in team_cache:
        team_cache[path] = ml._load_results(path)
    games = team_cache[path].get(rec['game_key'], [])
    numeric_id = str(rec['source_game_id'])
    if numeric_id.isdecimal():
        games = [g for g in games if g['nba_game_id'] == numeric_id.zfill(10)]
    if len(games) != 1:
        return None
    game = games[0]
    pstatus = result_for_prop(rec, game, player_data)
    # Moneyline settles as soon as team results are verified; prop may remain pending.
    return {
        'schema': SCHEMA, 'game_key': rec['game_key'], 'record_hash': rec['record_hash'],
        'nba_game_id_verified': game['nba_game_id'], 'winner': game['winner'],
        'ml_result': 'WIN' if game['winner'] == rec['moneyline']['team'] else 'LOSS',
        'prop_result': pstatus, 'home_pts': game['home_pts'], 'away_pts': game['away_pts'],
        'settled_at_utc': datetime.now(timezone.utc).isoformat(),
    }


def settle(root=ROOT):
    root = Path(root)
    cfg = init(root)
    paths = sorted((root/STORE/'picks').glob('*.json'))
    if not paths:
        return {'newly_settled': 0, 'pending': 0}
    team_cache = {}
    player_data = load_player_results(root)
    new = 0; pending = 0
    for path in paths:
        rec = read(path)
        if rec.get('record_hash') != digest({k:v for k,v in rec.items() if k!='record_hash'}):
            raise ValueError(f'Frozen pick checksum mismatch: {path}')
        if rec.get('config_hash') != cfg['config_hash']:
            raise ValueError('Comparison uses unknown strategy config')
        output = root/STORE/'settlements'/path.name
        current = settlement_for(rec, root, team_cache, player_data)
        if current is None:
            pending += 1
            continue
        if output.exists():
            old = read(output)
            for key in ('record_hash', 'game_key', 'nba_game_id_verified', 'winner', 'ml_result', 'home_pts', 'away_pts'):
                if current[key] != old.get(key):
                    raise ValueError(f'Conflicting settlement for {path}: no silent revision')
            prev_prop, next_prop = old.get('prop_result'), current['prop_result']
            if prev_prop not in (None, 'PENDING') and next_prop != prev_prop:
                if next_prop == 'PENDING':
                    continue  # Older player data must not revert a settled prop.
                raise ValueError(f'Conflicting player settlement for {path}: no silent revision')
            if prev_prop in (None, 'PENDING') and next_prop not in (None, 'PENDING'):
                ml._atomic(output, current)
                new += 1
        elif freeze(output, current):
            new += 1
    return {'newly_settled': new, 'pending': pending}


def leg_return(leg, outcome):
    if outcome == 'PUSH':
        return 0.0
    return (1/leg['yes_ask']-1) if outcome == 'WIN' else -1.0


def _totals(rows):
    totals = {}
    returns = {}
    for name in STRATEGIES:
        totals[name] = {'strategy': name, 'selected': 0, 'settled': 0, 'pending': 0,
                        'skipped': 0, 'positive_return_games': 0, 'net_units': 0.0,
                        'paper_roi': None, 'max_drawdown_units': None, 'stake_units_settled': 0.0}
        returns[name] = []
    for rec, settlement in rows:
        for name, legs in rec['strategies'].items():
            if not legs:
                totals[name]['skipped'] += 1
                continue
            out = totals[name]
            out['selected'] += 1
            if settlement is None:
                out['pending'] += 1
                continue
            result = {'moneyline': settlement['ml_result'], 'prop': settlement['prop_result']}
            if any(result.get(leg['kind']) not in ('WIN', 'LOSS', 'PUSH') for leg in legs):
                out['pending'] += 1
                continue
            total_ret = 0.0
            for selection in legs:
                kind = selection['kind']
                leg = rec['moneyline'] if kind == 'moneyline' else rec['prop']
                total_ret += selection['stake'] * leg_return(leg, result[kind])
            out['settled'] += 1
            out['stake_units_settled'] += sum(leg['stake'] for leg in legs)
            out['positive_return_games'] += int(total_ret > 0)
            out['net_units'] += total_ret
            returns[name].append((rec['tipoff_utc'], rec['game_key'], total_ret))
    for name, out in totals.items():
        out['net_units'] = round(out['net_units'], 6)
        if out['settled']:
            out['paper_roi'] = round(out['net_units']/out['stake_units_settled'], 6)
            out['positive_return_game_rate'] = round(out['positive_return_games']/out['settled'], 6)
            equity = 0; peak = 0; dd = 0
            for _,_,ret in sorted(returns[name]):
                equity += ret; peak = max(peak, equity); dd = max(dd, peak-equity)
            out['max_drawdown_units'] = round(dd, 6)
        else:
            out['positive_return_game_rate'] = None
    return totals


def report(root=ROOT):
    root=Path(root)
    configpath = root/STORE/CONFIG
    if not configpath.exists():
        return {'status': 'NOT_INITIALIZED', 'paper_only': True}
    cfg=init(root)
    paths=sorted((root/STORE/'picks').glob('*.json'))
    rows=[]; latest=[]
    for path in paths:
        rec=read(path)
        if rec.get('record_hash') != digest({k:v for k,v in rec.items() if k!='record_hash'}):
            raise ValueError(f'Frozen pick checksum mismatch: {path}')
        if rec.get('config_hash')!=cfg['config_hash']:
            raise ValueError('Frozen selection checksum does not match config')
        result_path=root/STORE/'settlements'/path.name
        settled=read(result_path) if result_path.exists() else None
        if settled and settled.get('record_hash')!=rec['record_hash']:
            raise ValueError('Settlement/pick hash mismatch')
        rows.append((rec, settled))
        latest.append({
            'game_key':rec['game_key'], 'matchup':f"{rec['away']} @ {rec['home']}",
            'date':rec['game_date'], 'captured_at_utc':rec['captured_at_utc'],
            'moneyline':rec['moneyline'], 'prop':rec['prop'],
            'prop_status':rec['prop_status'], 'strategies':rec['strategies'],
            'ml_result':settled['ml_result'] if settled else 'PENDING',
            'prop_result':settled['prop_result'] if settled and rec['prop'] else ('N/A' if not rec['prop'] else 'PENDING'),
        })
    paired=[row for row in rows if row[0]['prop'] is not None]
    all_stats=_totals(rows)
    paired_stats=_totals(paired)
    return {
        'schema':SCHEMA, 'status':'TRACKING' if paths else 'WAITING_FOR_COMMON_PREGAME_SNAPSHOT',
        'paper_only':True, 'frozen_at_utc':cfg['frozen_at_utc'], 'strategy_hash':cfg['config_hash'],
        'captured_games':len(paths), 'games_with_both_quotes':len(paired),
        'strategies':list(all_stats.values()), 'paired_games_strategies':list(paired_stats.values()),
        'recent_games':sorted(latest,key=lambda x:(x['date'],x['game_key']),reverse=True)[:60],
        'notice':('Shared-snapshot prospective paper comparison. Moneyline always picks Elo winner. '
                  'Player prop candidates are the existing positive stressed-EV shortlist on either team; '
                  'other props are not represented. All strategies have at most 1 paper unit per selected game. '
                  'Both means two 0.5-unit SINGLE bets, not a parlay. No real fills/fees/slippage. '
                  'Model EVs not independently calibrated. Missing prop or player results remain non-executed/pending; '
                  'no outcomes or historical quotes are backfilled.'),
    }


def on_live_refresh(snapshot, root=ROOT):
    cap = capture(snapshot, root)
    result = settle(root)
    summary = report(root)
    summary['newly_captured'] = cap['new']
    summary['unavailable_live_games'] = cap['unavailable']
    summary['newly_settled'] = result['newly_settled']
    return summary


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['init','report','settle','refresh-results'], nargs='?', default='report')
    parser.add_argument('--root', type=Path, default=ROOT)
    args=parser.parse_args(argv)
    root=args.root
    if args.action=='init':
        data=init(root)
    elif args.action=='settle':
        data={'settlement':settle(root),'report':report(root)}
    elif args.action=='refresh-results':
        # Explicit network fetch, never invoked from read-only report.
        from .predict_today import fetch_current_season
        seasons={f"{day[:4] if int(day[5:7])>=7 else int(day[:4])-1}-{((int(day[:4]) if int(day[5:7])>=7 else int(day[:4])-1)+1)%100:02d}"
                 for path in (Path(root)/STORE/'picks').glob('*.json') for day in [read(path)['game_date']]}
        for season in sorted(seasons):
            print('Refreshing completed NBA team history', season, flush=True)
            fetch_current_season(season,None)
        if seasons:
            from .auto_live import BASE_HISTORY, OVERLAY, ET
            from .live_player_data import update_live_history
            today=datetime.now(ET).date()
            current_season=f'{today.year}-{(today.year+1)%100:02d}' if today.month>=7 else f'{today.year-1}-{today.year%100:02d}'
            if current_season in seasons and BASE_HISTORY.exists():
                print('Refreshing completed NBA player history', current_season, flush=True)
                update_live_history(BASE_HISTORY,OVERLAY,current_season,today)
        data={'settlement':settle(root),'report':report(root)}
    else:
        data=report(root)
    print(json.dumps(data,indent=2,allow_nan=False))
    return 0


if __name__=='__main__':
    raise SystemExit(main())
