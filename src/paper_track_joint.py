"""Auditable, read-only paper research tracker for Parlay Guard.

Freeze one pre-tipoff research slate per NBA date. Immutable predictions and
observed single-contract asks are never edited after capture. Settlement is a
SEPARATE file generated only from complete, exact NBA team game IDs and
verified player appearances. No bets are placed, Combo quotes inferred, or
profits claimed from unexecutable parlay prices.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import tempfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

STORE = Path('data/paper_trades/joint_tracker')
SCHEMA = 1
STATS = {'PTS', 'REB', 'AST', 'FG3M'}
STRATEGIES = ('single_prop', 'cross_game_props', 'same_game_prop_line', 'cross_game_prop_line')


def _time(value):
    return pd.to_datetime(value, utc=True, errors='coerce')


def _number(value):
    try:
        n = float(value)
        return n if math.isfinite(n) else None
    except (ValueError, TypeError):
        return None


def _name(value):
    return re.sub(r'[^a-z0-9]', '', str(value).casefold())


def _gameid(value):
    return str(value).strip().removesuffix('.0').zfill(10)


def _playerid(value):
    number = _number(value)
    return str(int(number)) if number is not None and number >= 1 and number == int(number) else None


def _json_read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def _atomic_update(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, name = tempfile.mkstemp(prefix='.pending-', suffix='.json', dir=path.parent)
    try:
        with os.fdopen(handle, 'w', encoding='utf-8') as f:
            json.dump(obj, f, indent=2, sort_keys=True, allow_nan=False)
            f.write('\n')
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def _freeze(path: Path, obj):
    """Exclusive creation: repeated refreshes cannot overwrite a pregame slate."""
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open('x', encoding='utf-8') as f:
            json.dump(obj, f, indent=2, sort_keys=True, allow_nan=False)
            f.write('\n')
        return True
    except FileExistsError:
        return False


def _key_for_prop(prop):
    return (prop['game'], _name(prop['player']), prop['stat'].upper(),
            prop['side'].upper(), float(prop['line']))


def _feature_ids(root: Path):
    path = root/'data/current_player_features.csv'
    if not path.is_file():
        return {}
    df = pd.read_csv(path, usecols=['event_slug', 'player_name', 'PLAYER_ID'], low_memory=False)
    lookup = defaultdict(set)
    for row in df.itertuples(index=False):
        pid = _playerid(row.PLAYER_ID)
        if pid:
            lookup[(str(row.event_slug), _name(row.player_name))].add(pid)
    return {key: next(iter(ids)) for key, ids in lookup.items() if len(ids) == 1}


def _prop_leg(prop, features):
    try:
        game = str(prop['game'])
        player = str(prop['player'])
        stat, side = str(prop['stat']).upper(), str(prop['side']).upper()
        line = _number(prop['line'])
        pid = features.get((game, _name(player)))
        if not pid or stat not in STATS or side not in ('OVER', 'UNDER') or line is None:
            return None
        return {'kind': 'prop', 'game': game, 'player': player, 'player_id': pid,
                'team': str(prop['team']).upper(), 'stat': stat, 'side': side, 'line': line}
    except (KeyError, TypeError, ValueError):
        return None


def _line_leg(game, kind, description):
    match = re.fullmatch(r'([A-Za-z]{2,4})\s+([+-]\d+(?:\.\d+)?)', str(description)) if kind == 'spread' else None
    if kind == 'spread':
        if not match or match.group(1).upper() not in (game['home'], game['away']):
            return None
        return {'kind': kind, 'game': game['slug'], 'team': match.group(1).upper(),
                'line': float(match.group(2))}
    match = re.fullmatch(r'(OVER|UNDER)\s+(\d+(?:\.\d+)?)', str(description))
    if kind == 'total' and match:
        return {'kind': kind, 'game': game['slug'], 'side': match.group(1),
                'line': float(match.group(2))}
    return None


def _ticket(strategy, legs, probability=None, reference_price=None, price_basis='unpriced'):
    data = {'strategy': strategy, 'legs': legs,
            'model_probability_uncalibrated': _number(probability),
            'reference_price': _number(reference_price), 'price_basis': price_basis,
            'is_executable_quote': False}
    raw = json.dumps({'strategy': strategy, 'legs': legs}, sort_keys=True)
    data['id'] = hashlib.sha256(raw.encode()).hexdigest()[:20]
    return data


def create_pregame_slate(snapshot: dict, nba_date: str, root: Path, *, at=None):
    """Validate inputs and build a frozen record, or return None if not safe."""
    captured = _time(at or datetime.now(timezone.utc))
    updated = _time(snapshot.get('updated_at_utc'))
    if pd.isna(captured) or pd.isna(updated) or not (updated <= captured < updated + pd.Timedelta(minutes=20)):
        return None
    if (snapshot.get('refresh_status') or {}).get('stale'):
        return None
    all_games = {}
    for g in snapshot.get('games') or []:
        if g.get('game_date') != nba_date:
            continue
        tip = _time(g.get('tipoff_utc'))
        win = g.get('winner_forecast') or {}
        gameid = str(win.get('nba_game_id') or '').strip()
        # ESPN_FALLBACK_GAME_IDS_V1
        # The ESPN schedule ID is *not* an NBA GAME_ID. Freeze it as source
        # metadata and resolve the NBA ID only from completed, paired team logs.
        if pd.isna(tip) or not captured < tip or not re.fullmatch(r'(?:\d+|ESPN-\d+)', gameid):
            continue
        if not win.get('predicted_winner'):
            continue
        away, home = str(g.get('away', '')).upper(), str(g.get('home', '')).upper()
        if away == home or not away or not home:
            continue
        official_nba_id = _gameid(gameid) if gameid.isdecimal() else None
        all_games[g['slug']] = {'slug': str(g['slug']), 'game_date': nba_date,
                                'nba_game_id': official_nba_id,
                                'source_game_id': gameid,
                                'source_provider': 'NBA' if official_nba_id else 'ESPN',
                                'home': home, 'away': away,
                                'tipoff_utc': tip.isoformat()}
    if not all_games:
        return None
    ids = _feature_ids(root)
    singles, available = [], {}
    for g in snapshot.get('games') or []:
        if g.get('slug') not in all_games:
            continue
        win_team = str((g.get('winner_forecast') or {}).get('predicted_winner')).upper()
        for pick in g.get('selected_props') or []:
            p = _number(pick.get('model_probability'))
            ask = _number(pick.get('ask'))
            if str(pick.get('team')).upper() != win_team or p is None or ask is None or not (0 < p < 1 and 0 < ask < 1):
                continue
            candidate = {'game': g['slug'], 'team': win_team, 'player': pick.get('player'),
                         'stat': pick.get('stat'), 'side': pick.get('side'), 'line': pick.get('line')}
            leg = _prop_leg(candidate, ids)
            if leg is None:
                continue
            key = _key_for_prop(candidate)
            if key in available:
                continue
            available[key] = (leg, p, ask)
            singles.append(_ticket('single_prop', [leg], p, ask, 'observed_individual_best_ask_paper_only'))
    if not singles:
        return None
    joint = snapshot.get('joint_research') or {}
    tickets = singles[:]
    # Only same-date event pairs; real combo pricing is never inferred.
    for row in joint.get('cross_game') or []:
        legs = row.get('legs') or []
        if len(legs) != 2:
            continue
        try:
            left, right = [available.get(_key_for_prop(x)) for x in legs]
        except (KeyError, TypeError, ValueError):
            continue
        if not left or not right or left[0]['game'] == right[0]['game']:
            continue
        tickets.append(_ticket('cross_game_props', [left[0], right[0]], row.get('joint_probability'),
                               left[2] * right[2], 'synthetic_product_not_executable'))
    for field, strategy in (('same_game', 'same_game_prop_line'), ('cross_game_line', 'cross_game_prop_line')):
        for row in joint.get(field) or []:
            if row.get('joint_probability') is None:
                continue
            kind = str(row.get('kind', '')).removeprefix('prop_plus_')
            if kind not in ('spread', 'total'):
                continue
            pg = row.get('game') if field == 'same_game' else row.get('prop_game')
            lg = row.get('game') if field == 'same_game' else row.get('line_game')
            if pg not in all_games or lg not in all_games:
                continue
            player = row.get('player')
            prop_description = re.fullmatch(r'([A-Za-z0-9]+)\s+(OVER|UNDER)\s+(\d+(?:\.\d+)?)', str(row.get('prop', '')))
            if not prop_description:
                continue
            stat, side, line = prop_description.group(1), prop_description.group(2), float(prop_description.group(3))
            source = available.get((pg, _name(player), stat, side, line))
            if not source:
                continue
            line_leg = _line_leg(all_games[lg], kind, row.get('game_line'))
            if line_leg is None or (strategy == 'cross_game_prop_line' and pg == lg):
                continue
            tickets.append(_ticket(strategy, [source[0], line_leg], row.get('joint_probability')))
    unique = {t['id']: t for t in tickets}
    return {'schema': SCHEMA, 'date': nba_date, 'captured_at_utc': captured.isoformat(),
            'source_updated_at_utc': updated.isoformat(), 'games': all_games,
            'tickets': list(unique.values()),
            'disclaimer': ('Immutable PRETIPOFF paper-research record, no orders or fills. '
                           'Single asks are observed quotes, NOT executions. Cross-game props use a '
                           'non-executable synthetic product; game-line Combo prices unavailable. '
                           'Modeled probabilities are not calibrated.')}


def capture_pregame_snapshots(snapshot: dict, root: Path, *, at=None):
    """Auto-record one eligible, immutable slate per date (never overwrite)."""
    root = Path(root)
    dates = sorted({g.get('game_date') for g in snapshot.get('games') or [] if g.get('game_date')})
    created = []
    for day in dates:
        path = root/STORE/'snapshots'/f'{day}.json'
        if path.exists():
            continue
        frozen = create_pregame_slate(snapshot, day, root, at=at)
        if frozen and _freeze(path, frozen):
            created.append(day)
    return created


def _team_games(path: Path):
    if not path.is_file():
        return None
    df = pd.read_csv(path, dtype={'GAME_ID': str})
    required = {'GAME_ID', 'GAME_DATE', 'TEAM_ABBREVIATION', 'PTS', 'WL'}
    if not required.issubset(df):
        raise ValueError('Current-season team scores missing '+str(sorted(required-set(df))))
    df['GAME_ID'] = df['GAME_ID'].map(_gameid)
    df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'], errors='coerce').dt.strftime('%Y-%m-%d')
    df['PTS'] = pd.to_numeric(df['PTS'], errors='coerce')
    df['TEAM_ABBREVIATION'] = df['TEAM_ABBREVIATION'].astype(str).str.upper().str.strip()
    df['WL'] = df['WL'].astype(str).str.upper().str.strip()
    return df


def _player_games(path: Path):
    if not path.is_file():
        return None
    df = pd.read_csv(path, low_memory=False, dtype={'GAME_ID': str, 'PLAYER_ID': str, 'TEAM_ID': str})
    needed = {'GAME_ID', 'PLAYER_ID', 'TEAM_ID', 'MIN', 'PTS', 'REB', 'AST', 'FG3M'}
    if not needed.issubset(df):
        raise ValueError('Player game results missing '+str(sorted(needed-set(df))))
    df['GAME_ID'] = df['GAME_ID'].map(_gameid)
    df['PLAYER_ID'] = df['PLAYER_ID'].map(_playerid)
    df['TEAM_ID'] = df['TEAM_ID'].map(_playerid)
    for col in ('MIN', 'PTS', 'REB', 'AST', 'FG3M'):
        df[col] = pd.to_numeric(df[col], errors='coerce')
    return df


def _complete_game(info, team_games):
    """Return (scores, verified NBA GAME_ID), or None if missing/ambiguous.

    An ESPN source identifier is never converted into an NBA ID by stripping
    digits. For ESPN schedules we resolve an NBA ID via a unique, completed
    two-team record on the exact NBA date, then use that ID for player logs.
    """
    if team_games is None:
        return None
    teams = {info['home'], info['away']}
    official_id = info.get('nba_game_id')
    if official_id:
        candidates = [(official_id, team_games[team_games['GAME_ID'] == official_id])]
    elif re.fullmatch(r'ESPN-\d+', str(info.get('source_game_id') or '')):
        same_date_teams = team_games[
            (team_games['GAME_DATE'] == info['game_date']) &
            (team_games['TEAM_ABBREVIATION'].isin(teams))
        ]
        candidates = list(same_date_teams.groupby('GAME_ID'))
    else:
        return None

    verified = []
    for nba_id, rows in candidates:
        if len(rows) != 2 or set(rows['TEAM_ABBREVIATION']) != teams:
            continue
        if set(rows['GAME_DATE']) != {info['game_date']} or set(rows['WL']) != {'W', 'L'} or not rows['PTS'].notna().all():
            continue
        scores = {str(r.TEAM_ABBREVIATION): float(r.PTS) for r in rows.itertuples(index=False)}
        if scores[info['home']] == scores[info['away']]:
            continue
        winner = max(scores, key=scores.get)
        recorded_winner = rows.loc[rows['WL'] == 'W', 'TEAM_ABBREVIATION'].iloc[0]
        if winner != recorded_winner:
            continue
        verified.append((scores, str(nba_id)))
    return verified[0] if len(verified) == 1 else None


def _settle_leg(leg, games, team_data, player_data):
    info = games.get(leg['game'])
    if not info:
        return 'PENDING'
    verified = _complete_game(info, team_data)
    if verified is None:
        return 'PENDING'
    scores, verified_nba_game_id = verified
    kind = leg['kind']
    if kind == 'prop':
        if player_data is None:
            return 'PENDING'
        from .player_prop_common import TEAM_ID_TO_ABBR
        rows = player_data[(player_data['GAME_ID'] == verified_nba_game_id) &
                           (player_data['PLAYER_ID'] == leg['player_id'])]
        if len(rows) != 1:
            return 'PENDING'  # missing player is NOT automatically an under win / no-show loss
        rec = rows.iloc[0]
        team_code = TEAM_ID_TO_ABBR.get(int(rec['TEAM_ID'])) if rec['TEAM_ID'] else None
        if str(team_code).upper() != leg['team']:
            return 'PENDING'
        minutes, actual = _number(rec['MIN']), _number(rec[leg['stat']])
        if minutes is None or minutes <= 0 or actual is None:
            return 'PENDING'
        diff = actual - leg['line']
        if abs(diff) < 1e-9:
            return 'PUSH'
        return 'WIN' if (diff > 0 if leg['side'] == 'OVER' else diff < 0) else 'LOSS'
    if kind == 'spread':
        team = leg['team']
        diff = scores[team] - scores[info['away'] if team == info['home'] else info['home']] + leg['line']
        if abs(diff) < 1e-9:
            return 'PUSH'
        return 'WIN' if diff > 0 else 'LOSS'
    if kind == 'total':
        diff = sum(scores.values()) - leg['line']
        if abs(diff) < 1e-9:
            return 'PUSH'
        return 'WIN' if (diff > 0 if leg['side'] == 'OVER' else diff < 0) else 'LOSS'
    return 'PENDING'


def settle_slate(frozen: dict, team_data, player_data):
    games = frozen['games']
    cache = {}
    outcomes = []
    for ticket in frozen['tickets']:
        leg_results = []
        for leg in ticket['legs']:
            key = json.dumps(leg, sort_keys=True)
            if key not in cache:
                cache[key] = _settle_leg(leg, games, team_data, player_data)
            leg_results.append(cache[key])
        if 'PENDING' in leg_results:
            status = 'PENDING'
        elif 'PUSH' in leg_results:
            status = 'PUSH'
        else:
            status = 'WIN' if all(x == 'WIN' for x in leg_results) else 'LOSS'
        outcomes.append({'ticket_id': ticket['id'], 'strategy': ticket['strategy'],
                         'result': status, 'leg_results': leg_results})
    return {'schema': SCHEMA, 'slate_date': frozen['date'],
            'captured_at_utc': frozen['captured_at_utc'],
            'settled_as_of_utc': datetime.now(timezone.utc).isoformat(),
            'results': outcomes}


def settle_pending(root: Path, *, team_csv=None, player_csv=None):
    root = Path(root)
    snapshots = sorted((root/STORE/'snapshots').glob('*.json'))
    if not snapshots:
        return 0
    player_path = Path(player_csv) if player_csv else root/'data/live_player_training_data_v2.csv'
    player_data = _player_games(player_path)
    count = 0
    team_cache = {}
    for path in snapshots:
        frozen = _json_read(path)
        season = frozen['date'][:4] if int(frozen['date'][5:7]) >= 7 else str(int(frozen['date'][:4]) - 1)
        last_year = (int(season) + 1) % 100
        team_path = Path(team_csv) if team_csv else root/f'data/nba_{season}_{last_year:02d}_latest_raw.csv'
        if team_path not in team_cache:
            team_cache[team_path] = _team_games(team_path)
        settled = settle_slate(frozen, team_cache[team_path], player_data)
        out = root/STORE/'settlements'/path.name
        # Settlement sidecars may update when genuinely new completed results arrive.
        previous = _json_read(out) if out.is_file() else None
        normalized = lambda x: [(i['ticket_id'], i['result'], tuple(i['leg_results'])) for i in x['results']]
        if previous is None or normalized(previous) != normalized(settled):
            _atomic_update(out, settled)
            count += 1
    return count


def performance_summary(root: Path):
    root = Path(root)
    snapshots = sorted((root/STORE/'snapshots').glob('*.json'))
    total = {s: {'strategy': s, 'slates_with_tickets': 0, 'fully_settled_slates': 0,
                 'tickets': 0, 'wins': 0, 'losses': 0, 'pending': 0,
                 'pushes': 0, 'hit_rate': None, 'mean_brier': None,
                 'mean_paper_reference_roi': None, 'price_basis': None} for s in STRATEGIES}
    briers = defaultdict(list)
    slate_rois = defaultdict(list)
    pending_dates = []
    for path in snapshots:
        frozen = _json_read(path)
        settled_path = root/STORE/'settlements'/path.name
        results = {r['ticket_id']: r['result'] for r in _json_read(settled_path)['results']} if settled_path.is_file() else {}
        by_strategy = defaultdict(list)
        for ticket in frozen['tickets']:
            by_strategy[ticket['strategy']].append(ticket)
        slate_unsettled = False
        for strategy, tickets in by_strategy.items():
            row = total[strategy]
            row['slates_with_tickets'] += 1
            paper_returns = []
            complete = True
            for t in tickets:
                row['tickets'] += 1
                result = results.get(t['id'], 'PENDING')
                if result == 'WIN':
                    row['wins'] += 1
                elif result == 'LOSS':
                    row['losses'] += 1
                elif result == 'PUSH':
                    row['pushes'] += 1
                    complete = False
                else:
                    row['pending'] += 1
                    complete = False
                    slate_unsettled = True
                p = _number(t.get('model_probability_uncalibrated'))
                if result in ('WIN', 'LOSS') and p is not None:
                    briers[strategy].append((p-(1 if result == 'WIN' else 0))**2)
                price = _number(t.get('reference_price'))
                if result in ('WIN', 'LOSS') and price is not None and 0 < price < 1:
                    paper_returns.append((1/price - 1) if result == 'WIN' else -1.0)
                else:
                    complete = False if strategy in ('single_prop', 'cross_game_props') else complete
            if all(results.get(t['id']) in ('WIN', 'LOSS') for t in tickets):
                row['fully_settled_slates'] += 1
            if complete and len(paper_returns) == len(tickets) and strategy in ('single_prop', 'cross_game_props'):
                # Same total budget split equally across all tickets in the slate.
                slate_rois[strategy].append(sum(paper_returns)/len(paper_returns))
                row['price_basis'] = ('observed single best-ask paper reference' if strategy == 'single_prop'
                                      else 'NONEXECUTABLE synthetic product of single asks')
        if slate_unsettled:
            pending_dates.append(frozen['date'])
    for strategy, row in total.items():
        resolved = row['wins'] + row['losses']
        if resolved:
            row['hit_rate'] = round(row['wins']/resolved, 5)
        if briers[strategy]:
            row['mean_brier'] = round(sum(briers[strategy])/len(briers[strategy]), 5)
        if slate_rois[strategy]:
            row['mean_paper_reference_roi'] = round(sum(slate_rois[strategy])/len(slate_rois[strategy]), 5)
    return {'schema': SCHEMA, 'recorded_slates': len(snapshots), 'strategies': list(total.values()),
            'pending_dates': sorted(set(pending_dates)),
            'notice': ('Only fully resolved slates contribute to equal-budget paper-reference ROI. '
                       'Single best asks are not confirmed fills. Cross-game prop-product prices are NOT executable. '
                       'Same/cross-game prop+line strategies have NO price/ROI; hit rate and Brier only. '
                       'Missing player logs/no-shows remain PENDING; no wagers are placed.')}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['report', 'settle'], nargs='?', default='report')
    parser.add_argument('--root', type=Path, default=Path.cwd())
    parser.add_argument('--team-csv', type=Path)
    parser.add_argument('--player-csv', type=Path)
    args = parser.parse_args(argv)
    if args.action == 'settle':
        print('Updated settlement sidecars:', settle_pending(args.root, team_csv=args.team_csv, player_csv=args.player_csv))
    print(json.dumps(performance_summary(args.root), indent=2))


if __name__ == '__main__':
    main()
