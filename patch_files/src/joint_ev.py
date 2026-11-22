"""Research-only joint probability comparisons for Parlay Guard.

This module NEVER assumes a synthetic product of single-leg prices is an
executable parlay quote. Same-game estimates require empirical, pre-tipoff
historical correlation; game-line probabilities are exploratory normal-error
approximations from MAE, not calibrated cover probabilities.
"""
from __future__ import annotations

import math
import re
from collections import defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path
from statistics import NormalDist

import numpy as np
import pandas as pd

NORMAL = NormalDist()
MIN_CORRELATION_GAMES = 25
MAX_CORRELATION_GAMES = 100


def _finite(x):
    try:
        value = float(x)
        return value if math.isfinite(value) else None
    except (ValueError, TypeError):
        return None


def _key(x):
    return re.sub(r'[^a-z0-9]', '', str(x).lower())


def _game_id(x):
    return str(x).strip().removesuffix('.0').zfill(10)


def _utc(x):
    return pd.to_datetime(x, utc=True, errors='coerce')


def normal_cdf(x):
    return NORMAL.cdf(float(x))


def joint_gaussian(p_a: float, p_b: float, rho: float) -> float:
    """Gaussian copula P(A and B), using deterministic Plackett integration.

    The inputs are marginal event probabilities, *not* average point outputs.
    This is an assumption-based estimate, not an empirically calibrated joint
    probability. No scipy dependency is required.
    """
    p_a, p_b, rho = float(p_a), float(p_b), float(rho)
    if not 0 <= p_a <= 1 or not 0 <= p_b <= 1 or not -0.95 <= rho <= 0.95:
        raise ValueError('Invalid marginal probability or correlation')
    if p_a in (0, 1) or p_b in (0, 1):
        return max(0.0, p_a + p_b - 1) if p_a in (1,) or p_b in (1,) else 0.0
    if rho == 0:
        return p_a * p_b
    a, b = NORMAL.inv_cdf(p_a), NORMAL.inv_cdf(p_b)
    # Simpson's rule integrates d Phi_2(a,b,r) / dr from 0 to rho.
    steps = 120
    h = rho / steps
    integral = 0.0
    for i in range(steps + 1):
        r = i * h
        denom = 1 - r*r
        density = math.exp(-(a*a - 2*r*a*b + b*b)/(2*denom)) / (2*math.pi*math.sqrt(denom))
        integral += (1 if i in (0, steps) else 4 if i % 2 else 2) * density
    result = p_a*p_b + h*integral/3
    return round(min(min(p_a, p_b), max(max(0.0, p_a+p_b-1), result)), 6)


def estimate_correlation(samples: pd.DataFrame, x: str, y: str, *, min_n=MIN_CORRELATION_GAMES):
    """Return shrunken Pearson r + Fisher interval or None for sparse records."""
    if not {x, y}.issubset(samples):
        return None
    pairs = samples[[x, y]].replace([np.inf, -np.inf], np.nan).dropna().tail(MAX_CORRELATION_GAMES)
    n = len(pairs)
    if n < min_n or pairs[x].std() < 1e-8 or pairs[y].std() < 1e-8:
        return None
    raw = float(pairs[x].corr(pairs[y]))
    if not math.isfinite(raw):
        return None
    raw = max(-0.95, min(0.95, raw))
    shrink = n / (n + 50)
    r = raw * shrink
    z = math.atanh(raw)
    half = 1.96 / math.sqrt(n - 3)
    low = math.tanh(z - half)*shrink
    high = math.tanh(z + half)*shrink
    return {'n': n, 'raw_rho': round(raw, 4), 'shrunk_rho': round(r, 4),
            'rho_low': round(max(-0.95, low), 4),
            'rho_high': round(min(0.95, high), 4),
            'method': 'historical pregame player-stat and rolling team-line residuals; Fisher uncertainty + shrinkage',
            'calibrated': False}


def build_game_residuals(completed: pd.DataFrame) -> pd.DataFrame:
    """Rolling 10-game, same-date batched historical line residuals.

    Every residual baseline uses team games strictly before the completed game.
    No prior-day leakage from other games on the same NBA calendar date.
    """
    required = {'GAME_ID', 'GAME_DATE', 'HOME_TEAM', 'AWAY_TEAM', 'HOME_PTS', 'AWAY_PTS'}
    if not required.issubset(completed.columns):
        raise ValueError(f'Game history missing {sorted(required - set(completed.columns))}')
    games = completed.copy()
    games['GAME_DATE'] = pd.to_datetime(games['GAME_DATE']).dt.normalize()
    games = games.sort_values(['GAME_DATE', 'GAME_ID'])
    history = defaultdict(lambda: deque(maxlen=10))
    rows = []
    for day, daily in games.groupby('GAME_DATE', sort=True):
        updates = []
        for g in daily.itertuples(index=False):
            h, a = str(g.HOME_TEAM).upper(), str(g.AWAY_TEAM).upper()
            hp, ap = float(g.HOME_PTS), float(g.AWAY_PTS)
            if len(history[h]) == 10 and len(history[a]) == 10:
                home_for = np.mean([r[0] for r in history[h]])
                home_against = np.mean([r[1] for r in history[h]])
                away_for = np.mean([r[0] for r in history[a]])
                away_against = np.mean([r[1] for r in history[a]])
                projected_h = (home_for + away_against)/2
                projected_a = (away_for + home_against)/2
                rows.append({'GAME_ID': _game_id(g.GAME_ID), 'GAME_DATE': day,
                             'margin_residual': hp-ap-(projected_h-projected_a),
                             'total_residual': hp+ap-(projected_h+projected_a)})
            updates.append((h, a, hp, ap))
        for h, a, hp, ap in updates:
            history[h].append((hp, ap))
            history[a].append((ap, hp))
    return pd.DataFrame(rows, columns=['GAME_ID', 'GAME_DATE', 'margin_residual', 'total_residual'])


def _historical_player_residuals(history: pd.DataFrame, game_residuals: pd.DataFrame):
    # Historical player model features *_LAST_10 are pregame values.
    needed = {'GAME_ID', 'GAME_DATE', 'PLAYER_ID', 'PTS', 'REB', 'AST', 'FG3M',
              'PTS_LAST_10', 'REB_LAST_10', 'AST_LAST_10', 'FG3M_LAST_10'}
    if not needed.issubset(history):
        raise ValueError(f'Player history missing {sorted(needed - set(history))}')
    df = history[list(needed)].copy()
    df['GAME_ID'] = df['GAME_ID'].map(_game_id)
    df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'], errors='coerce').dt.normalize()
    df['PLAYER_ID'] = pd.to_numeric(df['PLAYER_ID'], errors='coerce')
    for stat in ('PTS', 'REB', 'AST', 'FG3M'):
        df[f'{stat}_residual'] = pd.to_numeric(df[stat], errors='coerce') - pd.to_numeric(df[f'{stat}_LAST_10'], errors='coerce')
    df = df.merge(game_residuals[['GAME_ID', 'margin_residual', 'total_residual']], on='GAME_ID', how='inner', validate='many_to_one')
    return df.sort_values(['GAME_DATE', 'GAME_ID'])


def _side_from_market(game, market):
    label = _key(market.get('team_label', ''))
    home, away = game['home'].upper(), game['away'].upper()
    from .auto_live import NICKNAMES
    def is_team(abbr):
        nickname = _key(NICKNAMES.get(abbr.lower(), ''))
        return label == abbr.lower() or (nickname and nickname in label)
    home_hit, away_hit = is_team(home), is_team(away)
    if home_hit == away_hit:
        return None
    return 'HOME' if home_hit else 'AWAY'


def _line_candidate(game, kind, sigma):
    """Return one game-line event with uncalibrated normal error probability."""
    forecast = game.get('game_line_prediction') or {}
    if not forecast or sigma is None or sigma <= 0:
        return None
    market = game.get('market_spread' if kind == 'spread' else 'market_total') or {}
    threshold = _finite(market.get('line'))
    if threshold is None:
        return None
    if kind == 'spread':
        where = _side_from_market(game, market)
        margin = _finite(forecast.get('home_margin'))
        if where is None or margin is None:
            return None
        # margin = home points minus away points; an offered team covers when
        # its points differential + offered spread > 0.
        probability = normal_cdf(((margin if where == 'HOME' else -margin) + threshold)/sigma)
        signal = 'margin_residual'
        return {'kind': kind, 'side': f"{game[where.lower()]} {threshold:+g}",
                'market_line': threshold, 'team_side': where, 'sigma': round(sigma, 3),
                'model_probability': round(probability, 5), 'residual': signal,
                'direction': 1 if where == 'HOME' else -1}
    total = _finite(forecast.get('projected_total'))
    if total is None:
        return None
    p_over = normal_cdf((total - threshold)/sigma)
    over = p_over >= 0.5
    return {'kind': kind, 'side': ('OVER ' if over else 'UNDER ') + f'{threshold:g}',
            'market_line': threshold, 'sigma': round(sigma, 3),
            'model_probability': round(max(p_over, 1-p_over), 5),
            'residual': 'total_residual', 'direction': 1 if over else -1}


def _sigma(metrics, kind):
    # For a centered normal, MAE = sigma * sqrt(2/pi); approximate only.
    key = 'home_margin_holdout_mae' if kind == 'spread' else 'total_holdout_mae'
    mae = _finite(metrics.get(key))
    holdout = metrics.get('holdout_games', 0)
    if mae is None or mae <= 0 or holdout < 50:
        return None
    return mae * math.sqrt(math.pi/2)


def build_joint_research(snapshot: dict, root: Path) -> dict:
    """Compare currently selected prop singles, cross-game pairs, prop+lines.

    Always return a safe object. When the required historical files are absent,
    same-game pair probability is *unavailable*, never defaulted to independence.
    """
    now = _utc(snapshot.get('updated_at_utc'))
    if pd.isna(now):
        now = pd.Timestamp.now(tz='UTC')
    games = snapshot.get('games') or []
    payload = {
        'generated_at_utc': now.isoformat(),
        'status': 'research_only',
        'disclaimer': ('Illustrative joint probabilities only. Game-line cover probabilities approximate '
                       '2025-26 holdout errors by a normal distribution and are NOT calibrated. '
                       'Historical player/game correlations can change and exclude injuries. '
                       'Combined asks are NOT executable Polymarket Combo quotes. No executable parlay EV is claimed.'),
        'singles': [], 'cross_game': [], 'cross_game_line': [], 'same_game': [], 'warnings': [],
    }
    selected = []
    for game in games:
        for pick in game.get('selected_props') or []:
            p = _finite(pick.get('model_probability'))
            ask = _finite(pick.get('ask'))
            if p is None or ask is None or not 0 < p < 1 or not 0 < ask < 1:
                continue
            stressed = max(0, p-0.05)
            item = {'game': game['slug'], 'team': pick['team'], 'player': pick['player'],
                    'stat': pick['stat'], 'side': pick['side'], 'line': pick['line'],
                    'model_probability': round(p, 5), 'stress_probability': round(stressed, 5),
                    'individual_ask': ask,
                    'stressed_single_roi': round(stressed/(ask+0.01)-1, 5)}
            selected.append(item)
            payload['singles'].append(item)
    # Cross-game independence is an ASSUMPTION; ranks are just research proposals.
    for i, left in enumerate(selected):
        for right in selected[i+1:]:
            if left['game'] == right['game']:
                continue
            p = round(left['stress_probability']*right['stress_probability'], 6)
            proxy = round(left['individual_ask']*right['individual_ask'], 6)
            payload['cross_game'].append({
                'kind': 'prop_plus_prop', 'legs': [left, right],
                'joint_probability': p, 'model': 'independence assumption (not calibrated)',
                'hypothetical_product_ask': proxy,
                'hypothetical_product_roi': round(p/proxy-1, 5),
                'breakeven_combo_ask': p,
                'executable_combo_ask': None, 'executable_ev_roi': None,
            })
    payload['cross_game'].sort(key=lambda v: v['hypothetical_product_roi'], reverse=True)
    payload['cross_game'] = payload['cross_game'][:12]
    if not selected:
        payload['warnings'].append('No winner-team positive-stress-EV props passed the current live screen.')
        return payload
    # Same-game game lines need holdout error, sufficient player history, and a
    # reliable mapping to an actual spread/total outcome.
    metrics = snapshot.get('line_model_metrics') or {}
    sigmas = {k: _sigma(metrics, k) for k in ('spread', 'total')}
    if all(v is None for v in sigmas.values()):
        payload['warnings'].append('No holdout MAE; cannot estimate game-line probability.')
        return payload
    # Cross-game prop+line pairs assume independence across two distinct games.
    # A real game-line ask is not available here, so ROI is intentionally NULL.
    for pick in selected:
        for other in games:
            if other['slug'] == pick['game']:
                continue
            for kind in ('spread', 'total'):
                line = _line_candidate(other, kind, sigmas[kind])
                if line is None:
                    continue
                p_joint = round(pick['stress_probability'] * line['model_probability'], 6)
                payload['cross_game_line'].append({
                    'kind': f'prop_plus_{kind}', 'prop_game': pick['game'],
                    'line_game': other['slug'], 'player': pick['player'],
                    'prop': f"{pick['stat']} {pick['side']} {pick['line']}",
                    'game_line': line['side'], 'joint_probability': p_joint,
                    'model': 'cross-game independence; game-line cover estimate uncalibrated',
                    'breakeven_combo_ask': p_joint, 'executable_combo_ask': None,
                    'executable_ev_roi': None,
                })
    payload['cross_game_line'].sort(key=lambda x: x['joint_probability'], reverse=True)
    payload['cross_game_line'] = payload['cross_game_line'][:12]
    historical_player = root/'data/live_player_training_data_v2.csv'
    if not historical_player.exists():
        historical_player = root/'data/processed/player_training_data_v2.csv'
    feature_path = root/'data/current_player_features.csv'
    if not historical_player.exists() or not feature_path.exists():
        payload['warnings'].append('Player history/features missing; same-game correlation unavailable.')
        return payload
    try:
        from .predict_today import load_history, fetch_current_season, clean_team_history, make_completed_games
        local_day = now.tz_convert('America/New_York').date()
        season = f'{local_day.year}-{(local_day.year+1)%100:02d}' if local_day.month >= 7 else f'{local_day.year-1}-{local_day.year%100:02d}'
        current = root/f'data/nba_{season.replace("-", "_")}_latest_raw.csv'
        if not current.exists():
            raise FileNotFoundError(current.name)
        old = load_history(root/'data/nba_games_updated.csv')
        current_games = fetch_current_season(season, current)
        completed = make_completed_games(clean_team_history(old, current_games, local_day))
        game_resids = build_game_residuals(completed)
        if game_resids.empty:
            raise ValueError('No historical game residuals with ten prior games')
        player_history = pd.read_csv(historical_player, low_memory=False, dtype={'GAME_ID': str})
        samples = _historical_player_residuals(player_history, game_resids)
        features = pd.read_csv(feature_path, usecols=['event_slug', 'player_name', 'PLAYER_ID'])
        ids = {(str(x.event_slug), _key(x.player_name)): x.PLAYER_ID for x in features.itertuples(index=False)}
    except (OSError, ValueError, KeyError, TypeError) as exc:
        payload['warnings'].append('Historical correlation unavailable: ' + str(exc)[:180])
        return payload
    by_player = {int(pid): grp for pid, grp in samples.groupby('PLAYER_ID') if pd.notna(pid)}
    for game in games:
        game_picks = [x for x in selected if x['game'] == game['slug']]
        if not game_picks:
            continue
        for kind in ('spread', 'total'):
            line = _line_candidate(game, kind, sigmas[kind])
            if line is None:
                continue
            for pick in game_picks:
                pid = ids.get((game['slug'], _key(pick['player'])))
                past = by_player.get(int(pid)) if pd.notna(pid) else None
                corr = None
                if past is not None:
                    game_day = pd.Timestamp(game['game_date'])
                    past = past[past['GAME_DATE'] < game_day].copy()
                    var = f"{pick['stat']}_residual"
                    if var in past.columns:
                        past[var] *= 1 if pick['side'].upper() == 'OVER' else -1
                        residual = line['residual']
                        past[residual] *= line['direction']
                        corr = estimate_correlation(past, var, residual)
                p_joint = interval = None
                if corr is not None:
                    prob_a, prob_b = pick['stress_probability'], line['model_probability']
                    p_joint = joint_gaussian(prob_a, prob_b, corr['shrunk_rho'])
                    interval = [joint_gaussian(prob_a, prob_b, corr['rho_low']),
                                joint_gaussian(prob_a, prob_b, corr['rho_high'])]
                payload['same_game'].append({
                    'kind': f"prop_plus_{kind}", 'game': game['slug'],
                    'player': pick['player'], 'prop': f"{pick['stat']} {pick['side']} {pick['line']}",
                    'game_line': line['side'], 'prop_probability_stressed': pick['stress_probability'],
                    'game_line_probability': line['model_probability'],
                    'joint_probability': p_joint, 'joint_sensitivity_range': interval,
                    'correlation': corr, 'breakeven_combo_ask': p_joint,
                    'executable_combo_ask': None, 'executable_ev_roi': None,
                    'status': 'ESTIMATED_UNCALIBRATED' if corr else 'INSUFFICIENT_CORRELATION_DATA',
                })
    payload['same_game'].sort(key=lambda x: (x['joint_probability'] is not None, x['joint_probability'] or 0), reverse=True)
    payload['same_game'] = payload['same_game'][:20]
    return payload
