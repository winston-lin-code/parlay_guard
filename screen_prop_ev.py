"""Research-only screening of Parlay Guard Polymarket NBA player-prop snapshots.

Reads data/polymarket_nba_ev.csv, filters quote quality and basic research
risks, and saves data/prop_ev_research_shortlist.csv. Does NOT make bets.

Example (update injury exclusions before each use):
  python screen_prop_ev.py --exclude "Deni Avdija,Toumani Camara" --stake 20

No-show probabilities, real fees, limits, correlations, and calibration are
NOT estimated by this script. Any remaining EV is exploratory.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_INPUT = Path('data/polymarket_nba_ev.csv')
DEFAULT_OUTPUT = Path('data/prop_ev_research_shortlist.csv')
REQUIRED = {
    'observed_utc', 'game_start_utc', 'event', 'event_slug', 'market_id',
    'player_name', 'stat', 'line', 'side', 'outcome', 'outcome_mapping',
    'ask_price', 'ask_size_shares', 'bid_price', 'book_status',
    'model_probability', 'estimated_ev_per_dollar',
}


def screen(csv_path: Path, output_path: Path, *, stake: float = 20.,
           max_age_min: float = 90., max_spread: float = 0.15,
           min_ask: float = 0.15, max_ask: float = 0.85,
           min_ev: float = 0.10, fee_buffer: float = 0.01,
           probability_haircut: float = 0.05,
           excluded_players=(), now=None) -> tuple[pd.DataFrame, dict]:
    df = pd.read_csv(csv_path)
    missing = REQUIRED - set(df.columns)
    if missing:
        raise ValueError(f'Missing expected columns in {csv_path}: {sorted(missing)}. '
                         'Use the newer polymarket_nba_ev.py scanner.')
    now = pd.Timestamp.now(tz='UTC') if now is None else pd.Timestamp(now)
    if now.tzinfo is None:
        now = now.tz_localize('UTC')
    else:
        now = now.tz_convert('UTC')

    for column in ['ask_price', 'ask_size_shares', 'bid_price',
                   'model_probability', 'estimated_ev_per_dollar']:
        df[column] = pd.to_numeric(df[column], errors='coerce')
    df['observed_utc'] = pd.to_datetime(df['observed_utc'], errors='coerce', utc=True)
    df['game_start_utc'] = pd.to_datetime(df['game_start_utc'], errors='coerce', utc=True)
    initial = len(df)
    stats = {'initial_outcomes': initial}
    def apply(name, mask):
        nonlocal df
        df = df.loc[mask].copy()
        stats[name] = len(df)

    apply('future_game', df['game_start_utc'].notna() & (df['game_start_utc'] > now))
    apply('fresh_quote', df['observed_utc'].notna()
          & ((now - df['observed_utc']) >= pd.Timedelta(0))
          & ((now - df['observed_utc']) <= pd.Timedelta(minutes=max_age_min)))
    apply('settlement_verified', df['outcome_mapping'].eq('description_yes_over')
          & df['side'].isin(['OVER', 'UNDER'])
          & ((df['outcome'].eq('YES') & df['side'].eq('OVER'))
             | (df['outcome'].eq('NO') & df['side'].eq('UNDER'))))
    apply('priced_with_model', df['book_status'].eq('priced')
          & df['model_probability'].between(0, 1, inclusive='both')
          & df['estimated_ev_per_dollar'].notna())
    apply('price_range', df['ask_price'].between(min_ask, max_ask))
    df['spread'] = df['ask_price'] - df['bid_price']
    apply('spread_limit', df['bid_price'].notna()
          & df['spread'].between(0, max_spread))
    # CLOB ask sizes represent SHARES. At top ask price p, spending stake
    # dollars requires stake / p shares. Do not mistake shares for dollars.
    df['top_ask_dollars'] = df['ask_price'] * df['ask_size_shares']
    apply('top_of_book_stake_capacity', df['top_ask_dollars'] >= stake)
    excluded = {str(p).strip().casefold() for p in excluded_players if str(p).strip()}
    apply('injury_exclusions', ~df['player_name'].astype(str).str.casefold().isin(excluded))
    # Stress test, NOT a calibration estimate: subtract five percentage points
    # and add a hypothetical per-share cost buffer to the acquisition price.
    df['stressed_probability'] = np.maximum(0., df['model_probability'] - probability_haircut)
    df['stressed_cost'] = df['ask_price'] + fee_buffer
    df['stressed_ev_roi'] = (df['stressed_probability'] - df['stressed_cost']) / df['stressed_cost']
    apply('stressed_ev_limit', (df['stressed_ev_roi'] >= min_ev) & (df['stressed_cost'] < 1))
    df['shares_at_top_for_stake'] = stake / df['ask_price']
    # For YES contracts only, this is a hypothetical break-even play probability
    # if the model P(YES) is conditional on playing and no-play settles NO.
    df['yes_break_even_play_probability'] = np.where(
        (df['outcome'] == 'YES') & (df['model_probability'] > 0),
        df['stressed_cost'] / df['model_probability'], np.nan)
    df = df.sort_values(['stressed_ev_roi', 'model_probability'], ascending=False)
    # Do not count alternative lines or different stats for the same player/game
    # as independent picks. Other cross-player correlations are still present.
    df = df.drop_duplicates(subset=['event_slug', 'player_name'], keep='first')
    stats['one_per_player_per_game'] = len(df)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return df, stats


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--input', type=Path, default=DEFAULT_INPUT)
    ap.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    ap.add_argument('--stake', type=float, default=20., help='Target dollar spend at best ask, default $20')
    ap.add_argument('--max-age-min', type=float, default=90., help='Reject quotes older than 90 minutes')
    ap.add_argument('--max-spread', type=float, default=0.15, help='Max absolute bid/ask spread')
    ap.add_argument('--min-ask', type=float, default=0.15)
    ap.add_argument('--max-ask', type=float, default=0.85)
    ap.add_argument('--min-ev', type=float, default=0.10, help='Minimum stressed ROI, default 10%%')
    ap.add_argument('--fee-buffer', type=float, default=0.01, help='Hypothetical dollar/share cost buffer, NOT real fee')
    ap.add_argument('--probability-haircut', type=float, default=0.05,
                    help='Stress test model probability by -5 percentage points')
    ap.add_argument('--exclude', default='', help='Comma-separated players to exclude based on injury/availability review')
    args = ap.parse_args()
    if not args.input.exists():
        ap.error(f'Input not found: {args.input}')
    if args.stake <= 0 or args.max_age_min <= 0 or not 0 <= args.max_spread <= 1:
        ap.error('--stake and --max-age-min must be positive; --max-spread must be in [0,1]')
    if not 0 < args.min_ask < args.max_ask < 1 or not 0 <= args.fee_buffer < 1:
        ap.error('Invalid prices or fee buffer')
    if not 0 <= args.probability_haircut < 1:
        ap.error('Probability haircut must be in [0,1)')
    df, stats = screen(
        args.input, args.output, stake=args.stake, max_age_min=args.max_age_min,
        max_spread=args.max_spread, min_ask=args.min_ask, max_ask=args.max_ask,
        min_ev=args.min_ev, fee_buffer=args.fee_buffer,
        probability_haircut=args.probability_haircut,
        excluded_players=args.exclude.split(','))
    print('Screening outcomes:')
    for name, count in stats.items():
        print(f'  {name}: {count}')
    print(f'Saved: {args.output}')
    if df.empty:
        print('No markets pass. This can be normal; check quote age, liquidity, and injuries. Do not weaken filters just to get picks.')
    else:
        cols = ['event', 'player_name', 'stat', 'side', 'line', 'ask_price',
                'bid_price', 'spread', 'ask_size_shares', 'model_probability',
                'estimated_ev_per_dollar', 'stressed_ev_roi',
                'yes_break_even_play_probability']
        print('\nResearch-only shortlist (NOT validated bets):')
        print(df[cols].head(15).to_string(index=False, float_format=lambda x: f'{x:.3f}'))
    print('\nNOT checked: true injury probabilities, game roster, market fee schedule, model calibration, or parlay correlations.')


if __name__ == '__main__':
    main()
