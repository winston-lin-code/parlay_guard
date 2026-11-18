"""Score a proposed matchup against manually entered market probabilities.

Market inputs must be decimal probabilities, e.g. 0.57. No trading is performed.
"""
import argparse
import joblib
import pandas as pd
from .common import DATA, MODELS
from .merge_features import FEATURES


def recent_features(team, df):
    games = df[df['TEAM_ABBREVIATION'].eq(team)].sort_values(['GAME_DATE','GAME_ID'])
    if games.empty:
        raise ValueError(f'Unknown team {team!r}')
    latest = games.iloc[-1]
    values = {}
    for n in (5,10):
        prior = games.tail(n)
        values[f'WIN_PCT_LAST_{n}'] = float(prior['IS_WIN'].mean()) if len(prior) == n else float('nan')
    values['DAYS_REST'] = None
    return values, latest['GAME_DATE']


def main():
    parser = argparse.ArgumentParser(description='Manual NBA matchup value scanner')
    parser.add_argument('--home', required=True, help='NBA abbreviation, e.g. BOS')
    parser.add_argument('--away', required=True, help='NBA abbreviation, e.g. NYK')
    parser.add_argument('--date', required=True, help='Scheduled game date YYYY-MM-DD')
    parser.add_argument('--home-price', required=True, type=float, help='Market YES price for home win, 0 to 1')
    parser.add_argument('--away-price', type=float, help='Market YES price for away win, 0 to 1')
    parser.add_argument('--min-edge', type=float, default=0.05, help='Minimum probability edge, default 0.05')
    args = parser.parse_args()
    if args.home == args.away:
        parser.error('Home and away must be different')
    if not (0 < args.home_price < 1) or (args.away_price is not None and not 0 < args.away_price < 1):
        parser.error('Prices must be between 0 and 1')
    if not 0 <= args.min_edge < 1:
        parser.error('--min-edge must be between 0 and 1')
    game_date = pd.Timestamp(args.date)
    history = pd.read_csv(DATA / 'nba_team_features.csv', dtype={'GAME_ID':str})
    history['GAME_DATE'] = pd.to_datetime(history['GAME_DATE'])
    history = history[history['GAME_DATE'] < game_date].copy()
    if history.empty:
        raise ValueError('No history prior to requested game date')
    # Recalculate the last N *completed* games: the last historical row's stored
    # rolling values exclude that historical game and cannot be reused directly.
    row = {'HOME_TEAM':args.home, 'AWAY_TEAM':args.away}
    for side, team in [('HOME',args.home),('AWAY',args.away)]:
        values, last_date = recent_features(team, history)
        for key, val in values.items():
            row[f'{side}_{key}'] = val
        row[f'{side}_DAYS_REST'] = max((game_date - last_date).days - 1, 0)
    artifact = joblib.load(MODELS / 'nba_home_win.joblib')
    x = pd.DataFrame([row])[artifact['inputs']]
    p = float(artifact['model'].predict_proba(x)[0,1])
    if artifact['calibrator'] is not None:
        p = float(artifact['calibrator'].predict_proba([[p]])[0,1])
    candidates = [('HOME YES', p, args.home_price)]
    if args.away_price is not None:
        candidates.append(('AWAY YES', 1-p, args.away_price))
    else:
        # Do not assume the away YES quote equals 1 - home YES quote.
        print('No away price provided; evaluating HOME YES only.')
    print(f'{args.away} @ {args.home} on {game_date.date()}')
    print(f'Estimated home win probability: {p:.1%}; away: {1-p:.1%}')
    print(f'Model training data through: {artifact["last_historical_date"]}')
    for label, probability, price in candidates:
        edge = probability - price
        action = 'PAPER-REVIEW' if edge >= args.min_edge else 'NO BET'
        print(f'{label}: model={probability:.1%} market={price:.1%} edge={edge:+.1%} -> {action}')
    print('Research only. Check market settlement rules, fees, spread, liquidity, injuries, and stale data.')

if __name__ == '__main__':
    main()
