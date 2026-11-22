"""Historical NBA game-line market audit, with no live trading or wager placement.

Requires real, timestamped PRETIPOFF market-line offers. Historical outcomes are
joined to prior-only team features; calibration is fit on 2024-25 and evaluated
once on 2025-26. Without market offers, no bookmaker calibration is claimed.

Commands: fetch-one, import-snapshots, validate, report.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
from statistics import NormalDist
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .game_lines_live import FEATURES, feature_training
from .calibrate_game_lines import reliability, read_completed_history

ET = ZoneInfo('America/New_York')
NORMAL = NormalDist()
SCHEMA = 1
DEFAULT_QUOTES = Path('data/historical_market_lines.csv')
DEFAULT_REPORT = Path('models/market_line_validation_v1.json')
DEFAULT_AUDIT = Path('data/market_line_validation_audit.csv')
TRAIN_END = '2024-10-01'
VALID_END = '2025-10-01'
TEST_END = '2026-07-01'

# Explicit mapping avoids substring collisions (LA Clippers versus LA Lakers,
# or 'New York' versus 'Brooklyn'). Unknown team names are rejected.
TEAM_NAMES = {
    'ATL': 'Atlanta Hawks', 'BOS': 'Boston Celtics', 'BKN': 'Brooklyn Nets',
    'CHA': 'Charlotte Hornets', 'CHI': 'Chicago Bulls', 'CLE': 'Cleveland Cavaliers',
    'DAL': 'Dallas Mavericks', 'DEN': 'Denver Nuggets', 'DET': 'Detroit Pistons',
    'GSW': 'Golden State Warriors', 'HOU': 'Houston Rockets', 'IND': 'Indiana Pacers',
    'LAC': 'Los Angeles Clippers', 'LAL': 'Los Angeles Lakers',
    'MEM': 'Memphis Grizzlies', 'MIA': 'Miami Heat', 'MIL': 'Milwaukee Bucks',
    'MIN': 'Minnesota Timberwolves', 'NOP': 'New Orleans Pelicans',
    'NYK': 'New York Knicks', 'OKC': 'Oklahoma City Thunder',
    'ORL': 'Orlando Magic', 'PHI': 'Philadelphia 76ers',
    'PHX': 'Phoenix Suns', 'POR': 'Portland Trail Blazers',
    'SAC': 'Sacramento Kings', 'SAS': 'San Antonio Spurs',
    'TOR': 'Toronto Raptors', 'UTA': 'Utah Jazz', 'WAS': 'Washington Wizards',
}
ALIASES = {name.casefold(): code for code, name in TEAM_NAMES.items()}
ALIASES.update({code.casefold(): code for code in TEAM_NAMES})
ALIASES.update({'la clippers': 'LAC', 'la lakers': 'LAL'})


def team_code(value: str) -> str | None:
    return ALIASES.get(str(value).strip().casefold())


def utc_timestamp(value):
    """No naive dates: local time mistakes could create false pregame quotes."""
    if value is None or pd.isna(value):
        return None
    try:
        t = pd.Timestamp(value)
        return t.tz_convert('UTC') if t.tzinfo is not None else None
    except (TypeError, ValueError, OverflowError):
        return None


def _finite(value):
    try:
        v = float(value)
        return v if math.isfinite(v) else None
    except (ValueError, TypeError):
        return None


def _source_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if path.is_dir():
        found = sorted(path.rglob('*.json'))
        if found:
            return found
    raise FileNotFoundError(f'No JSON historical snapshots found at {path}')


def extract_snapshot(data: dict) -> list[dict]:
    """Normalize The Odds API v4 historical envelope, decimal prices only.

    Quote timestamp is the envelope's actual historical `timestamp`, not the
    date requested from the API and not the date on which we imported it.
    """
    if not isinstance(data, dict) or 'timestamp' not in data or not isinstance(data.get('data'), list):
        raise ValueError('Expected historical envelope {timestamp, data: [events]}, not live odds')
    captured = utc_timestamp(data.get('timestamp'))
    if captured is None:
        raise ValueError('Historical envelope missing timezone-aware timestamp')
    rows = []
    for event in data['data']:
        if not isinstance(event, dict) or event.get('sport_key') != 'basketball_nba':
            continue
        home = team_code(event.get('home_team'))
        away = team_code(event.get('away_team'))
        start = utc_timestamp(event.get('commence_time'))
        if not home or not away or home == away or start is None or not captured < start:
            continue
        day = start.tz_convert(ET).strftime('%Y-%m-%d')
        for book in event.get('bookmakers') or []:
            book_key = str(book.get('key') or '').strip()
            if not book_key:
                continue
            for market in book.get('markets') or []:
                kind = market.get('key')
                if kind not in ('spreads', 'totals'):
                    continue
                updated = utc_timestamp(market.get('last_update'))
                # A market "updated" after the source snapshot is contradictory.
                if updated is not None and updated > captured:
                    continue
                for outcome in market.get('outcomes') or []:
                    name = str(outcome.get('name') or '').strip()
                    side = team_code(name) if kind == 'spreads' else name.upper()
                    # Primary market: HOME spread and OVER total, one row per
                    # game/market in the audit, to avoid double-counting sides.
                    if kind == 'spreads' and side != home:
                        continue
                    if kind == 'totals' and side != 'OVER':
                        continue
                    point = _finite(outcome.get('point'))
                    odds = _finite(outcome.get('price'))
                    if point is None or odds is None or not 1 < odds <= 1000:
                        continue
                    rows.append({
                        'source': 'the_odds_api_v4_historical',
                        'source_event_id': str(event.get('id') or ''),
                        'game_date': day,
                        'home_team': home,
                        'away_team': away,
                        'game_start_utc': start.isoformat(),
                        'snapshot_utc': captured.isoformat(),
                        'market_update_utc': updated.isoformat() if updated is not None else '',
                        'bookmaker': book_key,
                        'market': kind,
                        'side': 'HOME' if kind == 'spreads' else 'OVER',
                        'point': point,
                        'decimal_odds': odds,
                    })
    return rows


def import_snapshots(source: Path, dest: Path) -> dict:
    files = _source_files(source)
    all_rows = []
    rejected = 0
    for path in files:
        try:
            obj = json.loads(path.read_text(encoding='utf-8'))
            all_rows.extend(extract_snapshot(obj))
        except (ValueError, OSError, json.JSONDecodeError) as exc:
            raise ValueError(f'{path}: {exc}') from exc
    if not all_rows:
        raise ValueError('No valid historical NBA spreads or totals found. Confirm NBA, timestamps and decimal odds.')
    frame = pd.DataFrame(all_rows)
    # Provider event/time/market/side identifies a quote. Preserve original
    # snapshot-time series rather than replacing history with latest values.
    keys = ['source_event_id', 'snapshot_utc', 'bookmaker', 'market', 'side']
    before = len(frame)
    frame = frame.drop_duplicates(keys, keep='last').sort_values(keys).reset_index(drop=True)
    dest.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(dest, index=False)
    return {'files': len(files), 'quotes': len(frame), 'duplicates': before-len(frame), 'path': str(dest)}


def _read_quotes(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f'Missing {path}. Import real historical market quotes first.')
    q = pd.read_csv(path, low_memory=False)
    fields = {'game_date','home_team','away_team','game_start_utc','snapshot_utc',
              'bookmaker','market','side','point'}
    missing = fields-set(q.columns)
    if missing:
        raise ValueError(f'Quote CSV missing columns: {sorted(missing)}')
    for col in ('game_start_utc','snapshot_utc'):
        # Reject naive timestamps; do not assume UTC.
        if q[col].map(utc_timestamp).isna().any():
            raise ValueError(f'{col}: timestamps must have explicit timezone offsets')
        q[col] = pd.to_datetime(q[col], utc=True)
    if 'market_update_utc' in q:
        bad_updated = [utc_timestamp(v) for v in q['market_update_utc']]
        if any(v is None and pd.notna(raw) and str(raw).strip()
               for v,raw in zip(bad_updated,q['market_update_utc'])):
            raise ValueError('market_update_utc contains naive or invalid timestamps')
        q['market_update_utc'] = pd.to_datetime(q['market_update_utc'], utc=True, errors='coerce')
    q['game_date'] = q['game_date'].astype(str)
    q['home_team'] = q['home_team'].map(team_code)
    q['away_team'] = q['away_team'].map(team_code)
    q['market'] = q['market'].astype(str).str.lower()
    q['side'] = q['side'].astype(str).str.upper()
    q['bookmaker'] = q['bookmaker'].astype(str).str.strip().str.lower()
    if 'decimal_odds' not in q:
        q['decimal_odds'] = np.nan # calibration of real point lines only; no reference ROI
    for col in ('point','decimal_odds'):
        q[col] = pd.to_numeric(q[col], errors='coerce')
    q.loc[~q['decimal_odds'].between(1.00001,1000),'decimal_odds']=np.nan
    return q


def match_quotes(completed: pd.DataFrame, quotes: pd.DataFrame, *, bookmaker: str,
                 lead_minutes: int = 60, max_age_hours: int = 36) -> tuple[pd.DataFrame, dict]:
    """Only pregame, predeclared bookmaker, and one HOME/OVER quote per game/market."""
    if lead_minutes < 0 or max_age_hours <= 0:
        raise ValueError('Invalid pregame quote lead/age')
    q = quotes.copy()
    before = len(q)
    q = q[(q['bookmaker'] == bookmaker.lower())
          & (q['market'].isin(['spreads','totals']))
          & (((q['market']=='spreads') & (q['side']=='HOME'))
             | ((q['market']=='totals') & (q['side']=='OVER')))
          & q['home_team'].notna() & q['away_team'].notna()
          & q['point'].notna()].copy()
    q['actual_day'] = q['game_start_utc'].dt.tz_convert(ET).dt.strftime('%Y-%m-%d')
    q = q[q['game_date'] == q['actual_day']].copy()
    # The score features use all completed prior *calendar-day* games. A
    # quote from a previous ET date might precede one of those outcomes,
    # even if it is nominally pretipoff. Fail closed on such hindsight risk.
    quote_day = q['snapshot_utc'].dt.tz_convert(ET).dt.strftime('%Y-%m-%d')
    q = q[quote_day == q['game_date']].copy()
    age = (q['game_start_utc'] - q['snapshot_utc']).dt.total_seconds() / 3600
    q = q[(age >= lead_minutes / 60) & (age <= max_age_hours)].copy()
    if 'market_update_utc' in q:
        # The book can appear in a fresh provider snapshot despite having a
        # stale market update; reject these rather than assuming the quote
        # remained executable throughout the gap.
        market_age = (q['snapshot_utc'] - q['market_update_utc']).dt.total_seconds() / 3600
        q = q[q['market_update_utc'].notna() & (q['market_update_utc']<=q['snapshot_utc'])
              & market_age.between(0, max_age_hours)].copy()
    games = completed.copy()
    games['game_date'] = pd.to_datetime(games['GAME_DATE']).dt.strftime('%Y-%m-%d')
    games['home_team'] = games['HOME_TEAM'].astype(str).str.upper()
    games['away_team'] = games['AWAY_TEAM'].astype(str).str.upper()
    if games.duplicated(['game_date','home_team','away_team']).any():
        raise ValueError('Ambiguous duplicate NBA completed-game matchup/date; refusing join')
    merged = q.merge(games[['GAME_ID','game_date','home_team','away_team']],
                     on=['game_date','home_team','away_team'],how='inner',validate='many_to_one')
    merged['GAME_ID'] = merged['GAME_ID'].astype(str)
    # one snapshot selected deterministically at most lead_minutes before tipoff
    # Equal-time contradictory offers are ambiguous; do not choose a better
    # price retrospectively. Exact duplicates also count only once.
    merged=merged.drop_duplicates(['GAME_ID','market','snapshot_utc','point','decimal_odds'])
    ambiguous=merged.duplicated(['GAME_ID','market','snapshot_utc'],keep=False)
    ambiguity_count=int(ambiguous.sum())
    merged=merged[~ambiguous].copy()
    merged = merged.sort_values(['GAME_ID','market','snapshot_utc'])
    merged = merged.drop_duplicates(['GAME_ID','market'],keep='last').reset_index(drop=True)
    return merged, {'input_quotes':before,'filtered_book_and_pregame':len(q),
                    'ambiguous_equal_time_quotes_rejected':ambiguity_count,
                    'joined_quotes':len(merged),'unmatched_or_discarded':before-len(merged)}


def _ridge():
    return make_pipeline(StandardScaler(),Ridge(alpha=150.0))


def _probability(mu: float, threshold: float, sigma: float) -> float:
    if sigma<=0 or not all(math.isfinite(v) for v in (mu,threshold,sigma)):
        raise ValueError('Invalid line-model prediction')
    return min(1.,max(0.,NORMAL.cdf((mu-threshold)/sigma)))


def _audit_rows(joined: pd.DataFrame, model_rows: pd.DataFrame, train: pd.DataFrame,
                valid: pd.DataFrame) -> tuple[pd.DataFrame,dict]:
    predicted={}
    sigmas={}
    for target in ('home_margin','total'):
        model=_ridge().fit(train[FEATURES],train[target])
        vpred=np.asarray(model.predict(valid[FEATURES]),dtype=float)
        sigma=float(np.mean(abs(vpred-valid[target].to_numpy(dtype=float))) * math.sqrt(math.pi/2))
        if not math.isfinite(sigma) or sigma<=.001:
            raise ValueError(f'{target}: validation residual sigma unavailable')
        sigmas[target]=sigma
        predicted[target]=dict(zip(model_rows['_game_key'],model.predict(model_rows[FEATURES])))
    result=[]
    for q in joined.to_dict('records'):
        game_key=str(q['GAME_ID'])
        margin=predicted['home_margin'].get(game_key)
        total=predicted['total'].get(game_key)
        if margin is None or total is None:
            continue # not enough pregame team history
        if q['market']=='spreads':
            # The Odds API home spread point is already signed: e.g. -5.5.
            threshold=-float(q['point'])
            p=_probability(float(margin),threshold,sigmas['home_margin'])
            margin_real=float(q['HOME_PTS'])-float(q['AWAY_PTS'])
            result_value=margin_real+float(q['point'])
        else:
            threshold=float(q['point'])
            p=_probability(float(total),threshold,sigmas['total'])
            result_value=float(q['HOME_PTS'])+float(q['AWAY_PTS'])-threshold
        outcome='PUSH' if abs(result_value)<1.e-8 else ('WIN' if result_value>0 else 'LOSS')
        price=_finite(q.get('decimal_odds'))
        price=price if price is not None and 1<price<=1000 else None
        realized=(0. if outcome=='PUSH' else price-1. if outcome=='WIN' else -1.) if price is not None else None
        expected = p * price - 1. if price is not None else None
        result.append({'GAME_ID':game_key,'game_date':q['game_date'],
                       'home_team':q['home_team'],'away_team':q['away_team'],
                       'bookmaker':q['bookmaker'],'market':q['market'], 'side':q['side'],
                       'point':float(q['point']),'decimal_odds':price,
                       'game_start_utc':q['game_start_utc'].isoformat(),
                       'snapshot_utc':q['snapshot_utc'].isoformat(),
                       'model_p':p,'outcome':outcome,'realized_unit_roi':realized,
                       'model_unit_ev':expected,'period':('validation' if q['game_date']<VALID_END
                       else 'test' if q['game_date']<TEST_END else 'outside_test')})
    return pd.DataFrame(result),sigmas


def _score(frame: pd.DataFrame, *, column: str) -> dict:
    resolved=frame[frame['outcome'].isin(['WIN','LOSS'])]
    if resolved.empty:
        return {'resolved':0,'pushes':int((frame.outcome=='PUSH').sum()),'brier':None,
                'ece_10_bins':None, 'reliability_bins':[]}
    r=reliability(resolved[column].to_numpy(dtype=float),
                  (resolved.outcome=='WIN').astype(float).to_numpy())
    return {'resolved':len(resolved),'pushes':int((frame.outcome=='PUSH').sum()),**r}


def _roi(frame:pd.DataFrame,*, minimum_ev:float, prob_col:str) -> dict:
    priced=frame[frame['decimal_odds'].between(1.00001,1000)].copy()
    if priced.empty:
        return {'n_all':0,'mean_reference_roi_all':None,'n_selected':0,
                'mean_reference_roi_selected':None,
                'unpriced_line_quotes':len(frame)}
    chosen=priced[(priced['outcome']!='PUSH') &
        ((priced[prob_col]*priced['decimal_odds']-1)>=minimum_ev)]
    return {'n_all':len(priced),
            'unpriced_line_quotes':len(frame)-len(priced),
            'mean_reference_roi_all':round(float(priced.realized_unit_roi.mean()),6),
            'n_selected':len(chosen),
            'mean_reference_roi_selected':round(float(chosen.realized_unit_roi.mean()),6)
            if len(chosen) else None}


def evaluate_market_history(completed: pd.DataFrame, quote_frame: pd.DataFrame, *,
                            bookmaker: str, min_market_val=100, min_market_test=100,
                            lead_minutes=60, max_age_hours=36, min_ev=0.05):
    """Score real pregame market lines without retraining or peeking at TEST.

    One HOME spread and one OVER total per NBA game and one fixed bookmaker.
    This is a bookmaker historical reference-price audit, NOT Polymarket EV.
    """
    rows,_=feature_training(completed)
    if rows.empty:
        raise ValueError('No completed game features')
    # Build dataset without accidentally treating same-game duplicate IDs as rows.
    completed_keys=completed[['GAME_ID','GAME_DATE','HOME_PTS','AWAY_PTS']].copy()
    completed_keys['GAME_ID']=completed_keys['GAME_ID'].astype(str)
    rows=rows.copy()
    # game-line feature_training omits GAME_ID; match on unique home+away+date.
    game_index=completed[['GAME_ID','GAME_DATE','HOME_TEAM','AWAY_TEAM']].copy()
    game_index['date_key']=pd.to_datetime(game_index['GAME_DATE']).dt.strftime('%Y-%m-%d')
    # target values plus features make identification possible: robust against
    # same-day games and unusual data inconsistencies.
    obs=completed[['GAME_ID','GAME_DATE','HOME_TEAM','AWAY_TEAM','HOME_PTS','AWAY_PTS']].copy()
    obs['GAME_ID']=obs['GAME_ID'].astype(str)
    obs['date_key']=pd.to_datetime(obs['GAME_DATE']).dt.strftime('%Y-%m-%d')
    # feature_training appends feat in chronological group order, but it does
    # not emit game ID. Match by DATE+margin+total only when unique; duplicates
    # are refused rather than arbitrarily assigned to another matchup.
    obs['home_margin']=obs.HOME_PTS-obs.AWAY_PTS
    obs['total']=obs.HOME_PTS+obs.AWAY_PTS
    rows['date_key']=pd.to_datetime(rows['date']).dt.strftime('%Y-%m-%d')
    idx_cols=['date_key','home_margin','total']
    # Ensure uniqueness for exact audit matches; otherwise those feature rows
    # cannot be identified safely with current game_lines_live.feature_training.
    duplicate_features=rows.duplicated(idx_cols,keep=False)
    dup_obs=obs.duplicated(idx_cols,keep=False)
    clean_rows=rows[~duplicate_features].copy()
    clean_obs=obs[~dup_obs].copy()
    model_rows=clean_rows.merge(clean_obs[['GAME_ID']+idx_cols],on=idx_cols,how='inner',validate='one_to_one')
    model_rows['_game_key']=model_rows['GAME_ID'].astype(str)
    train=model_rows[model_rows.date<pd.Timestamp(TRAIN_END)].copy()
    valid=model_rows[(model_rows.date>=pd.Timestamp(TRAIN_END)) &(model_rows.date<pd.Timestamp(VALID_END))].copy()
    if len(train)<400 or len(valid)<100:
        raise ValueError(f'Not enough prior-only game features: train={len(train)}, validation={len(valid)}')
    joined,diag=match_quotes(completed,quote_frame,bookmaker=bookmaker,
                             lead_minutes=lead_minutes,max_age_hours=max_age_hours)
    # Need scores for calculating line outcomes; joined only has game ID.
    joined=joined.merge(completed_keys,on='GAME_ID',how='inner',validate='many_to_one')
    result,sigmas=_audit_rows(joined,model_rows,train,valid)
    if result.empty:
        raise ValueError('Zero aligned pregame market lines with feature-ready NBA scores')
    by_target={}
    audited=result.copy()
    audited['calibrated_p']=np.nan
    for kind,target in [('spreads','home_margin'),('totals','total')]:
        target_rows=audited[audited.market==kind].copy()
        val=target_rows[(target_rows.period=='validation') & (target_rows.outcome!='PUSH')]
        test=target_rows[(target_rows.period=='test') & (target_rows.outcome!='PUSH')]
        baseline=_score(test,column='model_p')
        status='insufficient_market_quotes'
        knots=None
        calibrated_stats=None
        support=0
        comparable_baseline=None
        if len(val)>=min_market_val and len(test)>=min_market_test:
            iso=IsotonicRegression(y_min=0.,y_max=1.,out_of_bounds='clip')
            iso.fit(val.model_p.to_numpy(),(val.outcome=='WIN').astype(float).to_numpy())
            xs=np.asarray(iso.X_thresholds_,dtype=float)
            ys=np.asarray(iso.y_thresholds_,dtype=float)
            if len(xs)>=2 and np.diff(xs).min()>0:
                mask=(test.model_p>=xs[0]) & (test.model_p<=xs[-1])
                if mask.sum()>=min_market_test:
                    test_cal=np.interp(test.loc[mask,'model_p'],xs,ys)
                    audited.loc[test.index[mask],'calibrated_p']=np.clip(test_cal,.005,.995)
                    evaluated=audited.loc[test.index[mask]]
                    calibrated_stats=_score(evaluated,column='calibrated_p')
                    # Fair baseline comparison on the exact same support rows.
                    comparable_baseline=_score(evaluated,column='model_p')
                    support=int(mask.sum())
                    pass_gate=(calibrated_stats['brier']<=comparable_baseline['brier']+.002 and
                               calibrated_stats['ece_10_bins']<=comparable_baseline['ece_10_bins']+.010 and
                               calibrated_stats['ece_10_bins']<=.075)
                    status='market_reference_test_passed' if pass_gate else 'market_reference_test_failed'
                    knots={'x':[float(v) for v in xs], 'y':[float(v) for v in ys]}
                else:
                    status='insufficient_out_of_sample_support'
            else:
                status='degenerate_calibrator'
        by_target[target]={
            'status':status,'bookmaker':bookmaker,
            'validation_quotes_resolved':len(val),'test_quotes_resolved':len(test),
            'test_baseline':baseline,
            'test_baseline_on_support':comparable_baseline if calibrated_stats else None,
            'test_calibrated_on_support':calibrated_stats,
            'test_quotes_on_calibration_support':support,
            'isotonic_knots':knots if status=='market_reference_test_passed' else None,
            'research_gate_passed':status=='market_reference_test_passed',
            'baseline_reference_roi':_roi(test,minimum_ev=min_ev,prob_col='model_p'),
            'not_executable_price':True,
        }
    payload={
        'schema':SCHEMA,'created_utc':datetime.now(timezone.utc).isoformat(),
        'validation_scope':'timestamped_historical_bookmaker_reference_lines',
        'polymarket_market_line_validated':False,
        'automatic_live_use':False,
        'not_trading_advice':True,
        'split':{'training_before':TRAIN_END,'validation_start':TRAIN_END,
                 'test_start':VALID_END,'test_end_exclusive':TEST_END,
                 'training_feature_games':len(train),'validation_feature_games':len(valid)},
        'quote_policy':{'bookmaker':bookmaker,'lead_minutes':lead_minutes,
                        'max_age_hours':max_age_hours,'chosen_side':'home_spread_and_total_over',
                        'min_modeled_ev_for_reference_roi':min_ev},
        'diagnostics':{**diag,'feature_identification_ambiguous_rows_excluded':int(duplicate_features.sum())},
        'targets':by_target,
        'caveat':('Reference sportsbook prices may not have been available for actual fills. '
                  'No transaction fees, limits, market impact or Polymarket Combo execution are modeled. '
                  'Do not transfer bookmaker-specific calibration to Polymarket automatically.'),
    }
    return payload,audited


def _atomic_json(path: Path, payload: dict):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(payload,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    os.replace(tmp,path)


def _fetch_one(args):
    """One explicitly requested paid historical API call. Never bulk fetch."""
    import requests
    if not args.confirm_paid_request:
        raise ValueError('Paid endpoint: add --confirm-paid-request after reviewing the provider credit cost.')
    if args.output.exists():
        raise FileExistsError(f'Will not overwrite an archived snapshot: {args.output}')
    api_key=os.environ.get('ODDS_API_KEY','')
    if not api_key:
        raise ValueError('Set ODDS_API_KEY locally; never paste the key in chat or commit it.')
    requested=utc_timestamp(args.at)
    if requested is None or requested > pd.Timestamp.now(tz='UTC'):
        raise ValueError('--at must be a past timezone-aware ISO8601 timestamp')
    params={'apiKey':api_key,'regions':args.region,'markets':'spreads,totals',
            'oddsFormat':'decimal','date':requested.strftime('%Y-%m-%dT%H:%M:%SZ')}
    if args.bookmaker:
        params['bookmakers']=args.bookmaker
    url='https://api.the-odds-api.com/v4/historical/sports/basketball_nba/odds'
    try:
        response=requests.get(url,params=params,timeout=(5,30))
    except requests.RequestException:
        raise RuntimeError('Historical odds request failed; check connectivity. API key omitted from error.') from None
    if not response.ok:
        # requests.HTTPError includes the URL and the apiKey parameter.
        # Never propagate a secret-bearing URL into logs or chat.
        raise RuntimeError(f'Historical odds HTTP {response.status_code}; check provider access, plan and quota.')
    try:
        data=response.json()
    except ValueError:
        raise RuntimeError('Historical odds provider returned invalid JSON.') from None
    if not isinstance(data,dict) or 'timestamp' not in data or 'data' not in data:
        raise ValueError('Provider did not return historical snapshot envelope')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    _atomic_json(args.output,data)
    print('Saved historical snapshot:',args.output,'actual timestamp',data['timestamp'])
    print('API credits remaining:',response.headers.get('x-requests-remaining','unknown'),
          '| last charge:',response.headers.get('x-requests-last','unknown'))


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    commands=parser.add_subparsers(dest='command',required=True)
    fetching=commands.add_parser('fetch-one',help='One paid API historical call; no bulk fetching')
    fetching.add_argument('--at',required=True,help='ISO8601 UTC timestamp (past)')
    fetching.add_argument('--region',choices=['us','us2','uk','eu','au'],default='us')
    fetching.add_argument('--confirm-paid-request',action='store_true',help='Explicitly authorize ONE credit-bearing provider call')
    fetching.add_argument('--bookmaker',default=None)
    fetching.add_argument('--output',type=Path,required=True)
    importing=commands.add_parser('import-snapshots',help='Normalize saved The Odds API historical JSON')
    importing.add_argument('--input',type=Path,required=True)
    importing.add_argument('--output',type=Path,default=DEFAULT_QUOTES)
    validating=commands.add_parser('validate',help='Evaluate actual timestamped market lines')
    validating.add_argument('--quotes',type=Path,default=DEFAULT_QUOTES)
    validating.add_argument('--completed-csv',type=Path)
    validating.add_argument('--bookmaker',required=True)
    validating.add_argument('--lead-minutes',type=int,default=60)
    validating.add_argument('--max-age-hours',type=int,default=36)
    validating.add_argument('--min-ev',type=float,default=.05)
    validating.add_argument('--min-market-val',type=int,default=100)
    validating.add_argument('--min-market-test',type=int,default=100)
    validating.add_argument('--output',type=Path,default=DEFAULT_REPORT)
    validating.add_argument('--audit-csv',type=Path,default=DEFAULT_AUDIT)
    commands.add_parser('report',help='Show saved market-line audit without rerunning models')
    args=parser.parse_args(argv)
    if args.command=='fetch-one':
        _fetch_one(args)
    elif args.command=='import-snapshots':
        print(json.dumps(import_snapshots(args.input,args.output),indent=2))
    elif args.command=='validate':
        if args.min_market_val<10 or args.min_market_test<10 or not 0<=args.min_ev<1:
            parser.error('Invalid minimum sample / EV settings')
        if args.completed_csv:
            completed=pd.read_csv(args.completed_csv,dtype={'GAME_ID':str})
        else:
            completed=read_completed_history(Path(__file__).resolve().parents[1])
        quote_frame=_read_quotes(args.quotes)
        artifact,audit=evaluate_market_history(completed,quote_frame,
            bookmaker=args.bookmaker,lead_minutes=args.lead_minutes,
            max_age_hours=args.max_age_hours,min_ev=args.min_ev,
            min_market_val=args.min_market_val,min_market_test=args.min_market_test)
        args.audit_csv.parent.mkdir(parents=True,exist_ok=True)
        audit.to_csv(args.audit_csv,index=False)
        _atomic_json(args.output,artifact)
        for target,item in artifact['targets'].items():
            b=item['test_baseline'];c=item['test_calibrated_on_support']
            print(target,'status',item['status'],'validation N',item['validation_quotes_resolved'],
                'test N',item['test_quotes_resolved'],
                'baseline Brier',b.get('brier'),'baseline ECE',b.get('ece_10_bins'),
                'calibrated Brier (support only)',c.get('brier') if c else None)
        print('Saved',args.output,'and row audit',args.audit_csv)
        print('Bookmaker reference lines ONLY; no Polymarket execution or profitable edge established.')
    elif args.command=='report':
        if not DEFAULT_REPORT.exists():
            print('No actual historical market-line audit yet. Import quotes and run validate.')
        else:
            result=json.loads(DEFAULT_REPORT.read_text(encoding='utf-8'))
            print(json.dumps(result,indent=2))
    return 0


if __name__=='__main__':
    raise SystemExit(main())
