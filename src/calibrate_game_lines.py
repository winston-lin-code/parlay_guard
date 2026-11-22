"""Chronological, out-of-sample game-line probability calibration.

Training: historical features before 2024-10-01; calibration validation:
2024-10-01 to 2025-09-30; untouched test: 2025-10-01 to 2026-06-30.

Uses *synthetic forecast-centered thresholds*, NOT historical bookmaker or
Polymarket quotes. Test scores assess probability reliability on that grid,
not profitability or calibration against offered market lines. No network
requests are made; all game feeds must already be cached on disk.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from statistics import NormalDist

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

from .game_lines_live import FEATURES, feature_training

VERSION = 1
DEFAULT_OUTPUT = Path('models/game_line_calibration_v1.json')
TRAIN_END = '2024-10-01'
VALID_END = '2025-10-01'
TEST_END = '2026-07-01'
# Values are multiples of the validation residual sigma; these are NOT observed
# market offers. This gives a spread of event probabilities for reliability tests.
OFFSETS = (-1.75, -1.25, -0.75, -0.25, 0.25, 0.75, 1.25, 1.75)
NORM = NormalDist()


def probability_over(mean: float, threshold: float, sigma: float) -> float:
    if not (math.isfinite(mean) and math.isfinite(threshold) and math.isfinite(sigma) and sigma > 0):
        raise ValueError('Finite forecast, threshold and positive sigma required')
    return float(min(1.0, max(0.0, NORM.cdf((mean - threshold) / sigma))))


def reliability(predicted: np.ndarray, outcome: np.ndarray, *, bins=10) -> dict:
    """Brier and ECE; multiple thresholds from one game are dependent rows."""
    p = np.asarray(predicted, dtype=float)
    y = np.asarray(outcome, dtype=float)
    if p.size != y.size or p.size == 0 or not (np.isfinite(p).all() and np.isfinite(y).all()):
        raise ValueError('Missing or non-finite predictions/outcomes')
    if np.any((p < 0) | (p > 1)) or np.any((y != 0) & (y != 1)):
        raise ValueError('Invalid probability or binary outcome')
    err = float(np.mean((p-y)**2))
    rows=[]
    ece=0.0
    for i in range(bins):
        lo,hi=i/bins,(i+1)/bins
        mask=(p>=lo)&(p<=hi if i==bins-1 else p<hi)
        if not mask.any():continue
        conf=float(p[mask].mean())
        actual=float(y[mask].mean())
        ece+=float(mask.sum()/p.size)*abs(conf-actual)
        rows.append({'bin_start':round(lo,3),'bin_end':round(hi,3),
                     'n_threshold_events':int(mask.sum()),'mean_probability':round(conf,5),
                     'observed_rate':round(actual,5)})
    return {'brier':round(err,6), 'ece_10_bins':round(ece,6),
            'threshold_events':int(p.size), 'reliability_bins':rows}


def candidate_grid(means: np.ndarray, actuals: np.ndarray, sigma: float):
    """Expand each pregame forecast to synthetic, forecast-centered events.

    All thresholds for the same game have equal total weighting across games;
    the entries are NOT independent observations and are not market lines.
    """
    means=np.asarray(means,dtype=float)
    actuals=np.asarray(actuals,dtype=float)
    if means.shape!=actuals.shape or not means.size or not np.isfinite(means).all() or not np.isfinite(actuals).all():
        raise ValueError('Invalid forecasts or actual scores')
    offsets=np.asarray(OFFSETS,dtype=float)
    thresholds=(means[:,None]+sigma*offsets[None,:]).reshape(-1)
    baseline=np.asarray([probability_over(float(mu),float(t),sigma)
                         for mu,t in zip(np.repeat(means,len(offsets)),thresholds)])
    outcomes=(np.repeat(actuals,len(offsets))>thresholds).astype(int)
    return baseline, outcomes


def _ridge():
    return make_pipeline(StandardScaler(), Ridge(alpha=150.0))


def evaluate_calibration(completed: pd.DataFrame, *, train_end=TRAIN_END,
                         valid_end=VALID_END, test_end=TEST_END,
                         min_train=500, min_val=150, min_test=150) -> dict:
    """Freeze model on TRAIN, fit calibrator on VALID, evaluate once on TEST.

    Returns only safe JSON numbers and isotonic knot arrays; does not write any
    files. Do not tune hyperparameters based on the held-out TEST result.
    """
    if not (train_end < valid_end < test_end):
        raise ValueError('Chronological cutoffs must be increasing')
    rows,_=feature_training(completed)
    if rows.empty:
        raise ValueError('No feature-ready historical games')
    train=rows[rows.date<pd.Timestamp(train_end)].copy()
    valid=rows[(rows.date>=pd.Timestamp(train_end))&(rows.date<pd.Timestamp(valid_end))].copy()
    test=rows[(rows.date>=pd.Timestamp(valid_end))&(rows.date<pd.Timestamp(test_end))].copy()
    if min(len(train),len(valid),len(test))<=0 or len(train)<min_train or len(valid)<min_val or len(test)<min_test:
        raise ValueError(f'Insufficient chronological games: train={len(train)} validation={len(valid)} test={len(test)}')
    payload={
        'schema':VERSION, 'created_utc':datetime.now(timezone.utc).isoformat(),
        'validation_scope':'synthetic_forecast_centered_thresholds_only',
        'market_line_validated':False,
        'production_model_validation_note':('Frozen train-period Ridge checked on validation and test; '
            'live Ridge refits on all completed games, so transfer to live predictions is not guaranteed.'),
        'probability_method':'isotonic_normal_cdf',
        'train_end_exclusive':train_end, 'valid_end_exclusive':valid_end,
        'test_end_exclusive':test_end,
        'split':{'train_games':len(train),'validation_games':len(valid),'test_games':len(test),
                 'train_latest':str(train.date.max().date()),
                 'validation_latest':str(valid.date.max().date()),
                 'test_latest':str(test.date.max().date())},
        'targets':{},
    }
    for name in ('home_margin','total'):
        model=_ridge()
        model.fit(train[FEATURES],train[name])
        valid_mean=np.asarray(model.predict(valid[FEATURES]),dtype=float)
        test_mean=np.asarray(model.predict(test[FEATURES]),dtype=float)
        valid_actual=valid[name].to_numpy(dtype=float)
        test_actual=test[name].to_numpy(dtype=float)
        # Sigma is estimated only from VALID residuals, never TEST outcomes.
        sigma=float(np.mean(abs(valid_actual-valid_mean))*math.sqrt(math.pi/2))
        if not math.isfinite(sigma) or sigma <= 0.001:
            raise ValueError(f'{name}: validation residual variance too small')
        val_base,val_label=candidate_grid(valid_mean,valid_actual,sigma)
        iso=IsotonicRegression(y_min=0.0,y_max=1.0,out_of_bounds='clip')
        iso.fit(val_base,val_label)
        knots_x=np.asarray(iso.X_thresholds_,dtype=float)
        knots_y=np.asarray(iso.y_thresholds_,dtype=float)
        test_base,test_label=candidate_grid(test_mean,test_actual,sigma)
        test_adjusted=np.interp(test_base,knots_x,knots_y)
        base_stats=reliability(test_base,test_label)
        adjusted_stats=reliability(test_adjusted,test_label)
        # Predetermined gates. A failed gate does not erase negative test
        # findings or silently switch the live estimator to something else.
        passes=(adjusted_stats['brier']<=base_stats['brier']+0.002 and
                adjusted_stats['ece_10_bins']<=base_stats['ece_10_bins']+0.010 and
                adjusted_stats['ece_10_bins']<=0.075)
        payload['targets'][name]={
            'sigma_validation':round(sigma,6),
            'validation_games':len(valid),'test_games':len(test),
            'knots_x':[round(float(v),9) for v in knots_x],
            'knots_y':[round(float(v),9) for v in knots_y],
            'test_baseline':base_stats,'test_calibrated':adjusted_stats,
            'passed_predeclared_reliability_gates':bool(passes),
            'use_for_research':bool(passes),
        }
    return payload


def _valid_knots(data: dict) -> bool:
    try:
        x=np.asarray(data['knots_x'],dtype=float);y=np.asarray(data['knots_y'],dtype=float)
        return (x.ndim==1 and len(x)>=2 and len(x)==len(y) and np.isfinite(x).all()
                and np.isfinite(y).all() and (np.diff(x)>0).all()
                and (np.diff(y)>=-1e-10).all() and np.all((0<=x)&(x<=1))
                and np.all((0<=y)&(y<=1)))
    except (KeyError,ValueError,TypeError):
        return False


def load_calibrator(path: Path) -> dict | None:
    """Load validated JSON only; fail closed when missing/corrupted/stale format."""
    try:
        artifact=json.loads(Path(path).read_text(encoding='utf-8'))
        if (artifact.get('schema')!=VERSION or
            artifact.get('validation_scope')!='synthetic_forecast_centered_thresholds_only' or
            artifact.get('market_line_validated') is not False or
            artifact.get('probability_method')!='isotonic_normal_cdf'):
            return None
        targets=artifact['targets']
        if set(targets)!={'home_margin','total'}:
            return None
        if any(not _valid_knots(v) for v in targets.values()):
            return None
        if any((not math.isfinite(float(v.get('sigma_validation', float('nan'))))
                or not 0<float(v['sigma_validation'])<100)
               for v in targets.values()):
            return None
        return artifact
    except (OSError,ValueError,TypeError,KeyError):
        return None


def calibrated_probability(artifact: dict | None, kind: str, probability: float) -> tuple[float, str]:
    """Return (P, status). Never claim market quotes have been calibrated."""
    p=float(probability)
    if not math.isfinite(p) or not 0<=p<=1:
        raise ValueError('Invalid probability')
    key={'spread':'home_margin','total':'total'}.get(kind)
    if artifact is None or key is None:
        return p,'uncalibrated_normal_approximation'
    target=artifact.get('targets',{}).get(key,{})
    if not target.get('use_for_research') or not _valid_knots(target):
        return p,'uncalibrated_normal_approximation'
    knots_x=target['knots_x']
    if p < knots_x[0] or p > knots_x[-1]:
        # Real quoted lines can be much farther from a forecast than the
        # synthetic threshold grid. Do NOT extrapolate isotonic probabilities.
        return p,'outside_calibration_support'
    mapped=float(np.interp(p,knots_x,target['knots_y']))
    # No finite validation sample establishes literal zero/one certainty.
    return max(0.005,min(0.995,mapped)),'test_evaluated_synthetic_thresholds'


def read_completed_history(root: Path) -> pd.DataFrame:
    """Same verified two-team data path as the live line model; no downloads."""
    from .predict_today import load_history,fetch_current_season,clean_team_history,make_completed_games
    from zoneinfo import ZoneInfo
    local_day=datetime.now(timezone.utc).astimezone(ZoneInfo('America/New_York')).date()
    season=f'{local_day.year}-{(local_day.year+1)%100:02d}' if local_day.month>=7 else f'{local_day.year-1}-{local_day.year%100:02d}'
    current=root/'data'/f'nba_{season.replace("-","_")}_latest_raw.csv'
    if not current.is_file():
        raise FileNotFoundError(f'Missing {current}. Refresh the live model to cache current season team logs.')
    old=load_history(root/'data/nba_games_updated.csv')
    now=fetch_current_season(season,current)
    cleaned=clean_team_history(old,now,local_day)
    return make_completed_games(cleaned)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--completed-csv',type=Path,default=None,
                        help='Optional pre-verified one-row-per-game CSV. By default, use cached NBA team history.')
    parser.add_argument('--output',type=Path,default=DEFAULT_OUTPUT)
    args=parser.parse_args(argv)
    if args.completed_csv is None:
        root=Path(__file__).resolve().parents[1]
        completed=read_completed_history(root)
    else:
        completed=pd.read_csv(args.completed_csv,dtype={'GAME_ID':str})
        mandatory={'GAME_DATE','HOME_TEAM','AWAY_TEAM','HOME_PTS','AWAY_PTS'}
        if not mandatory.issubset(completed):
            parser.error(f'Missing columns: {sorted(mandatory-set(completed))}')
    artifact=evaluate_calibration(completed)
    for target,stats in artifact['targets'].items():
        print(f"{target} | test games {stats['test_games']} | ",
              f"baseline Brier {stats['test_baseline']['brier']:.4f} -> calibrated {stats['test_calibrated']['brier']:.4f} | ",
              f"ECE {stats['test_baseline']['ece_10_bins']:.4f} -> {stats['test_calibrated']['ece_10_bins']:.4f} | ",
              f"research gate {stats['use_for_research']}")
    out=args.output
    out.parent.mkdir(parents=True,exist_ok=True)
    staging=out.with_suffix(out.suffix+'.tmp')
    staging.write_text(json.dumps(artifact,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    os.replace(staging,out)
    print('Saved:',out)
    print('SCOPE: calibrated on SYNTHETIC forecast-centered thresholds, not observed sports-market lines.')
    print('Do not use these probabilities to assert executable positive EV.')
    return 0


if __name__=='__main__':
    raise SystemExit(main())
