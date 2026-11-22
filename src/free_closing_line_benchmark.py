"""Free NBA closing-line benchmark (never an executable-odds backtest).

Two public Hugging Face consensus datasets provide OPEN/CLOSE spread and total,
but NOT verified quote-publication timestamps or single-book executable prices.
The test outcomes are strictly after the training and calibration periods.

Usage from the project root:
    python -m src.free_closing_line_benchmark download
    python -m src.free_closing_line_benchmark evaluate
    python -m src.free_closing_line_benchmark report

This module is deliberately independent from the existing timestamped market-line
validator; it does not write data/historical_market_lines.csv and never claims
that retrospective closing quotes existed at a prior decision cutoff.
"""
from __future__ import annotations

import argparse
from collections import defaultdict, deque
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
from statistics import NormalDist

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .game_lines_live import FEATURES, _features
from .calibrate_game_lines import read_completed_history, reliability
from .market_line_validation import TEAM_NAMES

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / 'data/external/free_closing_lines'
DEFAULT_REPORT = ROOT / 'models/free_closing_line_benchmark_v1.json'
DEFAULT_AUDIT = ROOT / 'data/free_closing_line_benchmark_v1.csv'
# These are public, third-party datasets; license and provenance at linked source.
DATASETS = {
    '2024-25': (
        'nba_2024_25_closing_lines_ats.csv',
        'https://huggingface.co/datasets/minmaxxxer/nba-closing-lines-ats-2024-25/resolve/main/nba_2024_25_closing_lines_ats.csv',
    ),
    '2025-26': (
        'nba_2025_26_closing_lines_ats.csv',
        'https://huggingface.co/datasets/minmaxxxer/nba-closing-lines-ats-2025-26/resolve/main/nba_2025_26_closing_lines_ats.csv',
    ),
}
TRAIN_END = '2023-10-01'
SIGMA_START, SIGMA_END = '2023-10-01', '2024-07-01'
CAL_START, CAL_END = '2024-10-01', '2025-07-01'
TEST_START, TEST_END = '2025-10-01', '2026-07-01'
TEAM_MAP = {name.casefold():abbr for abbr, name in TEAM_NAMES.items()}
NORM = NormalDist()


def download_files(dest: Path=DATA_DIR, *, overwrite: bool=False) -> dict:
    """One GET per file; no API key or paid API access; don't mutate source files."""
    import requests
    dest.mkdir(parents=True,exist_ok=True)
    summary={}
    for season,(fname,url) in DATASETS.items():
        out=dest/fname
        if out.exists() and not overwrite:
            summary[season]={'status':'existing','path':str(out),'bytes':out.stat().st_size}
            continue
        try:
            response=requests.get(url,timeout=(10,60),stream=True,
                headers={'User-Agent':'ParlayGuardResearch/1.0'})
            response.raise_for_status()
        except requests.RequestException as exc:
            raise RuntimeError(f'{season}: free download failed ({type(exc).__name__}). '
                f'Download manually from {url} and put it in {dest}') from None
        staging=out.with_name(out.name+'.partial')
        n=0; digest=hashlib.sha256()
        try:
            with staging.open('wb') as f:
                for chunk in response.iter_content(chunk_size=65536):
                    if not chunk:continue
                    n+=len(chunk)
                    if n>12_000_000:raise ValueError('Unexpectedly large CSV response')
                    f.write(chunk);digest.update(chunk)
            # Do not replace a good file with an HTML error page.
            with staging.open('r',encoding='utf-8-sig') as f:
                header=f.readline().strip()
            if not {'date_et','home_team','away_team','close_spread_home','close_total'}.issubset(set(header.split(','))):
                raise ValueError(f'{season}: response is not the expected CSV')
            os.replace(staging,out)
        finally:
            staging.unlink(missing_ok=True)
            response.close()
        summary[season]={'status':'downloaded','path':str(out),'bytes':n,'sha256':digest.hexdigest()}
    return summary


def normalize_source(source: pd.DataFrame, season: str) -> tuple[pd.DataFrame,dict]:
    required={'date_et','season','season_type','home_team','away_team','home_score',
              'away_score','close_spread_home','close_total','close_home_cover_prob','close_over_prob'}
    missing=required-set(source.columns)
    if missing:raise ValueError(f'{season} CSV missing columns: {sorted(missing)}')
    df=source.copy()
    input_count=len(df)
    df=df[(df['season'].astype(str)==season)&
          (df['season_type'].astype(str).str.casefold()=='regular')].copy()
    regular_count=len(df)
    df['GAME_DATE']=pd.to_datetime(df['date_et'],errors='coerce').dt.normalize()
    df['HOME_TEAM']=df['home_team'].astype(str).str.strip().str.casefold().map(TEAM_MAP)
    df['AWAY_TEAM']=df['away_team'].astype(str).str.strip().str.casefold().map(TEAM_MAP)
    fields=['home_score','away_score','close_spread_home','close_total','close_home_cover_prob','close_over_prob']
    for col in fields:df[col]=pd.to_numeric(df[col],errors='coerce')
    ok=(df['GAME_DATE'].notna() & df['HOME_TEAM'].notna() & df['AWAY_TEAM'].notna() &
        df['HOME_TEAM'].ne(df['AWAY_TEAM']) & df[fields].notna().all(axis=1) &
        df['close_spread_home'].between(-40,40) & df['close_total'].between(150,330) &
        df['close_home_cover_prob'].between(0.01,0.99) & df['close_over_prob'].between(0.01,0.99) &
        df['home_score'].between(60,190) & df['away_score'].between(60,190))
    df=df.loc[ok].copy()
    # Do not choose between ambiguous odds rows or duplicate same-date fixtures.
    duplicated=df.duplicated(['GAME_DATE','HOME_TEAM','AWAY_TEAM'],keep=False)
    duplicate_count=int(duplicated.sum())
    df=df.loc[~duplicated].copy()
    df=df.rename(columns={'home_score':'SOURCE_HOME_PTS','away_score':'SOURCE_AWAY_PTS'})
    columns=['GAME_DATE','HOME_TEAM','AWAY_TEAM','SOURCE_HOME_PTS','SOURCE_AWAY_PTS',
             'close_spread_home','close_total','close_home_cover_prob','close_over_prob']
    stats={'source_rows':input_count,'regular_rows':regular_count,
           'accepted_rows':len(df),'ambiguous_duplicate_rows':duplicate_count}
    return df[columns].reset_index(drop=True),stats


def load_closing_quotes(directory:Path=DATA_DIR) -> tuple[pd.DataFrame,dict]:
    parts=[];diagnostics={}
    for season,(fname,url) in DATASETS.items():
        path=directory/fname
        if not path.is_file():
            raise FileNotFoundError(f'Missing {path}. Run: python -m src.free_closing_line_benchmark download')
        part,diag=normalize_source(pd.read_csv(path,low_memory=False),season)
        diagnostics[season]={'source_url':url,'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),**diag}
        parts.append(part)
    return pd.concat(parts,ignore_index=True),diagnostics


def keyed_pregame_features(completed:pd.DataFrame)->pd.DataFrame:
    """Rebuild same shifted 10-game inputs as src.game_lines_live, retaining game identity."""
    required={'GAME_DATE','HOME_TEAM','AWAY_TEAM','HOME_PTS','AWAY_PTS'}
    if not required.issubset(completed):raise ValueError(f'Missing completed columns {sorted(required-set(completed))}')
    games=completed.copy()
    games['GAME_DATE']=pd.to_datetime(games['GAME_DATE'],errors='raise').dt.normalize()
    games=games.sort_values(['GAME_DATE','GAME_ID'] if 'GAME_ID' in games else ['GAME_DATE'])
    if games.duplicated(['GAME_DATE','HOME_TEAM','AWAY_TEAM']).any():
        raise ValueError('Ambiguous duplicate completed games by date and teams')
    history=defaultdict(lambda:deque(maxlen=10))
    observations=[]
    for day,group in games.groupby('GAME_DATE',sort=True):
        updates=[]
        for game in group.itertuples(index=False):
            home,away=str(game.HOME_TEAM).upper(),str(game.AWAY_TEAM).upper()
            feat=_features(home,away,day,history)
            if feat is not None:
                observations.append({**feat,'GAME_DATE':day,'HOME_TEAM':home,'AWAY_TEAM':away,
                    'HOME_PTS':float(game.HOME_PTS),'AWAY_PTS':float(game.AWAY_PTS),
                    'home_margin':float(game.HOME_PTS)-float(game.AWAY_PTS),
                    'total':float(game.HOME_PTS)+float(game.AWAY_PTS)})
            updates.append((home,away,float(game.HOME_PTS),float(game.AWAY_PTS)))
        for home,away,hp,ap in updates:
            history[home].append((hp,ap,day));history[away].append((ap,hp,day))
    return pd.DataFrame(observations)


def _score(df:pd.DataFrame,column:str)->dict|None:
    if len(df)==0:return None
    stats=reliability(df[column].to_numpy(dtype=float),df['won'].to_numpy(dtype=int))
    return {'n_games':len(df),'brier':stats['brier'],'ece_10_bins':stats['ece_10_bins'],
            'reliability_bins':stats['reliability_bins']}


def _prob(mean: np.ndarray,threshold:np.ndarray,sigma:float)->np.ndarray:
    from scipy.special import ndtr  # scipy is already a scikit-learn dependency
    return np.clip(ndtr((np.asarray(mean)-np.asarray(threshold))/sigma),0.005,0.995)


def evaluate(completed:pd.DataFrame,quotes:pd.DataFrame,*,min_scale=150,min_cal=100,min_test=100)->tuple[dict,pd.DataFrame]:
    rows=keyed_pregame_features(completed)
    if rows.empty:raise ValueError('No feature-ready historical games')
    rows['GAME_DATE']=pd.to_datetime(rows['GAME_DATE']).dt.normalize()
    train=rows[rows.GAME_DATE<pd.Timestamp(TRAIN_END)]
    scale=rows[(rows.GAME_DATE>=pd.Timestamp(SIGMA_START))&(rows.GAME_DATE<pd.Timestamp(SIGMA_END))]
    if len(train)<500 or len(scale)<min_scale:
        raise ValueError(f'Insufficient training/scale games: train={len(train)} scale={len(scale)}')
    model_outputs=rows[['GAME_DATE','HOME_TEAM','AWAY_TEAM','HOME_PTS','AWAY_PTS','home_margin','total']].copy()
    sigmas={}
    for target in ('home_margin','total'):
        model=make_pipeline(StandardScaler(),Ridge(alpha=150.0))
        model.fit(train[FEATURES],train[target])
        mu=model.predict(rows[FEATURES]);scale_mu=model.predict(scale[FEATURES])
        sigma=float(np.mean(abs(scale[target].to_numpy()-scale_mu))*math.sqrt(math.pi/2))
        if not np.isfinite(sigma) or sigma<=0.1:raise ValueError(f'Invalid held-out {target} sigma')
        sigmas[target]=sigma
        model_outputs[target+'_forecast']=mu
    q=quotes.copy()
    q['GAME_DATE']=pd.to_datetime(q['GAME_DATE'],errors='raise').dt.normalize()
    if q.duplicated(['GAME_DATE','HOME_TEAM','AWAY_TEAM']).any():
        raise ValueError('Ambiguous quotes by date and team pair')
    merged=model_outputs.merge(q,on=['GAME_DATE','HOME_TEAM','AWAY_TEAM'],how='inner',validate='one_to_one')
    mismatch=(merged['HOME_PTS']!=merged['SOURCE_HOME_PTS']) | (merged['AWAY_PTS']!=merged['SOURCE_AWAY_PTS'])
    rejected=int(mismatch.sum()); merged=merged.loc[~mismatch].copy()
    if merged.empty:raise ValueError('No score-verified quote/game matches')
    output=[]
    for typ,forecast,threshold,actual,market in (
        ('spreads','home_margin_forecast',-merged['close_spread_home'], 'home_margin','close_home_cover_prob'),
        ('totals','total_forecast',merged['close_total'],'total','close_over_prob'),
    ):
        t=merged.copy()
        t['target']=typ
        t['threshold']=threshold
        t['actual']=t[actual]
        t['won']=np.where(t['actual']>t['threshold'],1,np.where(t['actual']<t['threshold'],0,np.nan))
        t['model_p']=_prob(t[forecast],t['threshold'],sigmas['home_margin' if typ=='spreads' else 'total'])
        t['market_p']=t[market]
        t['calibrated_p']=np.nan
        t['phase']=np.select([
            (t.GAME_DATE>=pd.Timestamp(CAL_START))&(t.GAME_DATE<pd.Timestamp(CAL_END)),
            (t.GAME_DATE>=pd.Timestamp(TEST_START))&(t.GAME_DATE<pd.Timestamp(TEST_END)),
        ],['calibration_2024_25','test_2025_26'],default='not_audited')
        output.append(t)
    audit=pd.concat(output,ignore_index=True)
    metrics={}
    for typ in ('spreads','totals'):
        sel=audit[(audit.target==typ)&audit.won.notna()].copy()
        cal=sel[sel.phase=='calibration_2024_25']
        test=sel[sel.phase=='test_2025_26']
        info={'validation_game_count':len(cal),'test_game_count':len(test),
              'test_push_count':int(((audit.target==typ)&(audit.phase=='test_2025_26')&audit.won.isna()).sum()),
              'test_baseline':_score(test,'model_p'),'test_market_consensus_reference':_score(test,'market_p'),
              'test_calibrated_on_support':None,'test_baseline_on_support':None,
              'test_support_games':0,'isotonic_knots':None,'benchmark_status':'insufficient_evidence'}
        if len(cal)>=min_cal and len(test)>=min_test and cal['won'].nunique()==2:
            iso=IsotonicRegression(y_min=0,y_max=1,out_of_bounds='clip')
            iso.fit(cal['model_p'],cal['won'])
            xs=np.asarray(iso.X_thresholds_,dtype=float);ys=np.asarray(iso.y_thresholds_,dtype=float)
            if xs.size>=2 and np.all(np.diff(xs)>0):
                supported=(test.model_p>=xs.min())&(test.model_p<=xs.max())
                support=test.loc[supported]
                if len(support)>=min_test:
                    values=np.clip(np.interp(support.model_p,xs,ys),0.005,0.995)
                    audit.loc[support.index,'calibrated_p']=values
                    info['test_calibrated_on_support']=_score(audit.loc[support.index],'calibrated_p')
                    info['test_baseline_on_support']=_score(support,'model_p')
                    info['test_support_games']=len(support)
                    info['isotonic_knots']={'x':xs.tolist(),'y':ys.tolist()}
                    info['benchmark_status']='retrospective_closing_line_evaluated'
        metrics[typ]=info
    payload={
        'schema':1,'created_utc':datetime.now(timezone.utc).isoformat(),
        'validation_scope':'free_consensus_closing_lines_without_quote_timestamps',
        'has_verified_quote_publication_times':False,
        'has_bookmaker_executable_prices':False,
        'usable_for_prior_cutoff_ev_backtests':False,
        'usable_for_live_probability_replacement':False,
        'method':('Frozen Ridge trained before 2023-10-01; sigma from 2023-24 heldout games; '
                  'isotonic fitted on 2024-25 closing-line outcomes; untouched 2025-26 test.'),
        'training_games':len(train),'sigma_games':len(scale),
        'sigmas':{k:round(v,5) for k,v in sigmas.items()},
        'score_mismatched_quote_rows_rejected':rejected,
        'date_team_score_verified_games':len(merged),
        'targets':metrics,
        'notice':('Comparison uses closing consensus lines assembled retrospectively. '
                  'The quote capture times are unknown, so closing lines cannot support claims '
                  'about prices available earlier, executable ROI or Polymarket pricing. '
                  'Market consensus probability is a no-vig reference, not an offered price. '
                  'Push games excluded from Brier and ECE.'),
    }
    return payload,audit


def _write_json(path:Path, data:dict)->None:
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    os.replace(temp,path)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest='command',required=True)
    d=sub.add_parser('download',help='Download two free CSV datasets; no paid APIs or keys')
    d.add_argument('--data-dir',type=Path,default=DATA_DIR)
    d.add_argument('--overwrite',action='store_true')
    e=sub.add_parser('evaluate',help='Evaluate closing-line benchmark against completed NBA game logs')
    e.add_argument('--data-dir',type=Path,default=DATA_DIR)
    e.add_argument('--report',type=Path,default=DEFAULT_REPORT)
    e.add_argument('--audit',type=Path,default=DEFAULT_AUDIT)
    r=sub.add_parser('report',help='Read existing saved closing-line results')
    r.add_argument('--path',type=Path,default=DEFAULT_REPORT)
    args=p.parse_args(argv)
    if args.command=='download':
        print(json.dumps(download_files(args.data_dir,overwrite=args.overwrite),indent=2))
    elif args.command=='evaluate':
        quotes,source_stats=load_closing_quotes(args.data_dir)
        print('Reading cached completed NBA results (no odds network/API calls)...')
        completed=read_completed_history(ROOT)
        payload,audit=evaluate(completed,quotes)
        payload['datasets']=source_stats
        _write_json(args.report,payload)
        args.audit.parent.mkdir(parents=True,exist_ok=True)
        audit.to_csv(args.audit,index=False)
        for key,value in payload['targets'].items():
            model=value['test_baseline'] or {}
            market=value['test_market_consensus_reference'] or {}
            cal=value['test_calibrated_on_support'] or {}
            print(f'{key}: {value["benchmark_status"]} | val {value["validation_game_count"]} '
                  f'| test {value["test_game_count"]} | model Brier {model.get("brier")} '
                  f'| market-ref Brier {market.get("brier")} '
                  f'| calibrated-on-support Brier {cal.get("brier")}')
        print('Saved:',args.report,'and',args.audit)
        print('CLOSING-LINE BENCHMARK ONLY. No verified quote times, executable prices or EV.')
    elif args.command=='report':
        if not args.path.is_file():
            print('No saved closing-line benchmark. Run download, then evaluate.')
        else:
            print(args.path.read_text(encoding='utf-8'))
    return 0

if __name__=='__main__':
    raise SystemExit(main())
