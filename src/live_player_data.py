"""Append feature-compatible *completed* 2026-27 games to a COPY of player history.

Does not overwrite data/processed/player_training_data_v2.csv. New NBA player
logs are incorporated chronologically using the same audited feature builder.
Unknown players aren't assigned made-up 20-game histories; they stay unavailable.
"""
from __future__ import annotations
from datetime import date
from pathlib import Path
import pandas as pd


def _minutes(x):
    if isinstance(x,str) and ':' in x:
        a,b=x.split(':',1)
        try: return float(a)+float(b)/60.0
        except ValueError: return float('nan')
    return pd.to_numeric(x,errors='coerce')


def fetch_completed_player_games(season):
    from nba_api.stats.endpoints import playergamelogs
    df=playergamelogs.PlayerGameLogs(season_nullable=season,
       season_type_nullable='Regular Season',timeout=60).get_data_frames()[0]
    if df.empty: raise ValueError('NBA API returned no current-season completed player logs')
    return df


def build_overlay(base:pd.DataFrame, raw:pd.DataFrame, cutoff:date):
    from parlay_guard.src.player_prop_common import prepare_history, build_feature_row, season_of_date, ROLLING_STATS
    original_columns=list(base.columns)
    history=prepare_history(base)
    prior=history.groupby('PLAYER_ID',sort=False)
    by_id={str(int(pid)): g.copy() for pid,g in prior if pd.notna(pid)}
    incoming=raw.copy()
    for c in ('GAME_ID','GAME_DATE','PLAYER_ID','PLAYER_NAME','TEAM_ID'):
        if c not in incoming: raise ValueError(f'NBA player game logs lack {c}')
    if 'TOV' not in incoming and 'TO' in incoming:
        incoming['TOV']=incoming['TO']
    if 'IS_HOME' not in incoming:
        if 'MATCHUP' not in incoming:raise ValueError('Player logs missing MATCHUP/IS_HOME')
        incoming['IS_HOME']=incoming['MATCHUP'].astype(str).str.contains('vs.',regex=False)
    incoming['GAME_DATE']=pd.to_datetime(incoming['GAME_DATE'],errors='coerce').dt.normalize()
    incoming=incoming[(incoming['GAME_DATE'].notna()) &
                      (incoming['GAME_DATE'] < pd.Timestamp(cutoff))].copy()
    incoming['GAME_ID']=incoming['GAME_ID'].astype(str).str.replace(r'\.0$','',regex=True).str.zfill(10)
    incoming['PLAYER_ID']=pd.to_numeric(incoming['PLAYER_ID'],errors='coerce')
    for c in ROLLING_STATS:
        if c not in incoming: raise ValueError(f'NBA player game logs missing {c}')
        incoming[c]=incoming[c].map(_minutes) if c=='MIN' else pd.to_numeric(incoming[c],errors='coerce')
    incoming=incoming.dropna(subset=['PLAYER_ID','TEAM_ID','PTS','REB','AST','FG3M','MIN','FGA','FG3A','FTA','TOV'])
    incoming=incoming.sort_values(['GAME_DATE','GAME_ID']).drop_duplicates(['PLAYER_ID','GAME_ID'])
    new=[];skipped=set()
    for pid,g in incoming.groupby('PLAYER_ID',sort=False):
        key=str(int(pid))
        prior_games=by_id.get(key)
        if prior_games is None or len(prior_games)<20:
            skipped.add(key);continue
        prior_games=prior_games.copy()
        known=set(prior_games['GAME_ID'].astype(str).str.zfill(10)) if 'GAME_ID' in prior_games else set()
        last_date=prior_games['GAME_DATE'].max()
        for record in g.itertuples(index=False):
            gid=str(record.GAME_ID)
            if gid in known: continue
            day=pd.Timestamp(record.GAME_DATE)
            # If earlier gaps appear, rebuilding the whole engineered archive is required.
            if day <= last_date:continue
            season=season_of_date(day)
            home=record.IS_HOME
            try:
                home=int(str(home).strip().lower() in ('true','1')) if not isinstance(home,bool) else int(home)
                fields=build_feature_row(prior_games,target_day=day,
                                         season=season,is_home=home)
            except (ValueError,KeyError):continue
            new_row={c:None for c in original_columns}
            new_row.update(fields)
            new_row.update({'GAME_ID':gid,'GAME_DATE':day,'PLAYER_ID':int(pid),
                'PLAYER_NAME':str(record.PLAYER_NAME),'TEAM_ID':int(record.TEAM_ID),
                'SEASON':season,'IS_HOME':home})
            for stat in ROLLING_STATS:
                new_row[stat]=float(getattr(record,stat))
            new.append(new_row)
            prior_games=pd.concat([prior_games,pd.DataFrame([new_row])],ignore_index=True)
            known.add(gid);last_date=day
    if not new:
        return base.copy(),{'new_game_rows':0,'untracked_player_ids':len(skipped)}
    result=pd.concat([base,pd.DataFrame(new)],ignore_index=True)
    result['GAME_ID']=result['GAME_ID'].astype(str).str.replace(r'\.0$','',regex=True).str.zfill(10)
    if result.duplicated(['GAME_ID','PLAYER_ID']).any():
        raise ValueError('Appending current player logs would create duplicate player/game rows')
    return result,{'new_game_rows':len(new),'untracked_player_ids':len(skipped)}


def update_live_history(base_path:Path,overlay_path:Path,season:str,cutoff:date):
    base=pd.read_csv(base_path,low_memory=False,dtype={'GAME_ID':str})
    raw=fetch_completed_player_games(season)
    built,stats=build_overlay(base,raw,cutoff)
    overlay_path.parent.mkdir(parents=True,exist_ok=True)
    temp=overlay_path.with_suffix('.tmp')
    built.to_csv(temp,index=False)
    temp.replace(overlay_path)
    stats['latest_player_game']=str(pd.to_datetime(built.GAME_DATE).max().date())
    return stats
