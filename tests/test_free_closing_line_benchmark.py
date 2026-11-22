"""Offline tests for the FREE closing-line-only research add-on."""
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import pytest

from src.free_closing_line_benchmark import (
    normalize_source, keyed_pregame_features, evaluate, TEAM_NAMES,
)


def test_source_filters_regular_and_requires_real_numeric_lines():
    data=pd.DataFrame([
        {'date_et':'2024-10-25','season':'2024-25','season_type':'regular',
         'home_team':'Boston Celtics','away_team':'New York Knicks',
         'home_score':115,'away_score':107,'close_spread_home':-4.5,'close_total':220.5,
         'close_home_cover_prob':.51,'close_over_prob':.49},
        {'date_et':'2025-04-25','season':'2024-25','season_type':'postseason',
         'home_team':'Boston Celtics','away_team':'New York Knicks',
         'home_score':115,'away_score':107,'close_spread_home':-4.5,'close_total':220.5,
         'close_home_cover_prob':.51,'close_over_prob':.49},
    ])
    f,stats=normalize_source(data,'2024-25')
    assert len(f)==1 and f.iloc[0].HOME_TEAM=='BOS'
    assert stats['regular_rows']==1
    assert f.iloc[0].close_spread_home==-4.5


def test_unknown_team_and_duplicate_matchups_rejected():
    row={'date_et':'2024-10-25','season':'2024-25','season_type':'regular',
         'home_team':'Boston Celtics','away_team':'New York Knicks',
         'home_score':115,'away_score':107,'close_spread_home':-4.5,'close_total':220.5,
         'close_home_cover_prob':.51,'close_over_prob':.49}
    f,stats=normalize_source(pd.DataFrame([row,row]),'2024-25')
    assert f.empty and stats['ambiguous_duplicate_rows']==2
    bad={**row,'home_team':'Nonexistent NBA Team'}
    f,_=normalize_source(pd.DataFrame([bad]),'2024-25')
    assert f.empty


def fake_completed():
    rng=np.random.default_rng(98)
    teams=['BOS','NYK','GSW','LAL','CLE','MIA']
    dates=pd.date_range('2019-01-01','2026-05-15',freq='D')
    out=[]
    for i,day in enumerate(dates):
        h=teams[i%6];a=teams[(i+1+(i//6)%4)%6]
        if a==h:a=teams[(i+2)%6]
        out.append({'GAME_ID':f'{i:010d}','GAME_DATE':day,'HOME_TEAM':h,'AWAY_TEAM':a,
          'HOME_PTS':int(113+rng.normal(0,10)), 'AWAY_PTS':int(110+rng.normal(0,10))})
    return pd.DataFrame(out)


def test_pregame_features_never_see_same_game_result():
    games=fake_completed().head(160)
    before=keyed_pregame_features(games)
    mutated=games.copy()
    mutated.loc[mutated.index[-1], 'HOME_PTS']=170
    after=keyed_pregame_features(mutated)
    # Only outcome changes for the final fixture; its pregame features remain unchanged.
    assert len(before)==len(after)
    assert np.allclose(before.iloc[-1][['home_for_5','away_for_5']].to_numpy(dtype=float),
                       after.iloc[-1][['home_for_5','away_for_5']].to_numpy(dtype=float))


def test_closing_line_benchmark_uses_untouched_2025_26_and_never_claims_roi():
    games=fake_completed()
    snapshots=[]
    for g in games.itertuples(index=False):
        day=pd.Timestamp(g.GAME_DATE)
        season='2024-25' if pd.Timestamp('2024-10-01')<=day<pd.Timestamp('2025-07-01') else (
            '2025-26' if pd.Timestamp('2025-10-01')<=day<pd.Timestamp('2026-07-01') else None)
        if season is None:continue
        snapshots.append({'date_et':day.date().isoformat(),'season':season,'season_type':'regular',
            'home_team':TEAM_NAMES[g.HOME_TEAM],'away_team':TEAM_NAMES[g.AWAY_TEAM],
            'home_score':g.HOME_PTS,'away_score':g.AWAY_PTS,
            'close_spread_home':-3.5,'close_total':221.5,
            'close_home_cover_prob':0.50,'close_over_prob':0.50})
    arr=pd.DataFrame(snapshots)
    parts=[]
    for season in ('2024-25','2025-26'):
        part,_=normalize_source(arr,season)
        parts.append(part)
    report,audit=evaluate(games,pd.concat(parts,ignore_index=True),min_test=40)
    assert report['validation_scope']=='free_consensus_closing_lines_without_quote_timestamps'
    assert report['usable_for_prior_cutoff_ev_backtests'] is False
    assert report['usable_for_live_probability_replacement'] is False
    assert report['has_bookmaker_executable_prices'] is False
    assert report['targets']['spreads']['test_game_count'] > 40
    assert report['targets']['totals']['test_baseline']['brier'] is not None
    assert 'calibration_2024_25' in set(audit.phase)
    assert 'test_2025_26' in set(audit.phase)
    # A changed source final score cannot silently enter the audit.
    corrupted=pd.concat(parts,ignore_index=True).copy()
    corrupted.loc[0,'SOURCE_HOME_PTS']=999
    report2,_=evaluate(games,corrupted,min_test=40)
    assert report2['score_mismatched_quote_rows_rejected']==1


def test_free_downloader_rejects_non_csv_response(tmp_path, monkeypatch):
    import requests
    from src.free_closing_line_benchmark import download_files

    class FakeResponse:
        def raise_for_status(self):pass
        def iter_content(self,chunk_size):yield b'<html>not a CSV</html>'
        def close(self):pass
    monkeypatch.setattr(requests,'get',lambda *args,**kwargs: FakeResponse())
    with pytest.raises(ValueError,match='expected CSV'):
        download_files(tmp_path)
    assert not list(tmp_path.glob('*.csv'))
    assert not list(tmp_path.glob('*.partial'))
