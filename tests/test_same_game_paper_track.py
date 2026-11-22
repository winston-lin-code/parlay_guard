"""Offline tests: no network, bets, external feeds or historical backfills."""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from src import same_game_paper_track as sg

NOW = datetime(2026, 11, 20, 23, 20, tzinfo=timezone.utc)


def sample(now=NOW, prop=True):
    tip = now + timedelta(minutes=40)
    return {
        'updated_at_utc': (now-timedelta(minutes=1)).isoformat(),
        'games': [{
            'game_date': '2026-11-20', 'home': 'ATL', 'away': 'CHA',
            'slug': 'nba-cha-atl-2026-11-20', 'tipoff_utc': tip.isoformat(),
            'winner_forecast': {'home_probability': 0.65, 'predicted_winner': 'ATL',
                                'forecast_generated_at_utc': (now-timedelta(minutes=30)).isoformat(),
                                'nba_game_id': '0022600010'},
            'moneylines': {'ATL': {'ask': .70, 'shares': 100, 'token_id':'ml-token'}},
            'same_game_prop_candidates': ([{
                'player': 'Casey Player', 'team': 'CHA', 'stat': 'PTS',
                'side': 'OVER', 'line': 20.5, 'ask': .55,
                'model_probability': .74, 'stressed_ev_roi': .10,
                'observed_utc': (now-timedelta(minutes=3)).isoformat(),
                'top_ask_shares': 80,
            }] if prop else []),
        }],
    }


@pytest.fixture
def prepared(tmp_path):
    cfg = sg.init(tmp_path, at=NOW-timedelta(minutes=5))
    return tmp_path, cfg


def identifiers(snapshot):
    return {(snapshot['games'][0]['slug'], 'caseyplayer'): '42'}


def record(prepared, *, prop=True):
    root, cfg = prepared
    snap=sample(prop=prop)
    result, why = sg.candidate(snap['games'][0], NOW, NOW-timedelta(minutes=1), cfg,
                                identifiers(snap))
    assert why is None
    return result


def test_freeze_idempotent(prepared):
    root, cfg = prepared
    original = sg.init(root, at=NOW + timedelta(days=7))
    assert original == cfg
    assert original['frozen_at_utc'] != (NOW + timedelta(days=7)).isoformat()
    assert len(original['config_hash']) == 64


def test_prop_from_other_team_is_used(prepared):
    r = record(prepared)
    assert r['moneyline']['team'] == 'ATL'
    assert r['prop']['team'] == 'CHA'  # Does not force predicted winner's team.
    assert r['strategies']['moneyline_only'] == [{'kind': 'moneyline', 'stake': 1.0}]
    assert r['strategies']['prop_only'] == [{'kind': 'prop', 'stake': 1.0}]
    assert r['strategies']['best_positive_ev'] == [{'kind': 'prop', 'stake': 1.0}]
    assert r['strategies']['both_singles'] == [
        {'kind': 'moneyline', 'stake': .5}, {'kind': 'prop', 'stake': .5}]
    assert sum(i['stake'] for i in r['strategies']['both_singles']) == 1.0


def test_no_prop_still_bets_every_moneyline(prepared):
    r = record(prepared,prop=False)
    assert r['prop'] is None
    assert r['moneyline']['estimated_ev'] < 0
    assert r['strategies']['moneyline_only']
    for name in ('prop_only', 'best_positive_ev', 'both_singles'):
        assert r['strategies'][name] == []


def test_quote_validation_and_window(prepared):
    root,cfg=prepared
    snap=sample()
    g=snap['games'][0]
    g['moneylines']['ATL']['shares']=0
    v,why=sg.candidate(g,NOW,NOW,cfg,identifiers(snap))
    assert v is None and why=='moneyline_no_liquidity'
    g['moneylines']['ATL']['shares']=100
    v,why=sg.candidate(g,NOW-timedelta(hours=2),NOW-timedelta(hours=2),cfg,identifiers(snap))
    assert v is None and why=='outside_shared_capture_window'
    v,why=sg.candidate(g,NOW+timedelta(hours=1),NOW+timedelta(hours=1),cfg,identifiers(snap))
    assert v is None and why=='not_pretipoff'
    g['same_game_prop_candidates'][0]['observed_utc']=(NOW-timedelta(hours=1)).isoformat()
    v,why=sg.candidate(g,NOW,NOW,cfg,identifiers(snap))
    assert why is None and v['prop'] is None and v['strategies']['moneyline_only']


def test_capture_never_overwrites_existing_game(prepared,monkeypatch):
    root,cfg=prepared
    snap=sample()
    monkeypatch.setattr(sg,'player_ids',lambda _root: identifiers(snap))
    first=sg.capture(snap,root,at=NOW)
    assert len(first['new'])==1
    path=root/sg.STORE/'picks'/'2026-11-20_CHA_ATL.json'
    raw=path.read_bytes()
    snap['games'][0]['moneylines']['ATL']['ask']=.11
    second=sg.capture(snap,root,at=NOW+timedelta(minutes=1))
    assert not second['new']
    assert path.read_bytes()==raw


def test_capture_never_backfills_before_freeze(tmp_path):
    sg.init(tmp_path,at=NOW)
    with pytest.raises(ValueError,match='backdate'):
        sg.capture(sample(),tmp_path,at=NOW-timedelta(seconds=1))


def _write_game(root):
    path=root/'data/nba_2026_27_latest_raw.csv';path.parent.mkdir(parents=True,exist_ok=True)
    pd.DataFrame([
        {'GAME_ID':'0022600010','GAME_DATE':'2026-11-20','TEAM_ABBREVIATION':'ATL',
         'TEAM_ID':'1610612737','IS_HOME':True,'PTS':120,'WL':'W'},
        {'GAME_ID':'0022600010','GAME_DATE':'2026-11-20','TEAM_ABBREVIATION':'CHA',
         'TEAM_ID':'1610612766','IS_HOME':False,'PTS':115,'WL':'L'},
    ]).to_csv(path,index=False)


def _write_player(root):
    path=root/'data/live_player_training_data_v2.csv';path.parent.mkdir(parents=True,exist_ok=True)
    pd.DataFrame([{'GAME_ID':'0022600010','PLAYER_ID':'42','TEAM_ID':'1610612766',
        'MIN':30,'PTS':24,'REB':5,'AST':3,'FG3M':2}]).to_csv(path,index=False)


def _freeze_record(root,r):
    path=root/sg.STORE/'picks'/(r['game_key']+'.json')
    assert sg.freeze(path,r)


def test_moneyline_can_settle_before_player_results(prepared):
    root,cfg=prepared
    r=record(prepared);_freeze_record(root,r);_write_game(root)
    first=sg.settle(root)
    assert first['newly_settled']==1
    summary=sg.report(root)
    a={x['strategy']:x for x in summary['strategies']}
    assert a['moneyline_only']['settled']==1
    assert a['prop_only']['pending']==1
    assert a['both_singles']['pending']==1
    assert a['moneyline_only']['paper_roi']==pytest.approx(1/.70-1, abs=1e-6)
    _write_player(root)
    second=sg.settle(root)
    assert second['newly_settled']==1
    again=sg.settle(root)
    assert again['newly_settled']==0
    summary=sg.report(root)
    a={x['strategy']:x for x in summary['strategies']}
    assert a['prop_only']['paper_roi']==pytest.approx(1/.55-1, abs=1e-6)
    assert a['both_singles']['paper_roi']==pytest.approx(.5*(1/.70-1)+.5*(1/.55-1), abs=1e-6)
    assert a['best_positive_ev']['settled']==1
    assert summary['games_with_both_quotes']==1


def test_missing_prop_means_paired_population_empty(prepared):
    root,cfg=prepared
    r=record(prepared,prop=False);_freeze_record(root,r);_write_game(root)
    sg.settle(root)
    s=sg.report(root)
    assert s['captured_games']==1
    assert s['games_with_both_quotes']==0
    assert next(x for x in s['strategies'] if x['strategy']=='moneyline_only')['settled']==1
    assert next(x for x in s['paired_games_strategies'] if x['strategy']=='moneyline_only')['settled']==0


def test_tampering_is_rejected(prepared):
    root,cfg=prepared
    r=record(prepared);_freeze_record(root,r)
    path=root/sg.STORE/'picks'/(r['game_key']+'.json')
    x=json.loads(path.read_text());x['moneyline']['yes_ask']=.01
    path.write_text(json.dumps(x))
    with pytest.raises(ValueError,match='checksum'):
        sg.report(root)


def test_installer_patch_parses_and_is_idempotent():
    # Unit-test the injected snippets without importing external data models.
    import importlib.util
    from pathlib import Path
    spec=importlib.util.spec_from_file_location('same_game_installer',Path(__file__).parents[1]/'install_same_game_compare.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    api='''from fastapi import FastAPI, HTTPException\nfrom fastapi.responses import HTMLResponse,FileResponse\nfrom pathlib import Path\nROOT=Path('.')\napp=FastAPI()\nclass LiveState:\n    def update(self):\n        if True:\n            output={}\n            if True:\n                # PARLAY_GUARD_MONEYLINE_PAPER_TRACKER_V1\n                # Only publish fully assembled snapshots.\n'''
    # The exact anchor has 16 spaces to match the existing server loop.
    patched,changed=module.update_api(api)
    assert changed and patched.count(module.MARKER)==2
    same,changed=module.update_api(patched)
    assert not changed and same==patched
    live='''def eligible_props():\n    row={}\n    item={\n              'top_ask_shares':as_num(row.get('ask_size_shares'))}\n    return item\ndef assemble_data():\n    gs={\n            'moneylines':moneyline,'selected_props':chosen,\n    }\n    return gs\n'''
    live2,changed=module.update_live(live)
    assert changed and 'same_game_prop_candidates' in live2 and "'observed_utc'" in live2
    live3,changed=module.update_live(live2)
    assert not changed and live3==live2
