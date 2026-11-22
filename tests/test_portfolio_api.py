"""Offline compatibility and data-trust tests. No API calls, bets or model writes."""
from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from src import portfolio_api as p

NOW = datetime(2026, 11, 21, 21, 0, tzinfo=timezone.utc)
START = '2026-11-21T23:00:00+00:00'

@pytest.fixture(autouse=True)
def avoid_real_schedules(monkeypatch):
    # Network access is never used in offline tests.
    monkeypatch.setattr(p, 'discover_schedules', lambda now, offset: ([], []))


def owls(**kwargs):
    row={
        'event_id':'owls:nhl:matched-01', 'sport':'NHL','provider':'Owls Insight',
        'venue':'polymarket.com','game':'Bears @ Sharks','start_utc':START,
        'pick':'Bears','side':'away','model_probability':.62,'best_ask':.54,
        'fee_per_share':.002,'quote_age_seconds':80,
        'source_as_of_utc':'2026-11-21T20:58:00+00:00','source_url':'https://example.com/market',
        'liquidity_usd':12
    }
    return {**row,**kwargs}


def manual(**kwargs):
    row={
        'sport':'NFL','event_id':'example-7', 'start_utc':START,
        'away':'Bears', 'home':'Sharks','pick':'Bears', 'market_type':'moneyline_yes',
        'win_probability':'.58','ask':'.50','quote_observed_utc':'2026-11-21T20:58:00+00:00',
        'source':'Independent published model','venue':'polymarket.us',
        'market_url':'https://example.com/markets','fee_per_share':'.005', 'depth_shares':'20',
        'injury_note':'No confirmation of availability'
    }
    return {**row,**kwargs}


def test_owls_supported_and_quote_timestamp_reconstructed():
    r=p.normalized_online(owls(),NOW,0)
    assert r['sport']=='NHL' and r['venue'].startswith('polymarket.com')
    assert r['outcomes'][0]['observed_utc']=='2026-11-21T20:56:40+00:00'
    assert r['outcomes'][0]['depth_unit']=='USD'


@pytest.mark.parametrize('changes',[
    {'provider':'Other'}, {'venue':'sportsbook'}, {'best_ask':1.0},
    {'model_probability':1.2}, {'fee_per_share':.8},
    {'source_as_of_utc':'2026-11-21T21:01:00+00:00'},
    {'start_utc':'2026-11-20T22:00:00+00:00'},
    {'event_id':''}, {'quote_age_seconds':-1}, {'sport':'GAMBLING'},
])
def test_bad_online_rows_rejected(changes):
    assert p.normalized_online(owls(**changes),NOW,0) is None


def test_manual_rejects_non_yes_wrong_venue_and_bad_time():
    assert p.normalized_manual(manual(),NOW,0)['outcomes'][0]['name']=='Bears'
    assert p.normalized_manual(manual(market_type='spread_yes'),NOW,0) is None
    assert p.normalized_manual(manual(venue='sportsbook'),NOW,0) is None
    assert p.normalized_manual(manual(pick='Some other team'),NOW,0) is None
    assert p.normalized_manual(manual(quote_observed_utc=''),NOW,0) is None
    assert p.normalized_manual(manual(start_utc='2026-11-20T23:00:00+00:00'),NOW,0) is None


def test_merge_both_sides_same_game_but_not_conflicting_events():
    a=p.normalized_manual(manual(),NOW,0)
    b=p.normalized_manual(manual(pick='Sharks',win_probability='.42',ask='.51'),NOW,0)
    same=p.merge_events([a,b])
    assert len(same)==1
    assert {x['name'] for x in same[0]['outcomes']}=={'Bears','Sharks'}
    another=p.normalized_manual(manual(home='Sharks',away='Pandas',pick='Pandas'),NOW,0)
    assert len(p.merge_events([a,another]))==2


def test_report_watchlist_never_allocates_sportsbook_lines(tmp_path,monkeypatch):
    p_dir=tmp_path/'data'/'online_ev_plans';p_dir.mkdir(parents=True)
    day='2026-11-21'
    report={'date_et':day,'allocation':{'bets':[owls()]},
            'sportsbook_watchlist':[{'sport':'NHL','game':'Bears @ Sharks','pick':'Bears',
                'fair_probability':.7,'provider_ev_pct':25,'bookmaker':'example',
                'start_utc':START}]}
    (p_dir/(day+'.json')).write_text(json.dumps(report))
    monkeypatch.setattr(p,'ONLINE',p_dir);monkeypatch.setattr(p,'MANUAL',tmp_path/'missing.csv')
    result=p.get_external(NOW,0)
    assert len(result['games'])==1
    assert result['games'][0]['outcomes'][0]['ask']==.54
    assert len(result['sportsbook_watchlist'])==1
    assert result['sportsbook_watchlist'][0]['reason'].startswith('Sportsbook price')


def test_import_malformed_does_not_crash_other_feeds(tmp_path,monkeypatch):
    manual_path=tmp_path/'money.csv'
    with manual_path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(manual()));w.writeheader();w.writerow(manual())
        w.writerow(manual(event_id='bad',ask='NaN'))
    monkeypatch.setattr(p,'MANUAL',manual_path);monkeypatch.setattr(p,'ONLINE',tmp_path/'no-reports')
    out=p.get_external(NOW,0)
    assert len(out['games'])==1
    assert len(out['warnings'])==1
    assert 'row 3 rejected' in out['warnings'][0]


def test_missing_feeds_graceful(tmp_path,monkeypatch):
    monkeypatch.setattr(p,'MANUAL',tmp_path/'missing.csv');monkeypatch.setattr(p,'ONLINE',tmp_path/'missing-dir')
    out=p.get_external(NOW,0)
    assert out['games']==[] and out['warnings']==[] and out['sportsbook_watchlist']==[]


def test_date_offset_filter_correct():
    future=p.normalized_online(owls(start_utc='2026-11-22T23:00:00+00:00'),NOW,1)
    assert future and p.normalized_online(owls(start_utc='2026-11-22T23:00:00+00:00'),NOW,0) is None


def test_installer_atomic_and_idempotent(tmp_path,monkeypatch):
    import install_portfolio as installer
    api_src='from fastapi import FastAPI\napp=FastAPI()\n@app.get("/")\ndef index():return 1\n'
    next_src,changed=installer.patch_api(api_src)
    assert changed and 'app.include_router(portfolio_router)' in next_src
    again,changed=installer.patch_api(next_src)
    assert not changed and next_src==again
    old_live='<html><body><header><div>Hello</div></header><main>Original contents</main></body></html>'
    new_live,changed=installer.patch_live(old_live)
    assert changed and 'Multi-Sport EV Portfolio' in new_live and 'Original contents' in new_live
    assert installer.patch_live(new_live)==(new_live,False)
    assert installer.patch_live('<html></html>')==('<html></html>',False)


def test_espn_schedule_only_never_invents_probabilities():
    e={'id':'789','date':START,'status':{'type':{'state':'pre'}},'competitions':[{'competitors':[
        {'homeAway':'home','team':{'displayName':'Vegas Knights','logos':[{'href':'https://a.espncdn.com/logo.png'}]}},
        {'homeAway':'away','team':{'displayName':'Utah Mammoth','logos':[]}}
    ]}]}
    out=p.espn_schedule_payload({'events':[e]},'NHL',NOW,0)
    assert len(out)==1
    assert out[0]['outcomes'][0]['probability'] is None
    assert out[0]['outcomes'][0]['ask'] is None
    assert out[0]['provenance']=='schedule_only'
    assert p.espn_schedule_payload({'events':[{**e,'status':{'type':{'state':'in'}}}]},'NHL',NOW,0)==[]


def test_priced_feed_replaces_matching_schedule_card():
    sample=p.normalized_online(owls(),NOW,0)
    placeholder={**sample,'id':'espn-schedule:nhl:1','source':'ESPN schedule (unpriced)'}
    result=p.combine_with_schedules([sample],[placeholder])
    assert len(result)==1 and result[0]['id']==sample['id']


def test_schedule_errors_do_not_break_other_sports(monkeypatch,tmp_path):
    monkeypatch.setattr(p,'MANUAL',tmp_path/'missing.csv')
    monkeypatch.setattr(p,'ONLINE',tmp_path/'missing-dir')
    monkeypatch.setattr(p,'discover_schedules',lambda now,offset:([{'id':'s','sport':'NHL','start_utc':START,'matchup':'A @ B',
       'outcomes':[],'venue':'No market quote'}],['NFL schedule unavailable']))
    out=p.get_external(NOW,0)
    assert len(out['games'])==1
    assert 'NFL schedule unavailable' in out['warnings']
