"""Parlay Guard: moneyline / singles / same-game / cross-game strategy dashboard.

Read-only. Prop parlay prices are hypothetical PRODUCTS of CLOB single-leg
asks, NOT exchange Combo quotes. No parlay EV is claimed. No orders are placed.

Quickstart:
  python add_parlay_comparison.py --stake 20 --record-paper
  start models/parlay_guard_v1_vs_v2.html

Refresh polymarket_nba_ev.py and screen_prop_ev.py before running.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import html
import json
import math
from pathlib import Path
import re
from urllib.parse import quote

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

GAMMA = 'https://gamma-api.polymarket.com'
CLOB = 'https://clob.polymarket.com'
BEGIN = '<!-- PARLAY_GUARD_COMPARISON_START -->'
END = '<!-- PARLAY_GUARD_COMPARISON_END -->'
STYLE_BEGIN = '/* PARLAY_GUARD_COMPARISON_CSS_START */'
STYLE_END = '/* PARLAY_GUARD_COMPARISON_CSS_END */'
JS_BEGIN = '/* PARLAY_GUARD_COMPARISON_JS_START */'
JS_END = '/* PARLAY_GUARD_COMPARISON_JS_END */'
REQUIRED = {'event', 'event_slug', 'player_name', 'stat', 'side', 'line',
            'ask_price', 'stressed_ev_roi', 'observed_utc', 'game_start_utc'}


def esc(x):
    return html.escape(str(x), quote=True)


def finite(x):
    try:
        y = float(x)
        return y if math.isfinite(y) else None
    except (TypeError, ValueError):
        return None


def decode_list(x):
    if isinstance(x, list):
        return x
    if isinstance(x, str):
        try:
            r = json.loads(x)
            return r if isinstance(r, list) else []
        except (ValueError, TypeError):
            pass
    return []


def american(price):
    if price is None or not 0 < price < 1:
        return '—'
    decimal = 1.0 / price
    return f'+{round(100 * (decimal - 1))}' if decimal >= 2 else f'-{round(100 / (decimal - 1))}'


def money(x):
    return f'${x:,.2f}'


def leg_key(row):
    phrase = '|'.join([str(row['event_slug']), str(row['player_name']).casefold(),
                       str(row['stat']), str(row['side']).upper(), f"{float(row['line']):.4f}"])
    return 'prop_' + hashlib.sha256(phrase.encode('utf-8')).hexdigest()[:18]


def load_legs(path, max_age_min, *, max_games=4, now=None):
    """Choose TWO DISTINCT players from each of up to max_games games.

    Each selected leg is used exactly once in every compared prop strategy.
    Games with only one screened player are not forced into a parlay.
    """
    df = pd.read_csv(path)
    missing = REQUIRED - set(df.columns)
    if missing:
        raise ValueError(f'{path}: missing columns {sorted(missing)}. Run screen_prop_ev.py first.')
    for col in ('observed_utc', 'game_start_utc'):
        df[col] = pd.to_datetime(df[col], utc=True, errors='coerce')
    for col in ('line', 'ask_price', 'stressed_ev_roi', 'ask_size_shares', 'model_probability'):
        if col in df:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    if now is None:
        now = pd.Timestamp.now(tz='UTC')
    else:
        now = pd.Timestamp(now)
        now = now.tz_localize('UTC') if now.tzinfo is None else now.tz_convert('UTC')
    keep = (df['game_start_utc'].notna() & (df['game_start_utc'] > now)
            & df['observed_utc'].notna() & (df['observed_utc'] <= now)
            & ((now - df['observed_utc']) <= pd.Timedelta(minutes=max_age_min))
            & df['ask_price'].between(0.01, 0.99)
            & df['stressed_ev_roi'].notna()
            & df['event_slug'].notna() & df['player_name'].notna())
    if 'outcome_mapping' in df:
        keep &= df['outcome_mapping'].eq('description_yes_over')
    if 'book_status' in df:
        keep &= df['book_status'].eq('priced')
    if 'side' in df:
        keep &= df['side'].isin(['OVER', 'UNDER'])
    eligible = df.loc[keep].sort_values(['stressed_ev_roi', 'ask_price'], ascending=[False, True]).copy()
    eligible = eligible.drop_duplicates(['event_slug', 'player_name'], keep='first')
    groups = []
    for slug, g in eligible.groupby('event_slug', sort=False):
        if len(g) < 2:
            continue
        pair = g.head(2)
        groups.append((float(pair['stressed_ev_roi'].sum()), str(slug), pair))
    groups.sort(key=lambda g: (-g[0], g[1]))
    chosen = groups[:max_games]
    legs = []
    for _, _, pair in chosen:
        for rec in pair.to_dict('records'):
            legs.append({
                'id': leg_key(rec), 'event': str(rec['event']), 'event_slug': str(rec['event_slug']),
                'player': str(rec['player_name']), 'stat': str(rec['stat']),
                'side': str(rec['side']), 'line': float(rec['line']),
                'ask': float(rec['ask_price']),
                'ask_size': finite(rec.get('ask_size_shares')),
                'model_probability': finite(rec.get('model_probability')),
                'stressed_roi': float(rec['stressed_ev_roi']),
                'observed_utc': rec['observed_utc'].isoformat(),
                'tipoff_utc': rec['game_start_utc'].isoformat(),
                'market_id': str(rec.get('market_id') or ''),
            })
    return legs, {'eligible_rows': int(len(eligible)), 'eligible_games_with_pair': len(groups),
                  'selected_games': len(chosen), 'excluded_single_pick_games': int(eligible['event_slug'].nunique() - len(groups))}


def build_tickets(legs):
    groups = defaultdict(list)
    for item in legs:
        groups[item['event_slug']].append(item)
    ordered = list(groups.values())
    if any(len(g) != 2 for g in ordered):
        raise ValueError('Every selected game must have exactly two distinct players')
    singles = [[leg] for game in ordered for leg in game]
    same = [list(game) for game in ordered]
    cross = []
    if len(ordered) >= 2:
        cross = [[ordered[i][0], ordered[(i+1) % len(ordered)][1]] for i in range(len(ordered))]
    for strat in (same, cross):
        if strat:
            ids = [x['id'] for ticket in strat for x in ticket]
            assert len(ids) == len(set(ids)) == len(legs)
    def serialize(name, legs_list):
        result = []
        for idx, ticket in enumerate(legs_list):
            p = math.prod(leg['ask'] for leg in ticket)
            result.append({'id': f'{name}_{idx+1}', 'strategy': name, 'leg_ids': [x['id'] for x in ticket],
                           'cost': p, 'american': american(p), 'decimal': 1/p,
                           'events': sorted(set(x['event_slug'] for x in ticket)),
                           'kind': 'single' if len(ticket) == 1 else 'hypothetical_pair'})
        return result
    return {'singles': serialize('singles', singles),
            'same_game': serialize('same_game', same),
            'cross_game': serialize('cross_game', cross)}


def http_session():
    s = requests.Session()
    s.headers.update({'User-Agent': 'ParlayGuardComparison/2.0', 'Accept': 'application/json'})
    retry = Retry(total=2, backoff_factor=0.4, status_forcelist=[429, 500, 502, 503, 504],
                  allowed_methods=frozenset(['GET', 'POST']))
    s.mount('https://', HTTPAdapter(max_retries=retry))
    return s


def best_ask(book):
    pairs = []
    for level in book.get('asks') or []:
        if not isinstance(level, dict):
            continue
        p, size = finite(level.get('price')), finite(level.get('size'))
        if p is not None and size is not None and 0 < p < 1 and size > 0:
            pairs.append((p, size))
    return min(pairs, key=lambda x: x[0]) if pairs else None


def fetch_moneylines(slugs, *, session=None):
    """Fetch best ASK by team for each game. Never substitute Gamma midpoint prices."""
    own = session is None
    if own:
        session = http_session()
    results, tokens = {}, {}
    try:
        for slug in slugs:
            results[slug] = {'status': 'unavailable', 'event': slug, 'sides': []}
            try:
                r = session.get(GAMMA + '/events/slug/' + quote(slug, safe=''), timeout=(5, 12))
                r.raise_for_status()
                e = r.json()
                if not isinstance(e, dict) or str(e.get('slug')) != slug:
                    continue
                results[slug]['event'] = str(e.get('title') or slug)
                for market in e.get('markets') or []:
                    if str(market.get('sportsMarketType') or '').lower() != 'moneyline':
                        continue
                    if market.get('closed') or market.get('active') is False or market.get('acceptingOrders') is False:
                        continue
                    outcomes, ids = decode_list(market.get('outcomes')), decode_list(market.get('clobTokenIds'))
                    if len(outcomes) != 2 or len(ids) != 2:
                        continue
                    names = [str(a).strip() for a in outcomes]
                    if (len(set(names)) != 2 or any(a.casefold() in ('yes','no','draw','tie') for a in names)):
                        continue
                    for team, token in zip(names, ids):
                        token = str(token)
                        tokens[token] = slug
                        results[slug]['sides'].append({'team': team, 'ask': None, 'size': None, 'token': token})
                    results[slug]['status'] = 'awaiting_books'
                    break
            except (requests.RequestException, ValueError, TypeError):
                results[slug]['status'] = 'gamma_unavailable'
        for offset in range(0, len(tokens), 100):
            batch_ids = list(tokens)[offset:offset+100]
            try:
                r = session.post(CLOB + '/books', json=[{'token_id': t} for t in batch_ids], timeout=(5, 15))
                r.raise_for_status()
                books = r.json()
                if not isinstance(books, list):
                    raise ValueError('Bad books response')
                for book in books:
                    if not isinstance(book, dict):
                        continue
                    token = str(book.get('asset_id') or book.get('token_id') or '')
                    if token not in tokens:
                        continue
                    quoted = best_ask(book)
                    if quoted is not None:
                        for side in results[tokens[token]]['sides']:
                            if side['token'] == token:
                                side['ask'], side['size'] = quoted
            except (requests.RequestException, ValueError):
                for token in batch_ids:
                    results[tokens[token]]['status'] = 'books_unavailable'
        for item in results.values():
            if item['status'] == 'awaiting_books':
                item['status'] = 'priced' if any(s['ask'] for s in item['sides']) else 'no_asks'
    finally:
        if own:
            session.close()
    return results


def benchmark_moneylines(legs, quotes):
    by_game = {}
    for leg in legs:
        slug = leg['event_slug']
        if slug not in by_game:
            by_game[slug] = {'slug': slug, 'event': leg['event'], 'sides': [], 'chosen': None}
    for slug, game in by_game.items():
        raw = quotes.get(slug, {})
        game['sides'] = [dict(team=x['team'], ask=x['ask'], size=x['size']) for x in raw.get('sides', [])]
        priced = [(i, x) for i, x in enumerate(game['sides']) if x['ask'] is not None]
        # If both quoted teams cost far more than $1 combined, the market is
        # exceptionally wide; show quote information but omit benchmark P/L.
        if len(priced) == 2 and sum(x['ask'] for _, x in priced) <= 1.25:
            game['chosen'] = max(priced, key=lambda pair: pair[1]['ask'])[0]
        elif priced:
            game['quote_warning'] = 'Both sides incomplete or unusually wide; benchmark disabled'
    return list(by_game.values())


CSS = r'''
/* PARLAY_GUARD_COMPARISON_CSS_START */
#pg-parlay-comparison{--pg-bg:#101e32;--pg-panel:#182a43;--pg-border:#35506b;--pg-light:#eef5ff;--pg-muted:#a9bfd7;max-width:1260px;margin:28px 0;padding:clamp(14px,3vw,27px);background:linear-gradient(125deg,#10203a,#11243d 70%,#122e3a);border:1px solid #35516c;border-radius:18px;color:var(--pg-light);font:14px/1.5 system-ui,Segoe UI,sans-serif}
#pg-parlay-comparison *{box-sizing:border-box}#pg-parlay-comparison h2{font-size:clamp(20px,3vw,26px);margin:0;color:#f5faff}#pg-parlay-comparison h3{font-size:16px;margin:0;color:#f5faff}#pg-parlay-comparison h4{font-size:14px;margin:0;color:#f5faff}#pg-parlay-comparison p{margin:6px 0 10px;color:var(--pg-muted)}
#pg-parlay-comparison .pg-top{display:flex;justify-content:space-between;gap:18px;align-items:center;flex-wrap:wrap}#pg-parlay-comparison .pg-tag{color:#a7d3ff;font-size:11px;font-weight:750;text-transform:uppercase;letter-spacing:.12em}#pg-parlay-comparison .pg-time{color:#b8c8d9;font-size:12px}#pg-parlay-comparison .pg-muted{color:var(--pg-muted);font-size:12px}
#pg-parlay-comparison .pg-controls{display:grid;grid-template-columns:minmax(220px,1fr) 1fr;gap:18px;margin:18px 0;padding:16px;border:1px solid var(--pg-border);border-radius:12px;background:rgba(13,26,44,.75)}#pg-parlay-comparison input[type=range]{width:100%;accent-color:#75b6ff}#pg-parlay-comparison .pg-mini{font-size:12px;color:#bfd0e5}
#pg-parlay-comparison .pg-score{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px;margin:14px 0}#pg-parlay-comparison .pg-scorecard{min-width:0;background:var(--pg-panel);border:1px solid var(--pg-border);border-radius:12px;padding:15px;cursor:pointer;text-align:left;color:var(--pg-light);font:inherit}#pg-parlay-comparison .pg-scorecard[aria-pressed=true]{outline:2px solid #78c4ff;background:#213a59}#pg-parlay-comparison .pg-scorecard:focus-visible,#pg-parlay-comparison button:focus-visible,#pg-parlay-comparison select:focus-visible{outline:2px solid #9dcaff;outline-offset:3px}#pg-parlay-comparison .pg-name{display:block;color:#c4d5e9;font-size:12px;font-weight:650}#pg-parlay-comparison .pg-number{display:block;font-size:24px;font-weight:800;font-variant-numeric:tabular-nums;margin:3px 0}#pg-parlay-comparison .pg-meta{display:block;font-size:11px;color:#acbdd4}#pg-parlay-comparison .pg-up{color:#9ef0c0}#pg-parlay-comparison .pg-down{color:#ffb3b1}
#pg-parlay-comparison .pg-layout{display:grid;grid-template-columns:1fr 1.55fr;gap:14px;margin-top:18px}#pg-parlay-comparison .pg-panel{min-width:0;background:rgba(11,24,41,.5);border:1px solid var(--pg-border);border-radius:13px;padding:15px}#pg-parlay-comparison .pg-game{padding:12px;background:#15253c;border:1px solid #334963;border-radius:10px;margin:9px 0}#pg-parlay-comparison .pg-gamehead{display:flex;justify-content:space-between;gap:8px;align-items:center}#pg-parlay-comparison .pg-toggle{display:flex;gap:6px;padding:5px 0;align-items:center;justify-content:space-between;border-top:1px solid #334862}#pg-parlay-comparison .pg-switch{background:#253d56;border:1px solid #53708d;padding:5px 12px;border-radius:7px;color:#f1f6fc;cursor:pointer;font:inherit;font-size:12px;min-width:56px}#pg-parlay-comparison .pg-switch[aria-pressed=true]{background:#1a654a;border-color:#46b891;color:#fff}#pg-parlay-comparison .pg-switch[aria-pressed=false]{background:#6d3e46;border-color:#bf7275;color:#fff}
#pg-parlay-comparison .pg-ticket{background:#15263f;border:1px solid #364d68;border-radius:10px;padding:13px;margin:10px 0}#pg-parlay-comparison .pg-ticket-head{display:flex;justify-content:space-between;gap:10px;flex-wrap:wrap;align-items:center}#pg-parlay-comparison .pg-odds{font-size:19px;font-weight:800;font-variant-numeric:tabular-nums}#pg-parlay-comparison .pg-ticket-leg{font-size:12px;border-top:1px solid #30445f;padding:6px 0;display:flex;gap:10px;justify-content:space-between}#pg-parlay-comparison .pg-price{color:#a7bed9;white-space:nowrap}#pg-parlay-comparison .pg-numbers{font-size:12px;color:#d6e6f5;margin-top:9px}#pg-parlay-comparison .pg-numbers b{font-size:14px;color:#fff}#pg-parlay-comparison .pg-result{font-size:11px;color:#a0dfc6}#pg-parlay-comparison a{color:#a1ccff;text-decoration:underline}#pg-parlay-comparison select{max-width:100%;padding:6px;background:#1b304a;color:#f5f9ff;border:1px solid #6786a2;border-radius:7px}
#pg-parlay-comparison .pg-hidden{display:none!important}#pg-parlay-comparison .pg-callout{padding:11px 12px;margin:14px 0 3px;border-left:3px solid #d5ac67;background:#2b2a36;color:#f4dfb7;font-size:12px}#pg-parlay-comparison .pg-empty{padding:20px;border:1px dashed #64758b;border-radius:10px;color:#c7d7e5}#pg-parlay-comparison .pg-kpi{display:flex;gap:17px;flex-wrap:wrap;color:#aec6dd;font-size:12px}#pg-parlay-comparison .pg-kpi b{color:white}#pg-parlay-comparison .pg-table{width:100%;border-collapse:collapse;font-size:12px}#pg-parlay-comparison .pg-table td,#pg-parlay-comparison .pg-table th{padding:9px 7px;border-bottom:1px solid #334963;text-align:left}#pg-parlay-comparison .pg-table td:last-child,#pg-parlay-comparison .pg-table th:last-child{text-align:right}
@media(max-width:950px){#pg-parlay-comparison .pg-score{grid-template-columns:repeat(2,minmax(0,1fr))}#pg-parlay-comparison .pg-layout{grid-template-columns:1fr}}@media(max-width:550px){#pg-parlay-comparison .pg-controls{grid-template-columns:1fr}#pg-parlay-comparison .pg-score{grid-template-columns:1fr 1fr}#pg-parlay-comparison .pg-number{font-size:20px}}
/* PARLAY_GUARD_COMPARISON_CSS_END */
'''

JS = r'''
/* PARLAY_GUARD_COMPARISON_JS_START */
(()=>{
  const root=document.getElementById('pg-parlay-comparison');if(!root)return;
  const dataNode=root.querySelector('#pg-strategy-data');if(!dataNode)return;
  let data;try{data=JSON.parse(dataNode.textContent);}catch(e){return;}
  let active='same_game';const state={};const mlState={};
  const dollars=x=>'$'+x.toLocaleString('en-US',{minimumFractionDigits:2,maximumFractionDigits:2});
  const odds=p=>(!(p>0&&p<1))?'—':(p<=0.5?'+'+Math.round(100*(1/p-1)):'-'+Math.round(100/(1/p-1)));
  for(const l of data.legs)state[l.id]=l.event_slug!==data.first_game;
  for(const m of data.moneylines)mlState[m.slug]=m.slug!==data.first_game;
  const priceById=Object.fromEntries(data.legs.map(l=>[l.id,l]));
  const chooseSide=(m)=>{const sel=root.querySelector('[data-ml-select="'+m.slug+'"]');const ix=sel?Number(sel.value):m.chosen;return Number.isInteger(ix)&&ix>=0&&ix<m.sides.length?m.sides[ix]:null;};
  function mlTickets(){return data.moneylines.filter(m=>{const s=chooseSide(m);return s&&s.ask>0&&s.ask<1;}).map(m=>{
    const s=chooseSide(m);return {cost:s.ask,team:s.team,size:s.size,event:m.event,slug:m.slug,won:mlState[m.slug]};
  });}
  function calculate(name,budget){
    if(name==='moneyline'){
      const t=mlTickets();if(!t.length)return null;const part=budget/t.length;
      return {count:t.length,gross:t.reduce((a,b)=>a+(b.won?part/b.cost:0),0),full:t.reduce((a,b)=>a+part/b.cost,0)};
    }
    const tickets=data.tickets[name];if(!tickets||!tickets.length)return null;
    const part=budget/tickets.length;
    return {count:tickets.length,gross:tickets.reduce((acc,t)=>acc+(t.leg_ids.every(id=>state[id])?part/t.cost:0),0),full:tickets.reduce((acc,t)=>acc+part/t.cost,0)};
  }
  function update(){
    const budget=Number(root.querySelector('#pg-budget').value);
    root.querySelector('#pg-budget-label').textContent=dollars(budget);
    for(const name of ['singles','same_game','cross_game','moneyline']){
      const card=root.querySelector('[data-pg-strategy="'+name+'"]');if(!card)continue;
      const r=calculate(name,budget);const pnl=card.querySelector('.pg-number'),desc=card.querySelector('.pg-meta');
      if(!r){pnl.textContent='N/A';desc.textContent='Insufficient eligible market data';continue;}
      const profit=r.gross-budget;
      pnl.textContent=(profit>=0?'+':'−')+dollars(Math.abs(profit));
      pnl.classList.toggle('pg-up',profit>=0);pnl.classList.toggle('pg-down',profit<0);
      desc.textContent=r.count+' tickets · '+dollars(budget/r.count)+'/ticket · '+dollars(r.gross)+' scenario payout';
    }
    for(const b of root.querySelectorAll('[data-pg-leg-result]')){
      const won=state[b.dataset.pgLegResult];b.setAttribute('aria-pressed',String(won));b.textContent=won?'WIN':'LOSS';
    }
    for(const b of root.querySelectorAll('[data-pg-ml-result]')){
      const won=mlState[b.dataset.pgMlResult];b.setAttribute('aria-pressed',String(won));b.textContent=won?'WIN':'LOSS';
    }
    for(const ticket of root.querySelectorAll('.pg-ticket[data-pg-ticket]')){
      const name=ticket.dataset.pgTicket;const ix=Number(ticket.dataset.pgIndex);
      const t=data.tickets[name]?.[ix];if(!t)continue;
      const part=budget/data.tickets[name].length;
      const won=t.leg_ids.every(id=>state[id]);
      ticket.querySelector('.pg-bet-size').textContent=dollars(part);
      ticket.querySelector('.pg-if-win').textContent=dollars(part/t.cost);
      ticket.querySelector('.pg-ticket-result').textContent=won?'Scenario: WIN':'Scenario: LOSS';
      ticket.querySelector('.pg-ticket-result').classList.toggle('pg-down',!won);
    }
    for(const m of data.moneylines){
      const box=[...root.querySelectorAll('[data-pg-moneyline]')].find(x=>x.dataset.pgMoneyline===m.slug);
      if(!box||m.chosen===null)continue;const side=chooseSide(m),count=mlTickets().length;
      const s=side&&side.ask?side:null;
      box.querySelector('.pg-ml-odds').textContent=s?odds(s.ask):'—';
      box.querySelector('.pg-ml-ask').textContent=s?'Ask '+dollars(s.ask)+' / share':'Quote unavailable';
      box.querySelector('.pg-ml-payout').textContent=s&&count?dollars(budget/count/s.ask):'—';
      box.querySelector('.pg-ml-depth').textContent=s&&count&&s.size!=null&&s.ask*s.size < budget/count?'Insufficient top-ask depth for this stake':'Top-of-book depth/fees may change';
    }
    const panel=root.querySelector('#pg-selected-strategy');const r=calculate(active,budget);
    if(panel){panel.querySelector('#pg-selected-title').textContent=({singles:'Individual prop singles',same_game:'Separate same-game 2-leg tickets',cross_game:'Separate cross-game 2-leg tickets',moneyline:'Moneyline single-game tickets'})[active];
      panel.querySelector('#pg-selected-description').textContent=r?'Same total budget: '+dollars(budget)+' · '+r.count+' tickets · all-ticket-win payout '+dollars(r.full):'No comparable tickets currently available.';}
  }
  root.querySelector('#pg-budget')?.addEventListener('input',update);
  for(const b of root.querySelectorAll('[data-pg-leg-result]'))b.addEventListener('click',()=>{const id=b.dataset.pgLegResult;state[id]=!state[id];update();});
  for(const b of root.querySelectorAll('[data-pg-ml-result]'))b.addEventListener('click',()=>{const id=b.dataset.pgMlResult;mlState[id]=!mlState[id];update();});
  for(const s of root.querySelectorAll('[data-ml-select]'))s.addEventListener('change',update);
  for(const b of root.querySelectorAll('[data-pg-strategy]'))b.addEventListener('click',()=>{
    active=b.dataset.pgStrategy;
    for(const c of root.querySelectorAll('[data-pg-strategy]'))c.setAttribute('aria-pressed',String(c===b));
    for(const c of root.querySelectorAll('[data-pg-ticket-panel]'))c.classList.toggle('pg-hidden',c.dataset.pgTicketPanel!==active);
    update();
  });
  update();
})();
/* PARLAY_GUARD_COMPARISON_JS_END */
'''


def ticket_html(ticket, idx, by_id):
    parts = [f'<article class="pg-ticket" data-pg-ticket="{esc(ticket["strategy"])}" data-pg-index="{idx}">',
             f'<div class="pg-ticket-head"><b>Ticket {idx+1} · {len(ticket["leg_ids"])} leg(s)</b>',
             f'<span class="pg-odds">{esc(ticket["american"])}</span></div>',
             f'<p class="pg-muted">{ticket["decimal"]:.2f}× · reference cost ${ticket["cost"]:.4f}</p>']
    for id in ticket['leg_ids']:
        l = by_id[id]
        url = 'https://polymarket.com/event/' + quote(l['event_slug'], safe='')
        parts.append('<div class="pg-ticket-leg"><span>' +
                     f'<a target="_blank" rel="noopener noreferrer" href="{esc(url)}">{esc(l["player"])} {esc(l["side"])} {l["line"]:g} {esc(l["stat"])}</a>' +
                     f'<br><span class="pg-muted">{esc(l["event"])}</span></span><span class="pg-price">${l["ask"]:.3f}</span></div>')
    parts.append('<div class="pg-numbers">Stake <b class="pg-bet-size">—</b> · If all win <b class="pg-if-win">—</b></div><div class="pg-result pg-ticket-result">Scenario</div></article>')
    return ''.join(parts)


def html_fragment(legs, tickets, moneylines, stake, stats, fetch_time, max_age_min=90, paper_summary=None):
    by_id = {x['id']: x for x in legs}
    root = [BEGIN, '<section id="pg-parlay-comparison" aria-label="Parlay Guard moneyline and parlay strategy comparison">',
            '<div class="pg-top"><div><div class="pg-tag">Parlay Guard · Strategy Lab</div><h2>Singles vs. per-game vs. cross-game parlays</h2></div>',
            f'<span class="pg-time">Generated {esc(fetch_time)}</span></div>',
            '<p>Compare the <b>same screened player props and same total stake</b> across three hypothetical ticket structures. Moneylines are included as a separate odds benchmark, not picks recommended by the player model.</p>',
            '<div class="pg-controls"><div><label for="pg-budget">Total budget per strategy: <b id="pg-budget-label">'+money(stake)+'</b></label>',
            f'<input id="pg-budget" type="range" min="5" max="200" step="5" value="{int(stake)}" aria-label="Total budget per strategy">',
            '<div class="pg-mini">Budget is split equally across tickets in each strategy.</div></div>',
            '<div><b>Interactive outcome simulator</b><p class="pg-mini">Toggle WIN/LOSS for each prop below. The scenario starts with the first selected game losing both props and other games winning. These are <b>not predictions or actual game results</b>.</p></div></div>',
            '<div class="pg-score">']
    configs = [('singles', 'Single props', 'One ticket per prop'),
               ('same_game', 'Separate same-game parlays', '2 props in one game per ticket'),
               ('cross_game', 'Separate cross-game parlays', '2 games per ticket'),
               ('moneyline', 'Moneyline benchmark', 'Illustrative price-implied favorite')]
    for name, label, info in configs:
        root.append(f'<button type="button" class="pg-scorecard" data-pg-strategy="{name}" aria-pressed="{str(name == "same_game").lower()}">'
                    f'<span class="pg-name">{esc(label)}</span><strong class="pg-number">—</strong><span class="pg-meta">{esc(info)}</span></button>')
    root.append('</div><div class="pg-kpi">'
                f'<span>Eligible after fresh-quote screening: <b>{stats["eligible_rows"]}</b></span>'
                f'<span>Games with two distinct players: <b>{stats["eligible_games_with_pair"]}</b></span>'
                f'<span>Games compared: <b>{stats["selected_games"]}</b></span>'
                f'<span>Shared props: <b>{len(legs)}</b></span>'
                '</div><div class="pg-layout"><div class="pg-panel"><h3>Scenario controls</h3>')
    if not legs:
        root.append('<div class="pg-empty">No games have two distinct, freshly screened player props. No parlays will be invented. Refresh the scanner/shortlist and try again.</div>')
    else:
        seen = set()
        for leg in legs:
            if leg['event_slug'] not in seen:
                if seen:
                    root.append('</div>')
                root.append(f'<div class="pg-game"><div class="pg-gamehead"><b>{esc(leg["event"])}</b></div>')
                seen.add(leg['event_slug'])
            root.append(f'<div class="pg-toggle"><div><b>{esc(leg["player"])}</b><div class="pg-muted">{esc(leg["side"])} {leg["line"]:g} {esc(leg["stat"])}</div></div>'
                        f'<button type="button" class="pg-switch" data-pg-leg-result="{esc(leg["id"])}" aria-pressed="false">LOSS</button></div>')
        root.append('</div>')
    root.append('</div><div class="pg-panel" id="pg-selected-strategy"><h3 id="pg-selected-title">Separate same-game 2-leg tickets</h3>'
                '<p id="pg-selected-description" class="pg-muted">Hypothetical tickets</p>')
    for name in ('singles', 'same_game', 'cross_game'):
        hidden = ' pg-hidden' if name != 'same_game' else ''
        root.append(f'<div class="pg-ticket-list{hidden}" data-pg-ticket-panel="{name}">')
        if not tickets[name]:
            root.append('<div class="pg-empty">Not enough distinct games / players to construct this strategy using the same props.</div>')
        for i, ticket in enumerate(tickets[name]):
            root.append(ticket_html(ticket, i, by_id))
        root.append('</div>')
    root.append('<div class="pg-ticket-list pg-hidden" data-pg-ticket-panel="moneyline">')
    if not moneylines or not any(m['chosen'] is not None for m in moneylines):
        root.append('<div class="pg-empty">Moneyline CLOB asks unavailable. No odds substituted.</div>')
    for m in moneylines:
        root.append(f'<div class="pg-game" data-pg-moneyline="{esc(m["slug"])}"><h4>{esc(m["event"])}</h4>')
        priced = [(i, side) for i, side in enumerate(m['sides']) if side['ask'] is not None]
        if m['chosen'] is None:
            if priced:
                root.append('<p class="pg-muted">Market asks available but benchmark excluded: both sides unavailable or the total ask exceeds $1.25.</p>')
                for _, side in priced:
                    root.append(f'<div class="pg-ticket-leg"><span>{esc(side["team"])}</span><span>${side["ask"]:.3f} · {american(side["ask"])}</span></div>')
            else:
                root.append('<p>Moneyline ask unavailable.</p>')
        else:
            root.append(f'<div class="pg-ticket-leg"><label for="pg-ml-{esc(m["slug"])}">Choose side (not a recommendation)</label>'
                        f'<select id="pg-ml-{esc(m["slug"])}" data-ml-select="{esc(m["slug"])}">')
            for ix, side in priced:
                selected = ' selected' if ix == m['chosen'] else ''
                root.append(f'<option value="{ix}"{selected}>{esc(side["team"])} · ask ${side["ask"]:.3f}</option>')
            root.append('</select></div><div class="pg-ticket-head"><span class="pg-odds pg-ml-odds">—</span>'
                        '<span class="pg-muted pg-ml-ask">—</span></div>'
                        '<div class="pg-numbers">Payout if won <b class="pg-ml-payout">—</b></div>'
                        '<div class="pg-muted pg-ml-depth">—</div>'
                        f'<div class="pg-toggle"><span class="pg-muted">Simulate selected side outcome</span><button type="button" class="pg-switch" data-pg-ml-result="{esc(m["slug"])}" aria-pressed="false">LOSS</button></div>')
        root.append('</div>')
    root.append('</div></div></div>')
    root.append('<div class="pg-callout"><b>Research-only comparison.</b> Moneyline prices are actual CLOB best asks at generation time; their implied odds are not model recommendations. '
                'Singles use actual top asks as references. <b>Same-game and cross-game parlay prices are products of individual asks, not executable Polymarket Combo quotes.</b> '
                'Same-game legs can be correlated; cross-game independence is also not guaranteed. Fees, slippage, voids, player nonappearance, and injury risk are excluded. '
                'A higher payout does not mean higher profitability. Same-game tickets isolate a loss to one game; cross-game tickets link games on the same ticket.</div>')
    root.append('<h3 style="margin-top:18px">Paper-trading comparison</h3>')
    if not paper_summary:
        root.append('<p class="pg-muted">No completed paper-trading snapshots yet. Run this script with <code>--record-paper</code>, '
                    'update completed player/team game logs, then run <code>paper_strategy_results.py</code>. '
                    'Only fully settled snapshots appear; results use hypothetical combo-reference prices, not realized exchange fills.</p>')
    else:
        root.append('<div class="pg-panel"><table class="pg-table"><thead><tr><th>Strategy</th><th>Complete paper sessions</th><th>Hypothetical ROI</th></tr></thead><tbody>')
        for r in paper_summary:
            roi = finite(r.get('roi'))
            val = f'{roi:+.1%}' if roi is not None else 'N/A'
            root.append(f'<tr><td>{esc(r.get("strategy", ""))}</td><td>{int(r.get("complete_sessions", 0))}</td><td>{val}</td></tr>')
        root.append('</tbody></table><p class="pg-muted">Historical scenarios use recorded pregame quotes. Missing or unresolved sessions are excluded. Venue-executable Combo prices not verified.</p></div>')
    root.append(f'<p class="pg-muted">Snapshot age limit: {max_age_min} minutes at generation. '
                'Game roster, actual availability, and full settlement terms were not automatically verified. '
                'Regenerate after refreshing CSVs; changing the slider does not refresh prices.</p>')
    # Only data passed via trusted serialized JSON, never JS literal interpolation.
    data = {'legs': legs, 'tickets': tickets, 'moneylines': moneylines,
            'first_game': legs[0]['event_slug'] if legs else ''}
    serialized = json.dumps(data, ensure_ascii=True, separators=(',', ':'), allow_nan=False).replace('<','\\u003c')
    root.append(f'<script id="pg-strategy-data" type="application/json">{serialized}</script></section>{END}')
    return ''.join(root)


def replace_marked(doc, begin, end, replacement):
    return re.sub(re.escape(begin) + r'.*?' + re.escape(end), lambda _:replacement, doc, count=1, flags=re.S)


def patch_dashboard(doc, fragment):
    # Compatible with prior add_parlay_comparison marker block; no duplicates on rerun.
    if BEGIN in doc and END in doc:
        doc = replace_marked(doc, BEGIN, END, fragment)
    elif '<div class="live-toolbar">' in doc:
        doc = doc.replace('<div class="live-toolbar">', fragment+'\n<div class="live-toolbar">', 1)
    elif '</body>' in doc:
        doc = doc.replace('</body>', fragment+'\n</body>', 1)
    else:
        raise ValueError('Cannot locate dashboard insertion point')
    if STYLE_BEGIN in doc and STYLE_END in doc:
        doc = replace_marked(doc, STYLE_BEGIN, STYLE_END, CSS.strip())
    elif '</style>' in doc:
        doc = doc.replace('</style>', CSS+'\n</style>', 1)
    elif '</head>' in doc:
        doc = doc.replace('</head>', '<style>'+CSS+'</style></head>', 1)
    else:
        raise ValueError('Cannot locate HTML head')
    script = '<script>\n'+JS+'\n</script>'
    pattern = re.compile(r'<script>\s*'+re.escape(JS_BEGIN)+r'.*?'+re.escape(JS_END)+r'\s*</script>', re.S)
    if JS_BEGIN in doc and JS_END in doc:
        doc = pattern.sub(lambda _:script, doc, count=1)
    elif '</body>' in doc:
        doc = doc.replace('</body>', script+'\n</body>', 1)
    else:
        raise ValueError('Cannot locate HTML body end')
    return doc


def standalone_page(fragment):
    return ("<!doctype html><html lang='en'><head><meta charset='utf-8'>"
            "<meta name='viewport' content='width=device-width,initial-scale=1'>"
            "<title>Parlay Guard — Strategy Lab</title>"
            "<style>body{margin:25px auto;max-width:1300px;background:#0c1220;padding:0 15px;}"+CSS+"</style></head><body>"
            +fragment+'<script>'+JS+'</script></body></html>')


def record_snapshot(path, *, legs, tickets, moneylines, budget, now):
    path.mkdir(parents=True, exist_ok=True)
    when = now.strftime('%Y%m%dT%H%M%S_%fZ')
    file = path / f'strategy_snapshot_{when}.json'
    payload = {'version':2,'observed_utc':now.isoformat(),'budget':budget,
               'legs':legs,'tickets':tickets,'moneylines':moneylines,
               'note':'Hypothetical parlay prices based on single-leg CLOB asks, not combo fills.'}
    file.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding='utf-8')
    return file


def load_paper_summary(path):
    if not path.is_file():
        return []
    df = pd.read_csv(path)
    needed = {'strategy','complete_sessions','roi'}
    if not needed.issubset(df):
        return []
    return df.to_dict('records')


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--shortlist',type=Path,default=Path('data/prop_ev_research_shortlist.csv'))
    ap.add_argument('--dashboard',type=Path,default=Path('models/parlay_guard_v1_vs_v2.html'))
    ap.add_argument('--standalone',type=Path,default=Path('models/parlay_comparison.html'))
    ap.add_argument('--stake',type=int,default=20,help='Equal total stake per strategy, $5–$200')
    ap.add_argument('--max-age-min',type=int,default=90)
    ap.add_argument('--max-games',type=int,default=4,help='Maximum games with 2 distinct screened props (2–8)')
    ap.add_argument('--record-paper',action='store_true',help='Save a pregame paper snapshot for later settlement')
    ap.add_argument('--paper-dir',type=Path,default=Path('data/paper_trades'))
    ap.add_argument('--no-moneylines',action='store_true',help='Do not request API moneyline quotes')
    args = ap.parse_args(argv)
    if not args.shortlist.is_file():
        ap.error(f'Missing {args.shortlist}. Run screen_prop_ev.py first.')
    if not 5 <= args.stake <= 200 or args.stake % 5 or args.max_age_min <= 0 or not 2 <= args.max_games <= 8:
        ap.error('Stake must be a multiple of $5 from $5–$200, age must be positive, max-games 2–8')
    now = datetime.now(timezone.utc)
    legs, stats = load_legs(args.shortlist,args.max_age_min,max_games=args.max_games,now=now)
    tickets = build_tickets(legs)
    game_slugs = list(dict.fromkeys(l['event_slug'] for l in legs))
    quotes = fetch_moneylines(game_slugs) if game_slugs and not args.no_moneylines else {}
    ml = benchmark_moneylines(legs, quotes)
    stamp = now.strftime('%Y-%m-%d %H:%M UTC')
    history = load_paper_summary(args.paper_dir / 'strategy_paper_summary.csv')
    fragment = html_fragment(legs,tickets,ml,args.stake,stats,stamp,args.max_age_min,paper_summary=history)
    args.standalone.parent.mkdir(parents=True,exist_ok=True)
    args.standalone.write_text(standalone_page(fragment),encoding='utf-8')
    print('Screened outcomes:',stats['eligible_rows'],'| Games with pairs:',stats['eligible_games_with_pair'],
          '| Selected legs:',len(legs))
    print('Moneyline games with quoted sides:',sum(any(x['ask'] is not None for x in m['sides']) for m in ml))
    print('Saved standalone:',args.standalone)
    if args.dashboard.exists():
        before=args.dashboard.read_text(encoding='utf-8')
        backup=args.dashboard.with_suffix(args.dashboard.suffix+'.before_parlay.bak')
        if not backup.exists():
            backup.write_text(before,encoding='utf-8')
        after=patch_dashboard(before,fragment)
        args.dashboard.write_text(after,encoding='utf-8')
        print('Updated existing dashboard:',args.dashboard)
    else:
        print('Dashboard not found; standalone file is ready. Set --dashboard to your actual HTML path.')
    if args.record_paper:
        if not legs:
            print('No eligible legs; paper snapshot not saved')
        else:
            p=record_snapshot(args.paper_dir,legs=legs,tickets=tickets,moneylines=ml,budget=args.stake,now=now)
            print('Saved research paper snapshot:',p)
    print('WARNING: Parlay odds are hypothetical products of individual asks, not actual Combo quotes.')


if __name__ == '__main__':
    main()
