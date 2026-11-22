'use strict';
const assert = require('node:assert/strict');
const path = require('node:path');
const {pgAllocateMoneylines,pgETDay,teamFullName,PG_TEAMS} = require(path.resolve(__dirname,'../static/moneyline_budget.js'));
const now='2026-11-20T22:00:00Z',day='2026-11-20';
const make=(slug,p=.82,ask=.68,shares=1000,date=day,tip='2026-11-21T01:00:00Z') => ({
  slug,game_date:date,tipoff_utc:tip,away:'IND',home:'CLE',
  winner_forecast: p == null ? null : {predicted_winner:'CLE',predicted_winner_probability:p},
  moneylines:ask == null?{}:{CLE:{ask,shares}}
});
function allocate(games,overrides={}){return pgAllocateMoneylines({games,now,updated_at_utc:now,bankroll:1000,dailyPct:.05,perGamePct:.02,minEv:.05,...overrides});}
const one=allocate([make('a')]);
assert.equal(one.today,'2026-11-20');
assert.equal(one.spent,20);
assert.equal(one.rows[0].stake,20);
assert.ok(Math.abs(one.rows[0].ev-(.82/.68-1))<1e-10);
assert.equal(one.active,1);
assert.equal(one.unallocated,30);
const committed=allocate([make('a'),make('b')],{committed:35});
assert.equal(committed.dailyCap,50);
assert.equal(committed.spent,15);
assert.equal(committed.unallocated,0);
assert.ok(committed.rows[0].stake<=20);
assert.equal(allocate([make('a')],{committed:50}).spent,0);
const many=allocate(['a','b','c','d','e'].map(id=>make(id)));
assert.equal(many.spent,50);
assert.deepEqual(many.rows.map(r=>r.stake),[10,10,10,10,10]);
assert.equal(many.dailyCap,50);
const negative=allocate([make('a',.67,.70)]);
assert.equal(negative.rows[0].stake,0);
assert.match(negative.rows[0].reason,/below/);
assert.equal(allocate([make('a',.75,.72)],{minEv:.10}).spent,0);
const missing=allocate([make('a',.82,null),make('b',null,.68),make('c',.82,.68,0)]);
assert.equal(missing.spent,0);
assert.equal(missing.rows.length,3);
const liquidity=allocate([make('a',.82,.68,10)]);
assert.equal(liquidity.spent,6.8);
assert.equal(liquidity.rows[0].stake,6.8);
const old=allocate([make('a')],{updated_at_utc:'2026-11-20T21:40:00Z'});
assert.equal(old.spent,0);assert.equal(old.stale,true);
const alreadyStarted=allocate([make('a',.82,.68,1000,day,'2026-11-20T21:00:00Z')]);
assert.equal(alreadyStarted.spent,0);
const tomorrow=allocate([make('a',.82,.68,1000,'2026-11-21')]);
assert.equal(tomorrow.totalGames,0);
const badBudget=allocate([make('a')],{bankroll:0});
assert.equal(badBudget.valid,false);assert.equal(badBudget.spent,0);
const small=allocate([make('a')],{bankroll:1});
assert.ok(small.spent<=.02);
assert.equal(teamFullName('DET'),'Detroit Pistons');
assert.equal(teamFullName('NOP'),'New Orleans Pelicans');
assert.equal(teamFullName('XYZ'),'XYZ');
assert.equal(Object.keys(PG_TEAMS).length,30);
assert.equal(pgETDay('2026-11-21T03:00:00Z'),'2026-11-20');
assert.equal(pgETDay('2026-11-21T18:00:00Z'),'2026-11-21');
console.log('Moneyline bankroll JS: 17 cases passed');
