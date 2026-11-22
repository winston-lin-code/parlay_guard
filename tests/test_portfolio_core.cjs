const assert=require('node:assert/strict');
const P=require('../static/portfolio_core.js');
const NOW=Date.parse('2026-11-21T21:00:00Z'),start='2026-11-21T23:00:00Z',stamp='2026-11-21T20:57:00Z';
let passed=0;
function test(title,fn){fn();passed++;process.stdout.write('PASS '+title+'\n')}
const cfg={bankroll:40,riskPct:5,perGamePct:2,kellyPct:25,committed:0,minEV:5,feeCents:1,allowFlagged:false};
function game(id,p,ask,other={}){return {id,sport:'NHL',start_utc:start,matchup:'Alpha '+id+' @ Beta '+id,options:[{id:'away',name:'Alpha',probability:p,ask,fee:0,observed_utc:stamp,shares:null,depth_unit:'shares'}],modelValid:true,riskReview:false,...other}}
test('two-side NBA feed computes home/away probability and team names',()=>{
 const g=P.normalizeNBA({updated_at_utc:stamp,games:[{slug:'nba-det-nyk-2026-11-21',tipoff_utc:start,home:'NYK',away:'DET',winner_forecast:{home_probability:.7},moneylines:{NYK:{ask:.57,shares:40},DET:{ask:.45,shares:30}}}]})[0];
 assert(Math.abs(g.options[0].probability-.3)<1e-10);assert.equal(g.options[1].probability,.7);assert.equal(g.options[1].name,'New York Knicks');assert.match(g.options[1].logo,/1610612752/);
});
test('college injuries are visible and default allocation excludes flagged games',()=>{
 const g=P.normalizeCollege({games:[{espn_id:'123',sport:'mbb',start_utc:start,status:'ROSTER_INJURY_REVIEW',away:{id:'a',display_name:'Duke'},home:{id:'b',display_name:'UNC'},predictions:{away:.6,home:.4},market:{away:{ask:.4,shares:50,observed_utc:stamp}},injuries:{away:{injuries:[{player:'Player 1',status:'Out'}]}}}]})[0];
 assert.equal(g.riskReview,true);assert.equal(g.options[0].name,'Duke');assert.match(g.warnings.join(' '),/Out/);
 const m=P.optionMetrics(g.options[0],g,cfg,NOW);assert.equal(m.valid,false);assert.match(m.reason,/review/);
 assert.equal(P.optionMetrics(g.options[0],g,{...cfg,allowFlagged:true},NOW).valid,true);
});
test('correct YES EV and binary-contract Kelly formula',()=>{
 const g=game('a',.70,.50);const m=P.optionMetrics(g.options[0],g,cfg,NOW);
 assert(Math.abs(m.ev-(.7/.51-1))<1e-9);assert(Math.abs(m.fullKelly-(.7-.51)/(.49))<1e-9);
});
test('negative EV generates zero stake even if selected',()=>{
 const g=game('a',.48,.60);let r=P.allocatePortfolio([g],{a:{checked:true,side:'away'}},cfg,NOW);assert.equal(r.rows.length,0);assert.equal(r.total,0);
});
test('stale quote never gets allocation',()=>{
 const g=game('a',.9,.4);g.options[0].observed_utc='2026-11-21T20:00:00Z';const r=P.allocatePortfolio([g],{a:{checked:true,side:'away'}},cfg,NOW);assert.equal(r.rows.length,0);assert.match(r.skipped[0].reason,/older/);
});
test('games too close to tipoff cannot get stakes',()=>{
 const g=game('a',.9,.4,{start_utc:'2026-11-21T21:03:00Z'});const r=P.allocatePortfolio([g],{a:{checked:true,side:'away'}},cfg,NOW);assert.equal(r.rows.length,0);assert.match(r.skipped[0].reason,/tipoff/);
});
test('max per game and daily cap apply; unallocated money stays cash',()=>{
 const games=['a','b','c'].map(x=>game(x,.8,.5));const sel=Object.fromEntries(games.map(g=>[g.id,{checked:true,side:'away'}]));const r=P.allocatePortfolio(games,sel,cfg,NOW);
 assert.equal(r.rows.length,3);assert(r.total<=2.00001);for(const x of r.rows)assert(x.stake<=.8+1e-8);assert(Math.abs(r.rows[0].bankrollPct-100*r.rows[0].stake/40)<1e-10);
});
test('weights favor larger conservative Kelly within per-game cap',()=>{
 const c={...cfg,perGamePct:10,riskPct:20};const gs=[game('a',.75,.5),game('b',.61,.5)];const sel={a:{checked:true,side:'away'},b:{checked:true,side:'away'}};
 const r=P.allocatePortfolio(gs,sel,c,NOW);assert(r.rows.find(x=>x.gameId==='a').stake>r.rows.find(x=>x.gameId==='b').stake);
});
test('top of book depth reduces suggested paper stake',()=>{
 const g=game('a',.8,.5);g.options[0].shares=2;
 const r=P.allocatePortfolio([g],{a:{checked:true,side:'away'}},cfg,NOW);assert(r.total<=.25+1e-8);
});
test('committed money reduces daily risk budget',()=>{
 const g=game('a',.8,.5);const r=P.allocatePortfolio([g],{a:{checked:true,side:'away'}},{...cfg,committed:1.8},NOW);
 assert(r.total<=.2+1e-8);assert(r.availableAfter<=38.2+1e-8);
});
test('external outcomes maintain model and quote provenance',()=>{
 const g=P.normalizeExternal({games:[{id:'abc',sport:'NHL',start_utc:start,matchup:'Bears @ Sharks',source:'Owls Insight published Polymarket EV',venue:'polymarket.com (international; paper only)',outcomes:[{id:'away',name:'Bears',probability:.6,ask:.48,observed_utc:stamp}]}]})[0];
 assert.equal(g.options[0].probability,.6);assert.match(g.source,/Owls/);
});
test('invalid settings fail closed',()=>{
 assert.throws(()=>P.allocatePortfolio([],{},{...cfg,bankroll:0},NOW));assert.throws(()=>P.allocatePortfolio([],{},{...cfg,riskPct:101},NOW));
});
test('schedule-only games have no modeled probability and cannot be allocated',()=>{
  const g=P.normalizeExternal({games:[{id:'espn-1',sport:'NHL',start_utc:start,matchup:'A @ B',provenance:'schedule_only',outcomes:[{id:'away',name:'A',probability:null,ask:null}]}]})[0];
  assert.equal(g.modelValid,false);const r=P.allocatePortfolio([g],{[g.id]:{checked:true,side:'away'}},cfg,NOW);
  assert.equal(r.rows.length,0);
});
test('duplicate same-game feeds cannot double the per-game risk',()=>{
  const a=game('one',.70,.5,{matchup:'Alpha @ Beta'}),b=game('two',.65,.5,{matchup:'Alpha @ Beta'});
  const r=P.allocatePortfolio([a,b],{one:{checked:true,side:'away'},two:{checked:true,side:'away'}},cfg,NOW);
  assert.equal(r.rows.length,1);assert.equal(r.skipped.length,1);assert.match(r.skipped[0].reason,/Duplicate/);
});
console.log(passed+' portfolio core tests passed');
