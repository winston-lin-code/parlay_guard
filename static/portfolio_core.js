/* Parlay Guard cross-sport moneyline allocation — pure browser/Node functions.
 * All sizing is PAPER ONLY. Never interprets an EV estimate as a proven edge.
 */
(function(root){
'use strict';
const NBA={ATL:['Atlanta Hawks',1610612737],BOS:['Boston Celtics',1610612738],BKN:['Brooklyn Nets',1610612751],CHA:['Charlotte Hornets',1610612766],CHI:['Chicago Bulls',1610612741],CLE:['Cleveland Cavaliers',1610612739],DAL:['Dallas Mavericks',1610612742],DEN:['Denver Nuggets',1610612743],DET:['Detroit Pistons',1610612765],GSW:['Golden State Warriors',1610612744],HOU:['Houston Rockets',1610612745],IND:['Indiana Pacers',1610612754],LAC:['Los Angeles Clippers',1610612746],LAL:['Los Angeles Lakers',1610612747],MEM:['Memphis Grizzlies',1610612763],MIA:['Miami Heat',1610612748],MIL:['Milwaukee Bucks',1610612749],MIN:['Minnesota Timberwolves',1610612750],NOP:['New Orleans Pelicans',1610612740],NYK:['New York Knicks',1610612752],OKC:['Oklahoma City Thunder',1610612760],ORL:['Orlando Magic',1610612753],PHI:['Philadelphia 76ers',1610612755],PHX:['Phoenix Suns',1610612756],POR:['Portland Trail Blazers',1610612757],SAC:['Sacramento Kings',1610612758],SAS:['San Antonio Spurs',1610612759],TOR:['Toronto Raptors',1610612761],UTA:['Utah Jazz',1610612762],WAS:['Washington Wizards',1610612764]};
function number(v){if(v===null||v===undefined||v===''||typeof v==='boolean')return null;const n=Number(v);return Number.isFinite(n)?n:null;}
function dateMs(v){if(typeof v!=='string'||!/(Z|[+-]\d\d:\d\d)$/.test(v))return null;const ms=Date.parse(v);return Number.isFinite(ms)?ms:null;}
function dateET(value){const ms=dateMs(value);if(ms===null)return null;const d=new Intl.DateTimeFormat('en-US',{timeZone:'America/New_York',year:'numeric',month:'2-digit',day:'2-digit'}).formatToParts(new Date(ms));let v={};d.forEach(p=>{v[p.type]=p.value});return `${v.year}-${v.month}-${v.day}`;}
function nbaLogo(abbr){const t=NBA[abbr];return t?`https://cdn.nba.com/logos/nba/${t[1]}/primary/L/logo.svg`:'';}
function nbaName(abbr){return NBA[abbr]?.[0]||abbr;}
function normalizeNBA(payload){const asof=payload?.updated_at_utc||'';const games=[];
 for(const g of payload?.games||[]){if(!g?.slug||!g?.tipoff_utc)continue;let home=String(g.home||''),away=String(g.away||'');const win=g.winner_forecast||null;
  let hp=number(win?.home_probability);if(hp===null&&win?.predicted_winner){let w=number(win.predicted_winner_probability);if(w!==null)hp=win.predicted_winner===home?w:1-w;}
  if(hp!==null&&!(hp>=0&&hp<=1))hp=null;
  const m=g.moneylines||{}, opts=[['away',away, hp===null?null:1-hp],['home',home,hp]].map(([side,abbr,p])=>{
   const q=m[abbr]||{};return {id:side,name:nbaName(abbr),abbr,probability:p,ask:number(q.ask),fee:0,observed_utc:asof,
      shares:number(q.shares),depth_unit:'shares',logo:nbaLogo(abbr)};
  });
  games.push({id:'nba:'+g.slug,sport:'NBA',start_utc:g.tipoff_utc,matchup:`${nbaName(away)} @ ${nbaName(home)}`,options:opts,
   source:'Parlay Guard frozen Elo model / Polymarket.com CLOB',venue:'polymarket.com',warnings:[],riskReview:false,
   modelValid:!!win,sourceInfo:win?.forecast_generated_at_utc||'No fresh NBA forecast',source_url:'',
   note:'Injury/roster status not verified by Elo model. Model probability is unadjusted.'});
 }
 return games;
}
function normalizeCollege(payload){const result=[];
 for(const g of payload?.games||[]){if(!g?.espn_id||!g?.start_utc)continue;
  const sp=g.sport==='mbb'?'NCAAB':g.sport==='cfb'?'NCAAF':String(g.sport||'COLLEGE').toUpperCase();
  const alerts=[];let flagged=false;
  for(const side of ['away','home']){const r=g.injuries?.[side]||{};const nm=g[side]?.display_name||side;
   for(const injury of r.injuries||[]){alerts.push(nm+': '+(injury.player||'Player')+' — '+(injury.status||'reported issue'));flagged=true;}
   for(const delta of r.roster_changes||[]){alerts.push(nm+': '+(delta.player||'Roster')+' — '+(delta.change||'listing changed'));flagged=true;}
   if(!r.injury_status||r.injury_status==='not_checked'||r.injury_status==='unavailable')alerts.push(nm+': injury coverage not verified');
  }
  const opts=['away','home'].map(side=>{const q=g.market?.[side]||{};return {id:side,name:g[side]?.display_name||side,
    probability:number(g.predictions?.[side]),ask:number(q.ask),fee:0,
    observed_utc:q.observed_utc||'',shares:number(q.shares),depth_unit:'shares',logo:g[side]?.logo||''};});
  const unvalidated=g.status==='UNVALIDATED_MODEL'||g.status==='MODEL_NOT_TRAINED'||g.status==='UNKNOWN_TEAM_MODEL'||g.status==='MODEL_ERROR';
  result.push({id:`${sp}:${g.espn_id}`,sport:sp,start_utc:g.start_utc,
    matchup:`${g.away?.display_name||'Away'} @ ${g.home?.display_name||'Home'}`,options:opts,
    source:'Separate trained college model / Polymarket.com CLOB',venue:'polymarket.com',warnings:alerts,
    riskReview:flagged,modelValid:!!g.predictions&&!unvalidated,
    sourceInfo:g.model_metrics?.holdout_games?'Holdout games: '+g.model_metrics.holdout_games:g.status,
    source_url:'',note:'College injury feed is incomplete. Injuries do not automatically adjust model probabilities.'});
 }
 return result;
}
function normalizeExternal(payload){const result=[];
 for(const g of payload?.games||[]){if(!g?.id||!g?.sport||!g?.start_utc)continue;
  const opts=(g.outcomes||[]).map(q=>({id:String(q.id||q.name),name:String(q.name||'Outcome'),
    probability:number(q.probability),ask:number(q.ask),fee:number(q.fee_per_share)||0,
    observed_utc:q.observed_utc||'',shares:number(q.shares),depth_unit:q.depth_unit||'shares',logo:q.logo||''}));
  result.push({id:'ext:'+g.id+':'+g.start_utc,sport:String(g.sport).toUpperCase(),start_utc:g.start_utc,
     matchup:String(g.matchup||''),options:opts,source:String(g.source||''),venue:String(g.venue||''),
     warnings:g.alerts||[],riskReview:false,modelValid:opts.length>0&&g.provenance!=='schedule_only',
     sourceInfo:g.provenance==='user_import'?'Manual import / not independently verified':'Published external estimate',
     source_url:g.source_url||'',note:String(g.warning||'Other sport model/market not independently verified')});
 }
 return result;
}
function optionMetrics(option, game, cfg, nowMs=Date.now()){
  const p=number(option?.probability), ask=number(option?.ask),fee=number(option?.fee)||0;
  if(p===null||!(p>0&&p<1))return {valid:false,reason:'Missing model win probability',ev:null};
  if(ask===null||!(ask>0&&ask<1))return {valid:false,reason:'No verified YES ask',ev:null};
  const c=ask+fee+Math.max(0,number(cfg.feeCents)||0)/100;
  if(c>=1)return {valid:false,reason:'Cost including allowance too high',ev:null};
  const ev=p/c-1, rawKelly=Math.max(0,(p-c)/(1-c));
  const timestamp=dateMs(option.observed_utc), tipoff=dateMs(game.start_utc);
  const quoteAge=timestamp===null?null:(nowMs-timestamp)/1000;
  const remaining=tipoff===null?null:(tipoff-nowMs)/60000;
  let reason='';
  if(!game.modelValid)reason='No validated current model';
  else if(timestamp===null||quoteAge<0||quoteAge>900)reason='Quote older than 15 minutes';
  else if(tipoff===null||remaining<5)reason='Game started or tipoff too close';
  else if(game.riskReview&&!cfg.allowFlagged)reason='Injury/roster review required';
  else if(ev<((number(cfg.minEV)||0)/100))reason='EV below threshold';
  if(!reason&&rawKelly<=0)reason='Non-positive Kelly edge';
  return {valid:!reason,reason,ev,winProbability:p,effectiveCost:c,fullKelly:rawKelly,quoteAge,remaining};
}
function allocatePortfolio(games, choices, cfg, nowMs=Date.now()){
  const bankroll=number(cfg.bankroll),riskPct=number(cfg.riskPct),perGamePct=number(cfg.perGamePct),kelly=number(cfg.kellyPct),spent=number(cfg.committed)||0;
  if(bankroll===null||bankroll<=0||riskPct===null||riskPct<0||riskPct>100||perGamePct===null||perGamePct<0||perGamePct>100||kelly===null||kelly<0||kelly>100||spent<0||spent>bankroll){throw Error('Bankroll and risk settings are invalid');}
  const pool=Math.max(0,Math.min(bankroll-spent,bankroll*riskPct/100-spent));
  const rows=[],skipped=[];
  const offered=[];
  for(const game of games){const choice=choices[game.id];if(!choice?.checked)continue;
   const option=game.options.find(x=>x.id===choice.side)||game.options[0];
   if(!option){skipped.push({game:game.matchup,reason:'No sides available'});continue;}
   const metrics=optionMetrics(option,game,cfg,nowMs);
   if(!metrics.valid){skipped.push({game:game.matchup,pick:option.name,reason:metrics.reason});continue;}
   const raw=bankroll*(kelly/100)*metrics.fullKelly;
   let capped=Math.max(0,Math.min(raw,bankroll*perGamePct/100));
   const depth=number(option.shares),ask=number(option.ask);
   if(depth!==null&&depth>=0){
    const bookUSD=option.depth_unit==='USD'?depth:depth*ask;
    capped=Math.min(capped,bookUSD*.25);
   }
   if(capped<=0){skipped.push({game:game.matchup,pick:option.name,reason:'Insufficient top-of-book depth or stake cap'});continue;}
   offered.push({game,option,metrics,desired:capped});
  }
  // If two feeds describe the same matchup, allocate at most ONE selection
  // for that game, choosing the best currently eligible EV among those
  // explicitly selected. This prevents duplicate per-game exposure.
  const unique=[], seen=new Set();
  const keyOf=x=>x.game.sport+':'+(dateET(x.game.start_utc)||'')+':'+x.game.matchup.toLowerCase().replace(/[^a-z0-9]/g,'');
  offered.sort((a,b)=>b.metrics.ev-a.metrics.ev);
  for(const x of offered){
    const key=keyOf(x);
    if(seen.has(key)){skipped.push({game:x.game.matchup,pick:x.option.name,reason:'Duplicate selected matchup across feeds; higher EV choice retained'});continue;}
    seen.add(key);unique.push(x);
  }
  const totalDesired=unique.reduce((s,r)=>s+r.desired,0),ratio=totalDesired>0?Math.min(1,pool/totalDesired):0;
  for(const x of unique){const pennies=Math.max(0,Math.floor(x.desired*ratio*100+1e-8));if(!pennies){skipped.push({game:x.game.matchup,pick:x.option.name,reason:'Allocation below one cent after caps'});continue;}
   const stake=pennies/100;
   rows.push({sport:x.game.sport,game:x.game.matchup,gameId:x.game.id,
     pick:x.option.name,stake,bankrollPct:100*stake/bankroll,
     winProbability:x.metrics.winProbability,ev:x.metrics.ev,
     ask:x.option.ask,effectiveCost:x.metrics.effectiveCost,
     quoteObservedUTC:x.option.observed_utc,source:x.game.source,venue:x.game.venue,
     modelExpectedProfit:stake*x.metrics.ev,tipoff:x.game.start_utc,
     hasUnknownDepth:x.option.shares===null});
  }
  rows.sort((a,b)=>b.stake-a.stake||b.ev-a.ev);
  const total=rows.reduce((s,r)=>s+r.stake,0);
  return {bankroll,pool,committed:spent,total,availableAfter:Math.max(0,bankroll-spent-total),
    unusedRiskPool:Math.max(0,pool-total),expectedProfit:rows.reduce((s,r)=>s+r.modelExpectedProfit,0),
    rows,skipped,selectedCount:Object.values(choices).filter(c=>c?.checked).length};
}
const api={number,dateMs,dateET,nbaName,nbaLogo,normalizeNBA,normalizeCollege,normalizeExternal,optionMetrics,allocatePortfolio};
root.PGPortfolio=api;
if(typeof module!=='undefined'&&module.exports)module.exports=api;
})(typeof globalThis!=='undefined'?globalThis:this);
