// PARLAY_GUARD_BANKROLL_ALLOCATOR_V1
// Display-only moneyline bankroll allocator. NO trading, order placement, or server writes.
const PG_TEAMS = {
  ATL:['Atlanta','Hawks',1610612737],BOS:['Boston','Celtics',1610612738],CLE:['Cleveland','Cavaliers',1610612739],
  NOP:['New Orleans','Pelicans',1610612740],CHI:['Chicago','Bulls',1610612741],DAL:['Dallas','Mavericks',1610612742],
  DEN:['Denver','Nuggets',1610612743],GSW:['Golden State','Warriors',1610612744],HOU:['Houston','Rockets',1610612745],
  LAC:['Los Angeles','Clippers',1610612746],LAL:['Los Angeles','Lakers',1610612747],MIA:['Miami','Heat',1610612748],
  MIL:['Milwaukee','Bucks',1610612749],MIN:['Minnesota','Timberwolves',1610612750],BKN:['Brooklyn','Nets',1610612751],
  NYK:['New York','Knicks',1610612752],ORL:['Orlando','Magic',1610612753],IND:['Indiana','Pacers',1610612754],
  PHI:['Philadelphia','76ers',1610612755],PHX:['Phoenix','Suns',1610612756],POR:['Portland','Trail Blazers',1610612757],
  SAC:['Sacramento','Kings',1610612758],SAS:['San Antonio','Spurs',1610612759],OKC:['Oklahoma City','Thunder',1610612760],
  TOR:['Toronto','Raptors',1610612761],UTA:['Utah','Jazz',1610612762],MEM:['Memphis','Grizzlies',1610612763],
  WAS:['Washington','Wizards',1610612764],DET:['Detroit','Pistons',1610612765],CHA:['Charlotte','Hornets',1610612766]
};
function teamFullName(abbr) {
  const row = PG_TEAMS[String(abbr || '').toUpperCase()];
  return row ? row[0] + ' ' + row[1] : String(abbr || 'Unknown team');
}
function pgTeamBadge(abbr) {
  const row = PG_TEAMS[String(abbr || '').toUpperCase()];
  const div = el('div','pg-team');
  const logo = el('div','pg-logo');
  const fallback = el('span','pg-logo-fallback',String(abbr || '—'));
  logo.append(fallback);
  if (row) {
    const img = el('img','pg-logo-image');
    img.width = 48; img.height = 48; img.alt = teamFullName(abbr) + ' logo';
    img.loading = 'lazy'; img.decoding = 'async';
    img.onerror = () => { img.remove(); fallback.hidden = false; };
    img.onload = () => { fallback.hidden = true; };
    img.src = `https://cdn.nba.com/logos/nba/${row[2]}/global/L/logo.svg`;
    logo.append(img);
  }
  const text = el('div','pg-team-label');
  text.append(el('small','muted',row ? row[0] : ''),el('strong','',row ? row[1] : String(abbr || 'Unknown')));
  div.append(logo,text);
  return div;
}
function teamMatchup(away,home) {
  const div = el('div','pg-matchup');
  div.append(pgTeamBadge(away),el('span','pg-at','@'),pgTeamBadge(home));
  return div;
}
function pgNumber(value) {
  if (value === null || value === undefined || value === '' || typeof value === 'boolean') return null;
  const n = Number(value);
  return Number.isFinite(n) ? n : null;
}
function pgETDay(timestamp) {
  const d = new Date(timestamp);
  if (!Number.isFinite(d.getTime())) return null;
  const parts = new Intl.DateTimeFormat('en-US',{timeZone:'America/New_York',year:'numeric',month:'2-digit',day:'2-digit'}).formatToParts(d);
  const v = Object.fromEntries(parts.map(p=>[p.type,p.value]));
  return `${v.year}-${v.month}-${v.day}`;
}
function pgMoney(n) { return '$' + Number(n || 0).toFixed(2); }
function pgPercentage(n) { return (Number(n || 0) * 100).toFixed(1) + '%'; }
function pgAllocateMoneylines(input) {
  const games = Array.isArray(input.games) ? input.games : [];
  const bankroll = pgNumber(input.bankroll);
  const dailyPct = pgNumber(input.dailyPct);
  const perGamePct = pgNumber(input.perGamePct);
  const minEv = pgNumber(input.minEv);
  const committed = pgNumber(input.committed === undefined ? 0 : input.committed);
  const now = new Date(input.now || Date.now());
  const today = pgETDay(now);
  const snapshotTime = input.updated_at_utc ? new Date(input.updated_at_utc) : null;
  const snapshotAge = snapshotTime && Number.isFinite(snapshotTime.getTime()) ? (now - snapshotTime)/1000 : Infinity;
  const valid = bankroll !== null && bankroll > 0 && bankroll <= 10000000 &&
    dailyPct !== null && dailyPct >= 0 && dailyPct <= 0.20 &&
    perGamePct !== null && perGamePct >= 0 && perGamePct <= 0.05 &&
    minEv !== null && minEv >= 0 && minEv <= 1 &&
    committed !== null && committed >= 0 && committed <= bankroll &&
    Number.isFinite(now.getTime());
  const stale = input.stale === true || snapshotAge > 15 * 60 || snapshotAge < -120;
  const rows = [], allToday = games.filter(g=>g.game_date === today);
  for(const g of allToday) {
    const forecast = g.winner_forecast || null;
    const team = forecast && forecast.predicted_winner;
    const p = forecast ? pgNumber(forecast.predicted_winner_probability) : null;
    const quote = team && (g.moneylines || {})[team];
    const ask = quote ? pgNumber(quote.ask) : null;
    const shares = quote ? pgNumber(quote.shares) : null;
    const tipoff = new Date(g.tipoff_utc);
    const ev = p !== null && ask !== null && ask > 0 && ask < 1 ? p/ask - 1 : null;
    const row = {slug:g.slug,game_date:g.game_date,away:g.away,home:g.home,tipoff_utc:g.tipoff_utc,
      winner:team || null,p,ask,shares,ev,stake:0,rawStake:0,reason:'',eligible:false};
    if(!valid) row.reason='Enter a valid starting bankroll and risk caps';
    else if(stale) row.reason='Live quotes are stale — refresh data';
    else if(!Number.isFinite(tipoff.getTime()) || tipoff <= now) row.reason='Game has started';
    else if(!team || p === null || p <= 0 || p >= 1) row.reason='No valid game winner forecast';
    else if(ask === null || ask <= 0 || ask >= 1) row.reason='No valid moneyline best ask';
    else if(shares === null || shares <= 0) row.reason='No available best-ask shares';
    else if(ev === null || ev < minEv - 1e-12) row.reason=`Estimated EV below ${pgPercentage(minEv)} threshold`;
    else {
      // YES shares cost 'ask' each and settle at $1 if successful.
      // Full Kelly fraction = (p - ask) / (1 - ask). Quarter Kelly + exposure caps.
      const kelly = Math.max(0,(p-ask)/(1-ask));
      const liquidityDollars = ask*shares;
      row.rawStake = Math.max(0,Math.min(bankroll * 0.25*kelly,bankroll*perGamePct,liquidityDollars));
      row.eligible = row.rawStake >= 0.01;
      row.reason = row.eligible ? 'Model-qualified (research only)' : 'Quote depth or capped stake below $0.01';
    }
    rows.push(row);
  }
  const dailyCap = valid ? bankroll*dailyPct : 0;
  const availableCap = Math.max(0,dailyCap - (valid ? committed : 0));
  const totalRaw = rows.reduce((sum,r)=>sum+r.rawStake,0);
  const scale = totalRaw > availableCap && totalRaw > 0 ? availableCap / totalRaw : 1;
  for(const row of rows) {
    // Floor to whole cents, never exceed an exposure or observed quote-depth cap.
    row.stake = Math.floor((row.rawStake*scale + 1e-9)*100)/100;
    if(row.eligible && row.stake < .01) row.reason='Daily cap exhausted / stake below $0.01';
  }
  const spent = rows.reduce((sum,r)=>sum + Math.round(r.stake*100),0)/100;
  return {today,valid,stale,bankroll:bankroll || 0,dailyCap,committed:valid?committed:0,spent,unallocated:Math.max(0,dailyCap-(valid?committed:0)-spent),
    totalGames:allToday.length,active:rows.filter(r=>r.stake>0).length,rows};
}
function pgBudgetOptions() {
  const fields = id => document.getElementById(id);
  return {
    bankroll:fields('pg-bankroll').value,
    committed:fields('pg-committed').value,
    dailyPct:fields('pg-daily-pct').value,
    perGamePct:fields('pg-game-pct').value,
    minEv:fields('pg-min-ev').value
  };
}
let pgLastLoadedDay = null;
function pgCommittedKey(today) {return 'parlay-guard-moneyline-committed-v1-'+today;}
function pgRestoreBudget() {
  try {
    const saved = window.localStorage.getItem('parlay-guard-moneyline-bankroll-v1');
    if(saved && Number(saved)>0 && Number(saved)<=10000000) $('pg-bankroll').value=saved;
    const day=pgETDay(Date.now());
    const used=window.localStorage.getItem(pgCommittedKey(day));
    $('pg-committed').value=used && Number(used)>=0 ? used : '0';
    pgLastLoadedDay=day;
  } catch (_) {/* Private mode may disable storage */}
}
function pgBindBudget() {
  pgRestoreBudget();
  for(const id of ['pg-bankroll','pg-committed','pg-daily-pct','pg-game-pct','pg-min-ev']) {
    $(id).addEventListener('input',() => {
      if(id === 'pg-bankroll') {
        try { window.localStorage.setItem('parlay-guard-moneyline-bankroll-v1',$(id).value); } catch (_) {}
      }
      if(id === 'pg-committed') {
        try { window.localStorage.setItem(pgCommittedKey(pgETDay(Date.now())),$(id).value); } catch (_) {}
      }
      renderMoneylinePlanner();
    });
  }
}
function renderMoneylinePlanner() {
  const activeDay=pgETDay(Date.now());
  if(pgLastLoadedDay && pgLastLoadedDay !== activeDay) {
    try { $('pg-committed').value=window.localStorage.getItem(pgCommittedKey(activeDay)) || '0'; } catch (_) { $('pg-committed').value='0'; }
    pgLastLoadedDay=activeDay;
  }
  const root=$('pg-moneyline-results'), opts=pgBudgetOptions();
  const result=pgAllocateMoneylines({...opts,games:lastData.games||[],updated_at_utc:lastData.updated_at_utc,
    stale:lastData.refresh_status?.stale === true,now:Date.now()});
  $('pg-cap').textContent=pgMoney(result.dailyCap);
  $('pg-total').textContent=pgMoney(result.spent);
  $('pg-committed-total').textContent=pgMoney(result.committed);
  $('pg-left').textContent=pgMoney(result.unallocated);
  $('pg-count').textContent=String(result.active)+' / '+result.totalGames;
  $('pg-day').textContent=result.today || 'Today';
  root.replaceChildren();
  if(!result.valid) root.append(el('p','gold','Enter a valid bankroll and risk limits.'));
  else if(result.stale) root.append(el('p','gold','Snapshot is stale or missing. Allocations are $0 until a new quote is available.'));
  else if(!result.totalGames) root.append(el('p','muted','No remaining scheduled games for today in the current snapshot.'));
  else {
    for(const r of result.rows) {
      const row=el('div','pg-plan-row');
      const left=el('div','pg-plan-team');
      left.append(el('strong','',teamFullName(r.winner || r.home)),el('small','muted',teamFullName(r.away)+' @ '+teamFullName(r.home)));
      const right=el('div','pg-plan-numbers');
      right.append(el('strong',r.stake>0?'green':'muted',pgMoney(r.stake)),
        el('small','muted',r.ev==null?'No quote / probability':`Model EV ${pgPercentage(r.ev)} · ask ${r.ask.toFixed(3)}`));
      row.append(left,right);root.append(row);
    }
  }
  const bySlug=new Map(result.rows.map(r=>[r.slug,r]));
  document.querySelectorAll('[data-pg-moneyline-slug]').forEach(node=>{
    const r=bySlug.get(node.dataset.pgMoneylineSlug);
    node.replaceChildren();
    if(!r){node.append(el('strong','muted','Today’s moneyline allocation: Not scheduled for today'));return;}
    const header=el('div','pg-reco-head');
    header.append(el('span','muted small','Moneyline paper stake · '+teamFullName(r.winner || r.home)),
      el('strong',r.stake>0?'green':'muted',pgMoney(r.stake)));
    node.append(header);
    const detail= r.stake>0 ? `Elo ${pgPercentage(r.p)} · ask ${r.ask.toFixed(3)} · model EV ${pgPercentage(r.ev)} · max loss ${pgMoney(r.stake)}` : r.reason;
    node.append(el('small',r.stake>0?'muted':'gold',detail));
  });
}
if (typeof module !== 'undefined' && module.exports) {
  module.exports={pgAllocateMoneylines,pgETDay,teamFullName,PG_TEAMS};
}
