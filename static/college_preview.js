/* Embedded college EV preview on the existing NBA live dashboard. */
(()=>{
  const root=document.getElementById('college-overview-content');if(!root)return;
  async function refresh(){
    try{
      const response=await fetch('/api/college/live?date_offset=0',{cache:'no-store'});
      if(!response.ok)throw new Error('HTTP '+response.status);
      const data=await response.json();const games=data.games||[];
      const edges=games.filter(g=>g.best_pick).sort((a,b)=>b.best_pick.estimated_ev-a.best_pick.estimated_ev);
      root.replaceChildren();
      const p=document.createElement('p');p.textContent=games.length+' upcoming college games · '+edges.length+' with model-estimated EV ≥5% · '+games.filter(g=>g.status==='ROSTER_INJURY_REVIEW').length+' roster/injury reviews';root.append(p);
      if(!edges.length){const msg=document.createElement('div');msg.textContent='No quoted qualifying college picks yet, or models still need training.';root.append(msg);return}
      for(const g of edges.slice(0,5)){
        const row=document.createElement('p');
        row.textContent=(g.sport==='cfb'?'Football':'Basketball')+' · '+g.away.display_name+' @ '+g.home.display_name+' · '+g.best_pick.team_name+' · '+(g.best_pick.estimated_ev*100).toFixed(1)+'% est. EV'+(g.status==='ROSTER_INJURY_REVIEW'?' · INJURY/ROSTER REVIEW':'');
        root.append(row);
      }
    }catch(e){root.textContent='College data temporarily unavailable: '+e.message}
  }
  refresh();setInterval(refresh,60000);
})();
