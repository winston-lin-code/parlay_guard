"""Uvicorn entrypoint for Parlay Guard's auto-updating research dashboard.

From the project root: python -m uvicorn src.api:app --reload --host 127.0.0.1 --port 8000
"""
from __future__ import annotations
import os
import threading
import time
from contextlib import asynccontextmanager
from datetime import datetime,timezone
from pathlib import Path
from fastapi import FastAPI,HTTPException
from fastapi.responses import FileResponse,HTMLResponse,JSONResponse

ROOT=Path(__file__).resolve().parents[1]
LIVE=ROOT/'static/live.html'
ARCHIVE=ROOT/'models/parlay_guard_v1_vs_v2.html'
ARCHIVE_STRATEGY=ROOT/'models/parlay_comparison.html'

app = FastAPI(title='Parlay Guard', version='0.1.0')
from .live_dashboard_api import router as live_dashboard_router
app.include_router(live_dashboard_router)

class LiveState:
    def __init__(self):
        self.lock=threading.Lock();self.stop=threading.Event();self.wakeup=threading.Event()
        self.snapshot={'games':[],'updated_at_utc':None,'warnings':['Initial data refresh has not completed.']}
        self.running=False;self.last_error=None;self.last_started=None;self.last_finished=None
        self.player_last=0.0;self.team_last=0.0;self.thread=None

    def start(self):
        if self.thread is not None:return
        self.thread=threading.Thread(target=self._loop,daemon=True,name='parlay-guard-refresh')
        self.thread.start()

    def shutdown(self):
        self.stop.set();self.wakeup.set()
        if self.thread:self.thread.join(timeout=3)

    def trigger(self):
        self.wakeup.set()

    def _loop(self):
        from .auto_live import run_update
        while not self.stop.is_set():
            with self.lock:
                self.running=True;self.last_started=datetime.now(timezone.utc).isoformat()
            started=time.monotonic()
            refresh_player=not self.player_last or started-self.player_last>2*3600
            refresh_team=not self.team_last or started-self.team_last>45*60
            try:
                output=run_update(max_games=int(os.environ.get('PG_MAX_GAMES','10')),
                                  fetch_players=refresh_player,refresh_team=refresh_team)
                # Only publish fully assembled snapshots. A failed upstream
                # refresh never replaces the last usable results with partial CSVs.
                with self.lock:
                    self.snapshot=output;self.last_error=None
                    if refresh_player:self.player_last=time.monotonic()
                    if refresh_team:self.team_last=time.monotonic()
            except Exception as exc:
                with self.lock:self.last_error=f'{type(exc).__name__}: {exc}'[:700]
            finally:
                with self.lock:
                    self.running=False;self.last_finished=datetime.now(timezone.utc).isoformat()
            self.wakeup.wait(timeout=600) # 10-minute data updates while server is running
            self.wakeup.clear()

    def read(self):
        with self.lock:
            snap=dict(self.snapshot)
            status={'refreshing':self.running,'last_error':self.last_error,
                    'refresh_started_utc':self.last_started,'refresh_finished_utc':self.last_finished}
        # Hide games once tipoff passes, without waiting for next refresh.
        from .auto_live import utc
        now=datetime.now(timezone.utc)
        snap['games']=[g for g in snap.get('games',[])
                       if utc(g.get('tipoff_utc')) is not None and utc(g.get('tipoff_utc'))>now]
        t=utc(snap.get('updated_at_utc'))
        status['stale']=t is None or (now-t).total_seconds()>1200
        if status['stale']:
            # Market prices and EV recommendations aren't live after a failed
            # refresh. Keep old model forecasts visible with an explicit banner.
            snap['games']=[{**g,'selected_props':[],'moneylines':{}} for g in snap['games']]
            snap['warnings']=list(snap.get('warnings') or [])+['Market quotes are stale; prop selections and moneyline asks are hidden until refresh succeeds.']
        snap['refresh_status']=status
        return snap


state=LiveState()

@asynccontextmanager
async def lifespan(app:FastAPI):
    if os.getenv('PG_DISABLE_AUTO_REFRESH')!='1':state.start()
    yield
    state.shutdown()

app=FastAPI(title='Parlay Guard Auto Live',version='0.4',lifespan=lifespan)

@app.get('/',response_class=HTMLResponse)
def dashboard():
    if not LIVE.exists():return HTMLResponse('<h1>Missing static/live.html</h1>',status_code=503)
    return FileResponse(LIVE,media_type='text/html',headers={'Cache-Control':'no-store'})

@app.get('/strategy',response_class=HTMLResponse)
def strategy():return dashboard()

@app.get('/archive',response_class=HTMLResponse)
def old_dashboard():
    if not ARCHIVE.exists():raise HTTPException(404,'Archived generated report not found')
    return FileResponse(ARCHIVE,media_type='text/html')

@app.get('/archive-strategy',response_class=HTMLResponse)
def old_strategy():
    if not ARCHIVE_STRATEGY.exists():raise HTTPException(404,'Archived strategy report not found')
    return FileResponse(ARCHIVE_STRATEGY,media_type='text/html')

@app.get('/api/live/data')
def live_data():
    return JSONResponse(state.read(),headers={'Cache-Control':'no-store,max-age=0'})

@app.post('/api/live/refresh')
def live_refresh():
    state.trigger()
    return {'queued':True,'message':'Background refresh requested. This endpoint places no trades.'}

@app.get('/api/health')
def health():
    s=state.read()
    return {'status':'ok','refresh':s['refresh_status'],
            'live_dashboard_available':LIVE.exists(),
            'upcoming_game_count':len(s['games']),'url':'/'}

@app.get('/api/games')
def games():return {'games':state.read()['games']}

@app.get('/api/markets')
def markets():
    return {'games':[{'slug':g['slug'],'moneylines':g['moneylines'],
                      'spread':g['market_spread'],'total':g['market_total']} for g in state.read()['games']]}

@app.get('/api/predictions')
def predictions():
    return {'games':[{'slug':g['slug'],'winner':g['winner_forecast'],
                      'lines':g['game_line_prediction'],'props':g['selected_props']} for g in state.read()['games']]}
