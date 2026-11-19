from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, ConfigDict
from datetime import date as Date
from typing import List
from .live_polymarket import get_live_quotes
from .player_prop_recommendations import cached_analyze_game
from fastapi.responses import FileResponse, HTMLResponse

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / 'models' / 'parlay_guard_v1_vs_v2.html'

app = FastAPI(title='Parlay Guard', version='0.1.0')
from .live_dashboard_api import router as live_dashboard_router
app.include_router(live_dashboard_router)
@app.get('/api/health')
def health():
    return {'status': 'ok', 'dashboard_available': REPORT.is_file()}

@app.get('/api/games')
def games():
    raise HTTPException(status_code=501, detail='Live schedule integration is not configured yet.')

@app.get('/api/markets')
def markets():
    raise HTTPException(status_code=501, detail='Live Polymarket integration is not configured yet.')

@app.get('/api/predictions')
def predictions():
    raise HTTPException(status_code=501, detail='Live prediction integration is not configured yet.')

@app.get('/', response_class=HTMLResponse)
def dashboard():
    if REPORT.is_file():
        return FileResponse(REPORT, media_type='text/html')
    return HTMLResponse('<h1>Parlay Guard is running</h1><p>Generate and include models/parlay_guard_v1_vs_v2.html to display the dashboard.</p><p>Check <a href="/docs">/docs</a> or <a href="/api/health">/api/health</a>.</p>')

class GameRequest(BaseModel):
    away: str = Field(pattern=r"^[A-Z]{3}$", examples=["MIL"])
    home: str = Field(pattern=r"^[A-Z]{3}$", examples=["MIA"])
    date: Date = Field(examples=["2026-11-18"])

class LiveRequest(BaseModel):
    games: List[GameRequest] = Field(min_length=1, max_length=30)
    model_config = ConfigDict(json_schema_extra={'examples': [
        {'games': [{'away': 'MIL', 'home': 'MIA', 'date': '2026-11-18'}]}
    ]})

@app.post('/api/markets/live')
def live_markets(request: LiveRequest):
    try:
        return get_live_quotes([g.model_dump(mode='json') for g in request.games])
    except Exception as exc:
        raise HTTPException(status_code=503, detail='Polymarket live quotes unavailable') from exc


class RecommendationRequest(BaseModel):
    away: str = Field(pattern=r"^[A-Z]{3}$", examples=["MIL"])
    home: str = Field(pattern=r"^[A-Z]{3}$", examples=["MIA"])
    date: Date = Field(examples=["2026-11-18"])
    home_v1: float = Field(ge=0, le=1, examples=[0.35])
    home_v2: float = Field(ge=0, le=1, examples=[0.39])

@app.post('/api/recommendations')
def recommendations(request: RecommendationRequest):
    """Per-game moneyline & points-prop research; no actual Combo quote is inferred."""
    game=request.model_dump(mode='json')
    if game['away']==game['home']:
        raise HTTPException(422,'Away and home teams must differ')
    try:
        return cached_analyze_game(game)
    except ValueError as exc:
        raise HTTPException(422,detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(503,detail='Recommendation data unavailable; check application logs') from exc
