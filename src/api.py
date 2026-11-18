from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / 'models' / 'parlay_guard_v1_vs_v2.html'

app = FastAPI(title='Parlay Guard', version='0.1.0')

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
