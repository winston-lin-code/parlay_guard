# main.py
from fastapi import FastAPI
from fastapi.responses import FileResponse

app = FastAPI()

@app.get("/")
def dashboard():
    return FileResponse("static/index.html")

@app.get("/api/games")
def get_games():
    # Example data; replace with your model outputs
    return [
        {
            "id": 1,
            "matchup": "Warriors vs Mavericks",
            "elo_probability": 0.42,
            "expected_ev": 0.082,
            "status": "SKIP",
            "reason": "High-impact player unavailable"
        },
        {
            "id": 2,
            "matchup": "Celtics vs Knicks",
            "elo_probability": 0.67,
            "expected_ev": 0.065,
            "status": "ELIGIBLE",
            "reason": "Roster stable"
        }
    ]