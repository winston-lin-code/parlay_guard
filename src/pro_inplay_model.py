"""Score-state win probabilities for MLB and NHL full-game moneylines."""

from functools import lru_cache
import json
import math
from pathlib import Path

MODEL_DIR = Path(__file__).resolve().parents[1] / "models" / "pro_scores"
VERSION = "pro-live-moneyline-v1"


def baseball_state(score, inning, phase, outs=0, bases=(False, False, False)):
    away, home = score
    if not 1 <= inning <= 9:
        return None
    if phase == "middle" and (inning < 9 or home <= away):
        return home - away, (19 - 2 * inning) / 18, 1, 0
    if phase == "end" and inning < 9:
        return home - away, (18 - 2 * inning) / 18, 0, 0
    if (phase not in ("top", "bottom") or type(outs) is not int or not 0 <= outs <= 2
            or not bases or len(bases) != 3 or not all(type(on_base) is bool for on_base in bases)):
        return None
    home_batting = phase == "bottom"
    completed = 2 * (inning - 1) + home_batting + outs / 3
    home_extra = 1 - outs / 3 if home_batting else outs / 3
    runners = sum(weight for on_base, weight in zip(bases, (1, 1.4, 1.6)) if on_base)
    return home - away, (18 - completed) / 18, home_extra, runners if home_batting else -runners


def model_state(league, state):
    score = state.get("score_away_home")
    period = state.get("period")
    phase = state.get("phase")
    if not score or type(period) is not int:
        return None
    away, home = score
    if league == "mlb":
        if state.get("scheduled_innings") != 9:
            return None
        return baseball_state(score, period, phase, state.get("outs"), state.get("bases"))
    remaining = state.get("seconds_remaining")
    if (phase not in ("in_play", "intermission") or state.get("period_type") != "REG"
            or not 1 <= period <= 3 or type(remaining) is not int or not 0 <= remaining <= 1200):
        return None
    seconds = (3 - period) * 1200 + remaining
    if not 60 <= seconds <= 3600:
        return None
    return home - away, seconds / 3600, 0


def features(state):
    margin, remaining, home_extra_half, *runners = state
    scale = math.sqrt(remaining)
    return [margin / scale, remaining, home_extra_half / scale, *(v / scale for v in runners)]


@lru_cache(maxsize=2)
def load(league):
    try:
        model = json.loads((MODEL_DIR / f"{league}_live.json").read_text(encoding="utf-8"))
        expected = 4 if league == "mlb" else 3
        if model["version"] == VERSION and model["league"] == league and len(model["weights"]) == expected:
            return model
    except (OSError, ValueError, KeyError):
        pass
    return None


def forecast(state, artifact):
    logit = artifact["intercept"] + sum(w * v for w, v in zip(artifact["weights"], features(state)))
    adjusted = artifact["calibration_slope"] * logit + artifact["calibration_intercept"]
    return 1 / (1 + math.exp(-max(-30, min(30, adjusted))))
