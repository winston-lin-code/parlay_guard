from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data'
MODELS = ROOT / 'models'
DATA.mkdir(exist_ok=True)
MODELS.mkdir(exist_ok=True)
