"""Restore the forecast inputs supplied through the deployment environment."""

import base64
import gzip
import os
from pathlib import Path
from tempfile import NamedTemporaryFile


def restore_forecast_inputs(root: Path) -> None:
    inputs = {
        "PG_NBA_HISTORY": root / "data" / "nba_games_updated.csv",
        "PG_NBA_ELO_MODEL": root / "models" / "elo_final_holdout_model.joblib",
    }
    for prefix, path in inputs.items():
        count_key = f"{prefix}_PARTS"
        if count_key not in os.environ:
            continue
        try:
            count = int(os.environ[count_key])
            if not 1 <= count <= 64:
                raise ValueError("Invalid part count")
            keys = [f"{prefix}_{index}" for index in range(count)]
            encoded = "".join(os.environ[key] for key in keys)
            contents = gzip.decompress(base64.b64decode(encoded, validate=True))
        except (KeyError, ValueError, EOFError, OSError) as exc:
            raise RuntimeError(f"Invalid deployment input for {path.name}") from exc

        path.parent.mkdir(parents=True, exist_ok=True)
        with NamedTemporaryFile(dir=path.parent, delete=False) as staged:
            temporary = Path(staged.name)
            try:
                staged.write(contents)
                staged.close()
                temporary.replace(path)
            finally:
                temporary.unlink(missing_ok=True)
        for key in [count_key, *keys]:
            del os.environ[key]
