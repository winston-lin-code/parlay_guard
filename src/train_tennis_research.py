"""Rebuild and hold out the ITF men's and women's singles models."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
from tempfile import NamedTemporaryFile, mkdtemp

import requests

from .env_config import env_value
from .train_tennis_elo import read_matches, train as fit_elo
from .train_tennis_halftime import fit as fit_elo_breaks
from .train_tennis_live import fit_tour


ARCHIVE = ("https://raw.githubusercontent.com/micahp/tennis-point-by-point-archive/"
           "83733587353df8a41f2fd4f516147d5aa83f5a8d")
TOURS = {"ITFME": ("atp", "atp_matches_futures"), "ITFWO": ("wta", "wta_matches_qual_itf")}
REQUIRED_COLUMNS = {"tourney_id", "tourney_date", "match_num", "surface", "winner_id", "loser_id",
                    "winner_name", "loser_name", "round", "best_of", "score"}


def validate_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not REQUIRED_COLUMNS.issubset(reader.fieldnames or ()) or next(reader, None) is None:
            raise ValueError(f"{path.name} is missing match rows or required columns")


def obtain(data, subdirectory, filename, download):
    path = data / filename
    if not path.is_file() and (nested := data / subdirectory / filename).is_file():
        path = nested
    if path.is_file():
        validate_csv(path)
        return path
    if not download:
        raise FileNotFoundError(f"Missing {filename} in {data}")
    url = f"{ARCHIVE}/{subdirectory}/{filename}"
    pending = None
    try:
        with requests.get(url, timeout=(10, 45), stream=True) as response:
            if not response.ok:
                raise OSError(f"HTTP {response.status_code} for {filename}")
            with NamedTemporaryFile(dir=data, delete=False) as stream:
                pending = Path(stream.name)
                for chunk in response.iter_content(262144):
                    stream.write(chunk)
        validate_csv(pending)
        pending.replace(path)
        return path
    finally:
        if pending is not None:
            pending.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path(env_value("TENNIS_HISTORY_DIR") or "data/tennis-history"))
    parser.add_argument("--download", action="store_true", help="Fetch missing 2022–26 files from the pinned archive")
    parser.add_argument("--output", type=Path, default=Path("results/tennis-research"))
    args = parser.parse_args()
    if args.download:
        args.data.mkdir(parents=True, exist_ok=True)
    try:
        jobs = [(tour, folder, f"{stem}_{year}.csv") for tour, (folder, stem) in TOURS.items()
                for year in range(2022, 2027)]
        with ThreadPoolExecutor(max_workers=4) as pool:
            paths = list(pool.map(lambda job: obtain(args.data, job[1], job[2], args.download), jobs))

        score, elo, combined, rows = {}, {}, {}, []
        for tour in TOURS:
            matches = read_matches([path for (code, _, _), path in zip(jobs, paths) if code == tour])
            score[tour] = fit_tour([(match.date.strftime("%Y%m%d"), match.margins) for match in matches], 2024)
            elo[tour] = fit_elo(matches, 2025)
            combined[tour] = fit_elo_breaks(matches, elo[tour], score[tour])
            for stage, result in combined[tour].items():
                rows.append({"tour": tour, "sets_completed": stage, "holdout_year": 2025, **result["holdout"]})
            print(f"{tour} · {len(matches):,} completed main-draw singles matches", flush=True)

        args.output.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        output = Path(mkdtemp(prefix=f"itf-{stamp}-", dir=args.output))
        artifacts = {
            "itf_set_breaks.json": {"version": 1, "tours": score},
            "itf_elo.json": {"version": 1, "ratings_released_after_days": 14, "tours": elo},
            "itf_elo_set_breaks.json": {"version": 1, "holdout_year": 2025, "tours": combined},
        }
        for name, artifact in artifacts.items():
            (output / name).write_text(json.dumps(artifact, allow_nan=False) + "\n", encoding="utf-8")
        with (output / "holdout.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"2025 held-out metrics · {output.resolve() / 'holdout.csv'}")
    except (OSError, ValueError, KeyError, requests.RequestException) as exc:
        parser.exit(2, f"Tennis training stopped: {exc}\n")


if __name__ == "__main__":
    main()
