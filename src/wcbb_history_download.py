"""Download the public season inputs for the offline WCBB experiment."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tempfile import NamedTemporaryFile

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util import Retry

from .wcbb_feature_experiments import BOX, PLAYS, SCHEDULE, SEASONS


RELEASES = "https://github.com/sportsdataverse/sportsdataverse-data/releases/download"
DATASETS = (
    ("espn_womens_college_basketball_schedules", "wbb_schedule", SCHEDULE),
    ("espn_womens_college_basketball_team_boxscores", "team_box", BOX),
    ("espn_womens_college_basketball_pbp", "play_by_play", PLAYS),
)
GROUPS = ("league", "season", "level", "group_id", "name", "short_name", "abbreviation")
TEAMS = ("league", "season", "team_id", "team_id_source", "subdivision_id", "division_id")
DIVISION_I = {"divisioni", "division1", "ncaadivisioni", "ncaadivision1", "di", "d1"}


def validate_parquet(path, columns):
    from pyarrow.parquet import ParquetFile

    with ParquetFile(path) as data:
        missing = set(columns) - set(data.schema.names)
        if data.metadata.num_rows == 0 or missing:
            raise ValueError(f"{path.name} is empty or missing fields: {', '.join(sorted(missing))}")


def fetch(root, release, stem, columns, *, allow_csv=False):
    path = root / f"{stem}.parquet"
    if path.is_file():
        validate_parquet(path, columns)
        return path
    if allow_csv and (csv_path := path.with_suffix(".csv")).is_file():
        if set(columns) - set(pd.read_csv(csv_path, nrows=0)):
            raise ValueError(f"{csv_path.name} is missing fields required by the experiment")
        return csv_path
    url = f"{RELEASES}/{release}/{path.name}"
    retry = Retry(total=2, backoff_factor=1, status_forcelist=(429, 500, 502, 503, 504),
                  allowed_methods={"GET"}, raise_on_status=False)
    pending = None
    try:
        with requests.Session() as session:
            session.mount("https://", HTTPAdapter(max_retries=retry))
            with session.get(url, stream=True, timeout=(10, 60)) as response:
                if not response.ok:
                    raise OSError(f"HTTP {response.status_code} for {url}")
                with NamedTemporaryFile(mode="wb", dir=root, prefix=f".{stem}-", suffix=".part", delete=False) as file:
                    pending = Path(file.name)
                    for chunk in response.iter_content(1024 * 1024):
                        file.write(chunk)
        validate_parquet(pending, columns)
        if not path.exists():
            pending.replace(path)
        return path
    finally:
        if pending is not None:
            pending.unlink(missing_ok=True)


def membership(groups, teams):
    groups = groups[groups.league.eq("wbb") & groups.season.isin(SEASONS) & groups.level.isin(("subdivision", "division"))]
    labels = groups[["name", "short_name", "abbreviation"]].fillna("").apply(
        lambda col: col.str.lower().str.replace(r"[^a-z0-9]", "", regex=True))
    division = groups[labels.isin(DIVISION_I).any(axis=1)][["season", "group_id"]].drop_duplicates()
    counts = division.groupby("season").group_id.nunique()
    if set(counts.index) != set(SEASONS) or not counts.eq(1).all():
        raise ValueError("SportsDataverse did not identify one unambiguous Division I group for each experiment season")

    espn = teams[teams.league.eq("wbb") & teams.team_id_source.eq("espn") & teams.season.isin(SEASONS)].drop_duplicates()
    if espn.duplicated(["season", "team_id"]).any():
        raise ValueError("SportsDataverse returned conflicting memberships for an ESPN team and season")
    joined = espn.merge(division, on="season", validate="many_to_one")
    members = joined[joined.subdivision_id.eq(joined.group_id) | joined.division_id.eq(joined.group_id)][["season", "team_id"]].copy()
    members["team_id"] = members.team_id.astype("string")
    if not members.team_id.str.fullmatch(r"[1-9][0-9]*", na=False).all() or set(members.season) != set(SEASONS):
        raise ValueError("Division I team history is missing season coverage or contains invalid ESPN team IDs")
    return members.drop_duplicates().sort_values(["season", "team_id"])


def prepare(root, workers):
    root.mkdir(parents=True, exist_ok=True)
    membership_path = root / "division_i_teams.csv"
    jobs = [(release, f"{stem}_{year}", fields, True)
            for release, stem, fields in DATASETS for year in SEASONS]
    if not membership_path.is_file():
        jobs += [("wbb_groups", "wbb_group_seasons", GROUPS, False)]
        jobs += [("wbb_groups", f"wbb_team_group_seasons_{year}", TEAMS, False) for year in SEASONS]

    def obtain(job):
        release, stem, fields, csv = job
        try:
            path = fetch(root, release, stem, fields, allow_csv=csv)
        except (OSError, ValueError, requests.RequestException) as exc:
            return f"{stem}: {exc}"
        print(f"Ready · {path.name}", flush=True)
        return None

    with ThreadPoolExecutor(max_workers=workers) as executor:
        failures = [error for error in executor.map(obtain, jobs) if error]
    if failures:
        print("Missing or invalid files. Rerun to retry these inputs.", flush=True)
        for error in failures:
            print(error, flush=True)
        return 2

    if not membership_path.is_file():
        groups = pd.read_parquet(root / "wbb_group_seasons.parquet", columns=list(GROUPS))
        teams = pd.concat([pd.read_parquet(root / f"wbb_team_group_seasons_{year}.parquet", columns=list(TEAMS))
                           for year in SEASONS], ignore_index=True)
        members = membership(groups, teams)
        with NamedTemporaryFile(dir=root, delete=False) as file:
            pending = Path(file.name)
        try:
            members.to_csv(pending, index=False)
            if not membership_path.exists():
                pending.replace(membership_path)
        finally:
            pending.unlink(missing_ok=True)
        print(f"Ready · {membership_path.name} · {len(members):,} Division I team-seasons", flush=True)
    else:
        existing = pd.read_csv(membership_path, dtype={"team_id": "string"})
        if (not set(SEASONS) <= set(existing.season)
                or not existing.team_id.str.fullmatch(r"[1-9][0-9]*", na=False).all()):
            raise ValueError(f"{membership_path} needs valid ESPN IDs for all eight seasons")
        print(f"Using existing · {membership_path.name}", flush=True)
    print(f"WCBB history ready · {root.resolve()}", flush=True)
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/wcbb-history"), help="Destination for season files")
    parser.add_argument("--workers", type=int, choices=range(1, 9), default=3)
    args = parser.parse_args(argv)
    try:
        import pyarrow.parquet  # noqa: F401

        return prepare(args.data, args.workers)
    except ImportError:
        parser.exit(2, "Install the downloader dependencies with python -m pip install requests pyarrow\n")
    except (KeyError, ValueError, OSError) as exc:
        parser.exit(2, f"WCBB history setup stopped: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
