"""Recover reproducible Parlay Guard NBA and Polymarket historical datasets.

Run from the Parlay Guard repository root (place this file in the root):
    python download_parlay_guard_data.py --only nba --seasons 2024-25 2025-26 --max-games 3
    python download_parlay_guard_data.py --only polymarket
    python download_parlay_guard_data.py --only nba --seasons 2024-25 2025-26

Historical NBA data: NBA stats LeagueGameLog + NBA CDN play-by-play (stats V3 fallback).
Polymarket source: epinnock/polymarket-nba on Hugging Face.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
CDN_URL = "https://cdn.nba.com/static/json/liveData/playbyplay/playbyplay_{game_id}.json"
CLOCK_RE = re.compile(r"^PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+(?:\.\d+)?)S)?$")


def retry(fn, label: str, attempts: int = 3):
    for i in range(1, attempts + 1):
        try:
            return fn()
        except Exception as exc:
            print(f"  {label}: attempt {i}/{attempts} failed: {type(exc).__name__}: {exc}", flush=True)
            if i == attempts:
                raise
            time.sleep(min(20, 2 ** i * 2))


def get_polymarket() -> None:
    from huggingface_hub import hf_hub_download
    import pyarrow.parquet as pq

    repo = "epinnock/polymarket-nba"
    files = ["data/markets.parquet", "data/prices/prices-2025-26.parquet"]
    destination = DATA / "polymarket"
    destination.mkdir(parents=True, exist_ok=True)

    for name in files:
        print(f"Polymarket: fetching {name} (resumes via Hugging Face cache)", flush=True)
        actual = retry(
            lambda name=name: hf_hub_download(
                repo_id=repo,
                repo_type="dataset",
                filename=name,
                local_dir=destination,
            ),
            name,
        )
        path = Path(actual)
        rows = pq.ParquetFile(path).metadata.num_rows
        print(f"  OK: {path} | {path.stat().st_size:,} bytes | {rows:,} rows", flush=True)

    print("Polymarket archive retrieved. IMPORTANT: sampled market prices are not bid/ask order book quotes.")


def get_team_logs(seasons: list[str]):
    import pandas as pd
    from nba_api.stats.endpoints import leaguegamelog

    cache_dir = DATA / "nba" / "team_logs"
    cache_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    for season in seasons:
        cache_path = cache_dir / f"team_logs_{season}.csv"
        if cache_path.is_file():
            print(f"NBA {season}: reading cached team game logs", flush=True)
            df = pd.read_csv(cache_path, dtype={"GAME_ID": str})
        else:
            print(f"NBA {season}: downloading regular-season team game logs", flush=True)
            def fetch():
                endpoint = leaguegamelog.LeagueGameLog(
                    season=season,
                    player_or_team_abbreviation="T",
                    season_type_all_star="Regular Season",
                    timeout=60,
                )
                return endpoint.get_data_frames()[0]
            df = retry(fetch, f"NBA game logs {season}")
            if df.empty:
                raise RuntimeError(f"No NBA team game logs returned for {season}")
            df["GAME_ID"] = df["GAME_ID"].astype(str).str.zfill(10)
            df.to_csv(cache_path, index=False)
        df["GAME_ID"] = df["GAME_ID"].astype(str).str.zfill(10)
        df = df[df["GAME_ID"].str.fullmatch(r"002\d{7}")].copy()
        df["SEASON"] = season
        frames.append(df)

    result = pd.concat(frames, ignore_index=True)
    result = result.drop_duplicates(subset=["GAME_ID", "TEAM_ID"], keep="last")
    result = result.sort_values(["GAME_DATE", "GAME_ID", "TEAM_ID"])
    DATA.mkdir(parents=True, exist_ok=True)
    out = DATA / "nba_all_seasons.csv"
    result.to_csv(out, index=False)
    print(f"NBA team history: {len(result):,} team-game rows -> {out}", flush=True)
    return result


def make_games(team_logs):
    import pandas as pd

    result = []
    for game_id, df in team_logs.groupby("GAME_ID", sort=False):
        if len(df) != 2:
            continue
        matchup = df["MATCHUP"].fillna("").astype(str)
        home = df[matchup.str.contains("vs.", regex=False)]
        away = df[matchup.str.contains("@", regex=False)]
        if len(home) != 1 or len(away) != 1:
            continue
        h = home.iloc[0]
        a = away.iloc[0]
        if pd.isna(h["PTS"]) or pd.isna(a["PTS"]):
            continue
        result.append({
            "GAME_ID": str(game_id),
            "SEASON": str(h["SEASON"]),
            "GAME_DATE": str(h["GAME_DATE"]),
            "HOME_TEAM": str(h["TEAM_ABBREVIATION"]),
            "AWAY_TEAM": str(a["TEAM_ABBREVIATION"]),
            "HOME_PTS": int(h["PTS"]),
            "AWAY_PTS": int(a["PTS"]),
            "HOME_TEAM_ID": int(h["TEAM_ID"]),
            "AWAY_TEAM_ID": int(a["TEAM_ID"]),
        })
    games = pd.DataFrame(result)
    if games.empty:
        raise RuntimeError("No paired, completed NBA games found; inspect TEAM MATCHUP and PTS fields")
    games = games.sort_values(["SEASON", "GAME_DATE", "GAME_ID"])
    out = DATA / "nba" / "games.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    games.to_csv(out, index=False)
    print(f"NBA completed games: {len(games):,} -> {out}", flush=True)
    return games


def fetch_pbp(game_id: str):
    import requests

    try:
        response = requests.get(CDN_URL.format(game_id=game_id), timeout=30)
        response.raise_for_status()
        payload = response.json()
        actions = payload["game"]["actions"]
        if not isinstance(actions, list) or len(actions) < 10:
            raise ValueError("NBA CDN returned insufficient play-by-play actions")
        return {"game_id": game_id, "source": "nba_cdn", "actions": actions}
    except (requests.RequestException, ValueError, KeyError, TypeError) as exc:
        print(f"  CDN unavailable for {game_id} ({exc}); trying stats PlayByPlayV3", flush=True)
        from nba_api.stats.endpoints import playbyplayv3
        endpoint = playbyplayv3.PlayByPlayV3(game_id=game_id, timeout=45)
        actions = json.loads(endpoint.play_by_play.get_data_frame().to_json(orient="records"))
        if len(actions) < 10:
            raise ValueError("NBA stats V3 returned insufficient play-by-play actions")
        return {"game_id": game_id, "source": "nba_stats_v3", "actions": actions}


def clock_seconds(value) -> float | None:
    if value is None:
        return None
    match = CLOCK_RE.match(str(value))
    if not match:
        return None
    h, m, s = match.groups()
    return float(h or 0) * 3600 + float(m or 0) * 60 + float(s or 0)


def score(value):
    if value is None or str(value).strip() == "":
        return None
    try:
        n = int(value)
        return n if n >= 0 else None
    except (ValueError, TypeError):
        return None


def action_rows(game_id: str, actions):
    rows = []
    sortable = []
    for i, action in enumerate(actions):
        try:
            period = int(action["period"])
            sec = clock_seconds(action.get("clock"))
            if sec is None or period < 1:
                continue
            max_sec = 720 if period <= 4 else 300
            if not (0 <= sec <= max_sec + 0.01):
                continue
            action_id = int(action.get("actionNumber") or action.get("actionId") or i)
            sortable.append((period, -sec, action_id, action))
        except (TypeError, ValueError, KeyError):
            continue
    sortable.sort(key=lambda item: (item[0], item[1], item[2]))
    home = away = 0
    for period, neg_sec, action_id, action in sortable:
        h = score(action.get("scoreHome"))
        a = score(action.get("scoreAway"))
        if h is not None:
            home = h
        if a is not None:
            away = a
        sec = -neg_sec
        elapsed = ((period - 1) * 720 if period <= 4 else 4 * 720 + (period - 5) * 300)
        elapsed += (720 if period <= 4 else 300) - sec
        rows.append({
            "GAME_ID": game_id,
            "PERIOD": period,
            "CLOCK_SECONDS": float(sec),
            "ELAPSED_SECONDS": float(elapsed),
            "SCORE_HOME": int(home),
            "SCORE_AWAY": int(away),
            "SCORE_MARGIN": int(home - away),
            "SCORE_TOTAL": int(home + away),
            "ACTION_NUMBER": action_id,
            "ACTION_TYPE": str(action.get("actionType") or ""),
        })
    return rows


def write_season_pbp(season, games, limit: int, sleep: float):
    import pyarrow as pa
    import pyarrow.parquet as pq

    raw_dir = DATA / "nba" / "play_by_play" / season
    raw_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = DATA / "nba" / f"play_by_play_{season}.parquet"
    schema = pa.schema([
        ("GAME_ID", pa.string()), ("PERIOD", pa.int16()),
        ("CLOCK_SECONDS", pa.float32()), ("ELAPSED_SECONDS", pa.float32()),
        ("SCORE_HOME", pa.int16()), ("SCORE_AWAY", pa.int16()),
        ("SCORE_MARGIN", pa.int16()), ("SCORE_TOTAL", pa.int16()),
        ("ACTION_NUMBER", pa.int32()), ("ACTION_TYPE", pa.string()),
    ])
    subset = games[games["SEASON"] == season]
    if limit:
        subset = subset.head(limit)
    if subset.empty:
        print(f"No games for {season}; skipping PBP", flush=True)
        return

    writer = None
    successes = 0
    failures = []
    try:
        for pos, game in enumerate(subset.itertuples(index=False), start=1):
            game_id = str(game.GAME_ID)
            cache_path = raw_dir / f"{game_id}.json"
            try:
                if cache_path.is_file():
                    with cache_path.open(encoding="utf-8") as f:
                        payload = json.load(f)
                else:
                    payload = retry(lambda: fetch_pbp(game_id), f"NBA PBP {game_id}")
                    tmp_path = cache_path.with_suffix(".tmp")
                    with tmp_path.open("w", encoding="utf-8") as f:
                        json.dump(payload, f, ensure_ascii=False)
                    os.replace(tmp_path, cache_path)
                    time.sleep(sleep)
                rows = action_rows(game_id, payload["actions"])
                if len(rows) < 10:
                    raise ValueError("Too few parseable actions")
                last = rows[-1]
                if (last["SCORE_HOME"], last["SCORE_AWAY"]) != (int(game.HOME_PTS), int(game.AWAY_PTS)):
                    raise ValueError(
                        f"PBP final score {last['SCORE_HOME']}-{last['SCORE_AWAY']} "
                        f"does not match game log {game.HOME_PTS}-{game.AWAY_PTS}"
                    )
                if writer is None:
                    tmp_parquet = parquet_path.with_suffix(".parquet.part")
                    writer = pq.ParquetWriter(tmp_parquet, schema, compression="zstd")
                table = pa.Table.from_pylist(rows, schema=schema)
                writer.write_table(table)
                successes += 1
            except (Exception, KeyboardInterrupt) as exc:
                if isinstance(exc, KeyboardInterrupt):
                    raise
                failures.append((game_id, str(exc)))
                print(f"  SKIP {game_id}: {exc}", flush=True)
            if pos == 1 or pos % 25 == 0 or pos == len(subset):
                print(f"NBA PBP {season}: {pos}/{len(subset)} processed, {successes} valid, {len(failures)} failed", flush=True)
    finally:
        if writer is not None:
            writer.close()
            os.replace(parquet_path.with_suffix(".parquet.part"), parquet_path)

    failures_path = DATA / "nba" / f"play_by_play_failures_{season}.csv"
    import csv
    with failures_path.open("w", encoding="utf-8", newline="") as f:
        out = csv.writer(f)
        out.writerow(["GAME_ID", "ERROR"])
        out.writerows(failures)
    print(f"Season {season}: valid games={successes}, failed={len(failures)}; output={parquet_path}", flush=True)


def get_nba(seasons: list[str], limit: int, sleep: float):
    logs = get_team_logs(seasons)
    games = make_games(logs)
    for season in seasons:
        write_season_pbp(season, games, limit, sleep)


def main():
    parser = argparse.ArgumentParser(description="Download historical NBA and Polymarket data for Parlay Guard")
    parser.add_argument("--only", choices=["nba", "polymarket", "both"], default="both")
    parser.add_argument("--seasons", nargs="+", default=["2024-25", "2025-26"])
    parser.add_argument("--max-games", type=int, default=0, help="Number of completed games PER season (0 = all)")
    parser.add_argument("--sleep", type=float, default=0.65, help="Pause between game requests")
    args = parser.parse_args()
    if args.max_games < 0 or args.sleep < 0:
        parser.error("--max-games and --sleep must not be negative")
    for season in args.seasons:
        if not re.fullmatch(r"20\d\d-\d\d", season):
            parser.error(f"Invalid season: {season}; expected e.g. 2025-26")
    if args.only in {"polymarket", "both"}:
        get_polymarket()
    if args.only in {"nba", "both"}:
        get_nba(args.seasons, args.max_games, args.sleep)
    print("DONE. Inspect files under data/. Large raw data should stay in .gitignore.", flush=True)


if __name__ == "__main__":
    main()
