
"""Collect historical NBA player game logs for ML training.

Run from the project root:
    python -m src.collect_prop_history --seasons 2021-22 2022-23 2023-24 2024-25 2025-26 2026-27

The output contains regular-season player box scores.
"""

import argparse
from pathlib import Path
import time

import pandas as pd
from nba_api.stats.endpoints import playergamelogs

ROOT = Path(__file__).resolve().parents[1]

# Keep the original CSV untouched until the new dataset is verified.
OUTPUT = ROOT / "data" / "nba_player_stats_history.csv"

COLUMNS = [
    "GAME_ID",
    "GAME_DATE",
    "PLAYER_ID",
    "PLAYER_NAME",
    "TEAM_ID",
    "TEAM_ABBREVIATION",
    "MATCHUP",
    "WL",
    "MIN",
    "PTS",
    "REB",
    "AST",
    "FG3M",
    "STL",
    "BLK",
    "FGA",
    "FGM",
    "FG3A",
    "FTA",
    "FTM",
    "OREB",
    "DREB",
    "TOV",
    "PF",
    "PLUS_MINUS",
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seasons",
        nargs="+",
        default=[
            "2021-22",
            "2022-23",
            "2023-24",
            "2024-25",
            "2025-26",
            "2026-27",
        ],
    )
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--pause", type=float, default=3.0)
    args = parser.parse_args()

    frames = []

    for i, season in enumerate(args.seasons):
        print(f"Fetching NBA player logs: {season}", flush=True)

        endpoint = playergamelogs.PlayerGameLogs(
            season_nullable=season,
            season_type_nullable="Regular Season",
            timeout=60,
        )

        df = endpoint.get_data_frames()[0]

        missing = set(COLUMNS) - set(df.columns)
        if missing:
            raise RuntimeError(
                f"{season}: Missing columns: {sorted(missing)}"
            )

        df = df[COLUMNS].copy()
        df["SEASON"] = season

        frames.append(df)

        print(
            f"Retrieved {len(df):,} player appearances",
            flush=True,
        )

        if i + 1 < len(args.seasons):
            time.sleep(args.pause)

    if not frames:
        raise RuntimeError("No player logs returned")

    out = pd.concat(frames, ignore_index=True)

    out["GAME_ID"] = out["GAME_ID"].astype(str).str.zfill(10)

    out["GAME_DATE"] = pd.to_datetime(
        out["GAME_DATE"], errors="raise"
    ).dt.strftime("%Y-%m-%d")

    out = out.drop_duplicates(
        ["GAME_ID", "PLAYER_ID"]
    ).sort_values(["GAME_DATE", "PLAYER_ID"])

    args.output.parent.mkdir(parents=True, exist_ok=True)

    out.to_csv(args.output, index=False)

    print(f"\nSaved {len(out):,} player appearances")
    print(f"Output: {args.output}")
    print(f"Date range: {out['GAME_DATE'].min()} to {out['GAME_DATE'].max()}")
    print(f"Unique players: {out['PLAYER_ID'].nunique()}")
    print("\nNote: Injury and lineup availability require separate data.")


if __name__ == "__main__":
    main()
