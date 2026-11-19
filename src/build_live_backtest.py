"""Create leakage-safe NBA in-game snapshots, then train/test baseline live margin and total models.

Put this file in the Parlay Guard project root, next to the data/ folder.

Usage:
    python build_live_backtest.py --only snapshots
    python build_live_backtest.py --only evaluate
    python build_live_backtest.py --only both

Required: pandas, numpy, pyarrow, scikit-learn

The 2024-25 season is used to train and calibrate; 2025-26 is the held-out test.
No historical market odds are used in this baseline. This does NOT measure betting EV.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
NBA = ROOT / "data" / "nba"
OUTPUT = ROOT / "data" / "processed" / "live_games"
CHECKPOINTS = {
    "Q1_END": (1, 0.0),
    "HALFTIME": (2, 0.0),
    "Q3_END": (3, 0.0),
    "Q4_6M": (4, 360.0),
    "Q4_3M": (4, 180.0),
}
PBP_COLS = ["GAME_ID", "PERIOD", "CLOCK_SECONDS", "ELAPSED_SECONDS",
            "SCORE_HOME", "SCORE_AWAY", "ACTION_NUMBER"]
FEATURES = ["CURRENT_MARGIN", "CURRENT_TOTAL", "ELAPSED_SECONDS",
            "PREGAME_ELO_DIFF", "PREGAME_HOME_WIN_PROB", "PERIOD"]


def load_games() -> pd.DataFrame:
    path = NBA / "games.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}; run the NBA downloader first.")
    games = pd.read_csv(path, dtype={"GAME_ID": str}).copy()
    required = {"GAME_ID", "SEASON", "GAME_DATE", "HOME_TEAM", "AWAY_TEAM", "HOME_PTS", "AWAY_PTS"}
    missing = required - set(games.columns)
    if missing:
        raise ValueError(f"Missing game columns: {sorted(missing)}")
    games["GAME_ID"] = games["GAME_ID"].str.zfill(10)
    games["GAME_DATE"] = pd.to_datetime(games["GAME_DATE"])
    games["HOME_PTS"] = pd.to_numeric(games["HOME_PTS"], errors="raise")
    games["AWAY_PTS"] = pd.to_numeric(games["AWAY_PTS"], errors="raise")
    games = games.sort_values(["GAME_DATE", "GAME_ID"])
    if games["GAME_ID"].duplicated().any():
        raise ValueError("Duplicate GAME_ID in games.csv")
    return games.reset_index(drop=True)


def attach_pregame_elo(games: pd.DataFrame, k: float = 20.0, home_advantage: float = 65.0):
    """Illustrative Elo derived *only* from prior game results, not the final game being predicted."""
    ratings: dict[str, float] = {}
    last_season = None
    output = []
    for g in games.itertuples(index=False):
        if last_season is not None and g.SEASON != last_season:
            ratings = {team: 1500.0 + 0.75 * (rating - 1500.0)
                       for team, rating in ratings.items()}
        last_season = g.SEASON
        home = float(ratings.get(g.HOME_TEAM, 1500.0))
        away = float(ratings.get(g.AWAY_TEAM, 1500.0))
        elo_diff = home - away + home_advantage
        p_home = 1 / (1 + 10 ** (-elo_diff / 400.0))
        output.append((g.GAME_ID, elo_diff, p_home))
        actual = float(g.HOME_PTS > g.AWAY_PTS)
        delta = k * (actual - p_home)
        ratings[g.HOME_TEAM] = home + delta
        ratings[g.AWAY_TEAM] = away - delta
    return games.merge(pd.DataFrame(output, columns=["GAME_ID", "PREGAME_ELO_DIFF", "PREGAME_HOME_WIN_PROB"]),
                       on="GAME_ID", how="left", validate="one_to_one")


def snapshots_for_season(pbp: pd.DataFrame, season: str) -> pd.DataFrame:
    missing = set(PBP_COLS) - set(pbp.columns)
    if missing:
        raise ValueError(f"PBP for {season} missing columns: {sorted(missing)}")
    pbp = pbp.copy()
    pbp["GAME_ID"] = pbp["GAME_ID"].astype(str).str.zfill(10)
    pbp = pbp.dropna(subset=["PERIOD", "CLOCK_SECONDS", "ELAPSED_SECONDS", "SCORE_HOME", "SCORE_AWAY"])
    pbp = pbp.sort_values(["GAME_ID", "ELAPSED_SECONDS", "ACTION_NUMBER"])
    records = []
    for stage, (period, clock_at_checkpoint) in CHECKPOINTS.items():
        # Only events ALREADY OBSERVED at the checkpoint are eligible.
        # The game clock runs down, so events with CLOCK_SECONDS less than the
        # checkpoint would be from the future and must not be included.
        eligible = pbp[(pbp["PERIOD"] == period)
                       & (pbp["CLOCK_SECONDS"] >= clock_at_checkpoint - 0.0001)].copy()
        if eligible.empty:
            continue
        last_known = eligible.groupby("GAME_ID", sort=False).tail(1).copy()
        # Reject a suspiciously long event gap near the checkpoint.
        age = last_known["CLOCK_SECONDS"] - clock_at_checkpoint
        last_known = last_known[age <= 90.0].copy()
        if last_known.empty:
            continue
        # The score is as-of the checkpoint, but the event might have happened earlier.
        # Give all snapshots their actual fixed checkpoint elapsed time.
        last_known["ELAPSED_SECONDS"] = (period - 1) * 720 + (720 - clock_at_checkpoint)
        last_known["STAGE"] = stage
        last_known["OBSERVED_EVENT_CLOCK"] = last_known["CLOCK_SECONDS"]
        last_known["CLOCK_SECONDS"] = clock_at_checkpoint
        last_known["SEASON"] = season
        records.append(last_known[["GAME_ID", "SEASON", "STAGE", "PERIOD", "CLOCK_SECONDS",
                                   "ELAPSED_SECONDS", "OBSERVED_EVENT_CLOCK", "SCORE_HOME", "SCORE_AWAY"]])
    if not records:
        raise ValueError(f"No valid snapshots for {season}")
    return pd.concat(records, ignore_index=True)


def build_snapshots() -> pd.DataFrame:
    games = attach_pregame_elo(load_games())
    frames = []
    for season in sorted(games["SEASON"].unique()):
        path = NBA / f"play_by_play_{season}.parquet"
        if not path.exists():
            print(f"SKIP missing season PBP: {path}", flush=True)
            continue
        pbp = pd.read_parquet(path, columns=PBP_COLS)
        print(f"Loaded {season}: {len(pbp):,} PBP events / {pbp.GAME_ID.nunique():,} games", flush=True)
        frames.append(snapshots_for_season(pbp, season))
    if not frames:
        raise FileNotFoundError(f"No play_by_play_*.parquet files under {NBA}")
    snaps = pd.concat(frames, ignore_index=True)
    snaps = snaps.merge(games, on=["GAME_ID", "SEASON"], how="inner", validate="many_to_one")
    snaps["CURRENT_MARGIN"] = snaps["SCORE_HOME"] - snaps["SCORE_AWAY"]
    snaps["CURRENT_TOTAL"] = snaps["SCORE_HOME"] + snaps["SCORE_AWAY"]
    snaps["FINAL_MARGIN"] = snaps["HOME_PTS"] - snaps["AWAY_PTS"]
    snaps["FINAL_TOTAL"] = snaps["HOME_PTS"] + snaps["AWAY_PTS"]
    snaps["REMAINING_MARGIN"] = snaps["FINAL_MARGIN"] - snaps["CURRENT_MARGIN"]
    snaps["REMAINING_TOTAL"] = snaps["FINAL_TOTAL"] - snaps["CURRENT_TOTAL"]
    snaps["HOME_WIN"] = (snaps["FINAL_MARGIN"] > 0).astype(int)
    # Discard malformed games/snapshots: current score cannot exceed the final.
    before = len(snaps)
    snaps = snaps[(snaps["SCORE_HOME"] >= 0) & (snaps["SCORE_AWAY"] >= 0)
                  & (snaps["SCORE_HOME"] <= snaps["HOME_PTS"])
                  & (snaps["SCORE_AWAY"] <= snaps["AWAY_PTS"])
                  & (snaps["FINAL_MARGIN"] != 0)].copy()
    print(f"Removed {before - len(snaps):,} invalid snapshots", flush=True)
    if snaps.duplicated(["GAME_ID", "STAGE"]).any():
        raise AssertionError("Multiple snapshots for the same game and stage")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    out = OUTPUT / "snapshots.parquet"
    snaps.sort_values(["GAME_DATE", "GAME_ID", "ELAPSED_SECONDS"]).to_parquet(out, index=False)
    print(f"Saved {len(snaps):,} snapshots from {snaps.GAME_ID.nunique():,} games -> {out}", flush=True)
    summary = snaps.groupby(["SEASON", "STAGE"]).agg(
        games=("GAME_ID", "nunique"), rows=("STAGE", "size")
    ).reset_index()
    print(summary.to_string(index=False), flush=True)
    return snaps


def std_normal_cdf(z: np.ndarray) -> np.ndarray:
    from scipy.special import ndtr
    return ndtr(z)


def evaluate() -> None:
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.metrics import mean_absolute_error, mean_squared_error, brier_score_loss, log_loss

    path = OUTPUT / "snapshots.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}; run --only snapshots first")
    df = pd.read_parquet(path)
    train_seasons = ["2024-25"]
    test_season = "2025-26"
    old = df[df["SEASON"].isin(train_seasons)].copy()
    test = df[df["SEASON"] == test_season].copy()
    if old.GAME_ID.nunique() < 200 or test.GAME_ID.nunique() < 200:
        raise RuntimeError("Not enough training/test games (need at least 200 per group)")
    # Split training season by unique game date, ensuring no game snapshots leak.
    dates = sorted(old.GAME_DATE.unique())
    cutoff = dates[max(1, int(len(dates)*0.80))]
    fit = old[old.GAME_DATE < cutoff].copy()
    cal = old[old.GAME_DATE >= cutoff].copy()
    if fit.GAME_ID.nunique() < 100 or cal.GAME_ID.nunique() < 30:
        raise RuntimeError("Not enough chronological fit/calibration games")
    models = {}
    for target in ["REMAINING_MARGIN", "REMAINING_TOTAL"]:
        mdl = HistGradientBoostingRegressor(
            max_iter=180, max_leaf_nodes=15, min_samples_leaf=30,
            learning_rate=0.05, l2_regularization=0.2, random_state=42,
        )
        mdl.fit(fit[FEATURES], fit[target])
        models[target] = mdl
    cal = cal.copy()
    cal["PRED_MARGIN"] = cal["CURRENT_MARGIN"] + models["REMAINING_MARGIN"].predict(cal[FEATURES])
    cal["PRED_TOTAL"] = cal["CURRENT_TOTAL"] + models["REMAINING_TOTAL"].predict(cal[FEATURES])
    # Stage-specific uncertainty estimated only from the calibration period.
    stage_std = {}
    for stage, sub in cal.groupby("STAGE"):
        stage_std[stage] = {
            "margin_sigma": float(max(2.0, np.sqrt(np.mean((sub["FINAL_MARGIN"]-sub["PRED_MARGIN"])**2)))),
            "total_sigma": float(max(2.0, np.sqrt(np.mean((sub["FINAL_TOTAL"]-sub["PRED_TOTAL"])**2)))),
        }
    test = test.copy()
    test["PRED_MARGIN"] = test["CURRENT_MARGIN"] + models["REMAINING_MARGIN"].predict(test[FEATURES])
    test["PRED_TOTAL"] = test["CURRENT_TOTAL"] + models["REMAINING_TOTAL"].predict(test[FEATURES])
    test["MARGIN_SIGMA"] = test["STAGE"].map(lambda s: stage_std[s]["margin_sigma"])
    test["TOTAL_SIGMA"] = test["STAGE"].map(lambda s: stage_std[s]["total_sigma"])
    test["P_HOME_WIN"] = np.clip(std_normal_cdf(test["PRED_MARGIN"] / test["MARGIN_SIGMA"]), 0.0001, 0.9999)
    # A score/clock baseline with prior team Elo influence, not a direct market price.
    test["BASELINE_MARGIN"] = test["CURRENT_MARGIN"] + (2880.0-test["ELAPSED_SECONDS"])/2880.0 * (
        12.0 * (test["PREGAME_HOME_WIN_PROB"] - 0.5)
    )
    lines = []
    for stage, sub in test.groupby("STAGE", sort=False):
        lines.append({
            "stage": stage,
            "games": int(sub.GAME_ID.nunique()),
            "margin_mae": round(float(mean_absolute_error(sub.FINAL_MARGIN, sub.PRED_MARGIN)), 3),
            "baseline_margin_mae": round(float(mean_absolute_error(sub.FINAL_MARGIN, sub.BASELINE_MARGIN)), 3),
            "total_mae": round(float(mean_absolute_error(sub.FINAL_TOTAL, sub.PRED_TOTAL)), 3),
            "home_win_brier": round(float(brier_score_loss(sub.HOME_WIN, sub.P_HOME_WIN)), 5),
            "home_win_logloss": round(float(log_loss(sub.HOME_WIN, sub.P_HOME_WIN, labels=[0, 1])), 5),
        })
    results = pd.DataFrame(lines)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUTPUT / "test_metrics_2025-26.csv", index=False)
    test.to_parquet(OUTPUT / "test_predictions_2025-26.parquet", index=False)
    with (OUTPUT / "calibration_2024-25.json").open("w", encoding="utf-8") as f:
        json.dump(stage_std, f, indent=2)
    print(f"Fit: {fit.GAME_ID.nunique()} games; calibration: {cal.GAME_ID.nunique()} games; held-out test: {test.GAME_ID.nunique()} games", flush=True)
    print("\n=== HELD-OUT LIVE GAME BASELINE (2025-26) ===", flush=True)
    print(results.to_string(index=False), flush=True)
    print(f"\nOutput: {OUTPUT}", flush=True)
    print("NOTE: These are prediction metrics, NOT a historical betting/EV backtest.", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", choices=["both", "snapshots", "evaluate"], default="both")
    args = parser.parse_args()
    if args.only in {"both", "snapshots"}:
        build_snapshots()
    if args.only in {"both", "evaluate"}:
        evaluate()

if __name__ == "__main__":
    main()
