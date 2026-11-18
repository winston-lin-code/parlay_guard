"""Score a previously saved, timestamped Parlay Guard pre-tipoff forecast.

Example, from project root:
    python -u -m src.score_live_predictions --predictions models/live_predictions_2026-11-18_YYYYMMDDTHHMMSSZ.csv

The NBA Stats feed may not include all final results until the next morning.
Only completed games in the frozen prediction file are scored. If any game is
unavailable, the script stops and asks you to retry; it never silently treats
missing games as losses. The saved original prediction file is never modified.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from .predict_today_backup import fetch_current_season


def score_saved_predictions(predictions: pd.DataFrame, season_results: pd.DataFrame):
    required = {"GAME_ID", "GAME_DATE", "AWAY_TEAM", "HOME_TEAM", "P_HOME_WIN_GB", "P_HOME_WIN_ELO", "GENERATED_AT_UTC", "TIPOFF_UTC"}
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"Prediction CSV missing {sorted(missing)}")
    predictions = predictions.copy()
    predictions["GAME_ID"] = predictions["GAME_ID"].astype(str).str.zfill(10)
    if predictions["GAME_ID"].duplicated().any():
        raise ValueError("Duplicate GAME_IDs in predictions")
    # A prediction is considered valid for honest pregame scoring only if its
    # recorded generation timestamp precedes the scheduled tipoff timestamp.
    predicted_at = pd.to_datetime(predictions["GENERATED_AT_UTC"], utc=True)
    tipoffs = pd.to_datetime(predictions["TIPOFF_UTC"], utc=True)
    if not (predicted_at < tipoffs).all():
        raise ValueError("Some forecasts were timestamped at/after tipoff; refusing to score as pregame")

    games = season_results.copy()
    games["GAME_ID"] = games["GAME_ID"].astype(str).str.zfill(10)
    games = games.loc[games.GAME_ID.isin(predictions.GAME_ID)].copy()
    counts = games.groupby("GAME_ID").agg(teams=("TEAM_ABBREVIATION", "size"), winners=("WL", lambda s: (s=="W").sum()))
    complete_ids = set(counts.loc[(counts.teams == 2) & (counts.winners == 1)].index)
    missing_ids = set(predictions.GAME_ID) - complete_ids
    if missing_ids:
        raise ValueError(f"{len(missing_ids)} forecast game(s) are not FINAL in NBA Stats yet: "
                         f"{sorted(missing_ids)}. Retry after results are available.")

    home = games.loc[games.IS_HOME.astype(str).str.lower().isin(["true", "1"])].copy()
    home["HOME_WIN"] = home["WL"].eq("W").astype(int)
    actual = home[["GAME_ID", "TEAM_ABBREVIATION", "HOME_WIN"]].rename(columns={"TEAM_ABBREVIATION": "ACTUAL_HOME_TEAM"})
    out = predictions.merge(actual, on="GAME_ID", validate="one_to_one")
    if not out["HOME_TEAM"].eq(out["ACTUAL_HOME_TEAM"]).all():
        raise ValueError("NBA results HOME_TEAM doesn't match saved forecast")
    out["ACTUAL_WINNER"] = np.where(out.HOME_WIN == 1, out.HOME_TEAM, out.AWAY_TEAM)

    y = out["HOME_WIN"].astype(int).to_numpy()
    metrics = {}
    for name, col in (("gradient_boosting", "P_HOME_WIN_GB"), ("elo_formula", "P_HOME_WIN_ELO")):
        p = pd.to_numeric(out[col], errors="raise").to_numpy(dtype=float)
        if np.any(~np.isfinite(p)) or np.any((p < 0) | (p > 1)):
            raise ValueError(f"Invalid probabilities in {col}")
        pp = np.clip(p, 1e-7, 1-1e-7)
        metrics[name] = {
            "games": len(y),
            "accuracy": float(((p >= 0.5) == y).mean()),
            "log_loss": float(log_loss(y, pp, labels=[0, 1])),
            "brier": float(brier_score_loss(y, p)),
            "roc_auc": float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else None,
        }
        out[f"{name.upper()}_CORRECT"] = ((p >= 0.5) == y).astype(int)
    return out, metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True, type=Path,
                        help="A saved live_predictions_*.csv created before tipoff")
    parser.add_argument("--season-csv", default=None, type=Path,
                        help="Optional NBA LeagueGameFinder current-season results CSV")
    args = parser.parse_args()
    if not args.predictions.is_file():
        raise FileNotFoundError(args.predictions)
    original = pd.read_csv(args.predictions, dtype={"GAME_ID": str})
    date_values = original.GAME_DATE.astype(str).unique()
    if len(date_values) != 1:
        raise ValueError("Forecast CSV should contain one game date")
    target = pd.Timestamp(date_values[0]).date()
    season = (f"{target.year}-{(target.year+1)%100:02d}" if target.month>=7
              else f"{target.year-1}-{target.year%100:02d}")

    scored_path = args.predictions.with_name(args.predictions.stem + "_scored.csv")
    metrics_path = args.predictions.with_name(args.predictions.stem + "_metrics.json")
    if scored_path.exists() or metrics_path.exists():
        raise FileExistsError(f"Already scored at {scored_path}; original predictions are preserved")

    season_results = fetch_current_season(season, args.season_csv)
    out, metrics = score_saved_predictions(original, season_results)
    out.to_csv(scored_path, index=False)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"===== PARLAY GUARD LIVE EVALUATION: {target} =====", flush=True)
    print(pd.DataFrame([{"model": key, **value} for key, value in metrics.items()]).to_string(
        index=False, float_format=lambda x: f"{x:.5f}"), flush=True)
    print("\nGame-level picks (saved predictions versus actual winners):", flush=True)
    print(out[["AWAY_TEAM", "HOME_TEAM", "GB_PICK", "ELO_PICK", "ACTUAL_WINNER"]].to_string(index=False), flush=True)
    print(f"\nOriginal forecast unchanged: {args.predictions}", flush=True)
    print(f"Saved scored games: {scored_path}", flush=True)
    print(f"Saved metrics: {metrics_path}", flush=True)


if __name__ == "__main__":
    main()
