"""Backtest player props on 2025-26, then save 2026-27 production models.

Usage:
  python train_player_props.py

The historical 2025-26 predictions remain OUT OF SAMPLE (for calibration).
The production models are trained on 2021-22 through 2025-26, not 2026-27.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

from parlay_guard.src.player_prop_common import FEATURES, TARGETS

TRAIN_SEASONS = ["2021-22", "2022-23", "2023-24", "2024-25"]
HOLDOUT_SEASON = "2025-26"
MODEL_PATH = Path("models/player_prop_models_2026_27.joblib")
HOLDOUT_PATH = Path("data/processed/player_predictions_2025-26.parquet")


def make_model():
    return HistGradientBoostingRegressor(
        max_iter=250, max_leaf_nodes=20, min_samples_leaf=40,
        learning_rate=0.05, l2_regularization=0.5, random_state=42,
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, default=Path("data/processed/player_training_data_v2.csv"))
    args = ap.parse_args()
    if not args.data.exists():
        ap.error(f"Missing {args.data}")
    df = pd.read_csv(args.data, low_memory=False)
    needed = set(FEATURES + TARGETS + ["SEASON", "GAME_ID", "GAME_DATE", "PLAYER_ID", "PLAYER_NAME", "TEAM_ID"])
    missing = needed - set(df.columns)
    if missing:
        ap.error(f"Missing training columns: {sorted(missing)}")
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"], errors="raise")
    for col in FEATURES + TARGETS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    train = df[df.SEASON.isin(TRAIN_SEASONS)].copy()
    holdout = df[df.SEASON == HOLDOUT_SEASON].copy()
    if train.empty or holdout.empty:
        ap.error("Need 2021-22 through 2024-25 for training and 2025-26 for holdout")
    print(f"Train: {len(train):,} rows | 2025-26 holdout: {len(holdout):,} rows")
    prediction_rows = holdout[["GAME_ID", "GAME_DATE", "PLAYER_ID", "PLAYER_NAME", "TEAM_ID"] + TARGETS].copy()
    metrics = {}
    prod_models = {}
    for stat in TARGETS:
        valid_train = train.dropna(subset=[stat]); valid_holdout = holdout.dropna(subset=[stat])
        if valid_train.empty or valid_holdout.empty:
            raise ValueError(f"No usable rows for {stat}")
        evaluated = make_model()
        evaluated.fit(valid_train[FEATURES], valid_train[stat])
        pred = np.maximum(0.0, evaluated.predict(valid_holdout[FEATURES]))
        prediction_rows[f"PRED_{stat}"] = np.nan
        prediction_rows.loc[valid_holdout.index, f"PRED_{stat}"] = pred
        prediction_rows[f"RESIDUAL_{stat}"] = prediction_rows[stat] - prediction_rows[f"PRED_{stat}"]
        baseline_col = f"{stat}_LAST_10"
        valid_baseline = valid_holdout.dropna(subset=[baseline_col])
        mae = float(mean_absolute_error(valid_holdout[stat], pred))
        rmse = float(np.sqrt(mean_squared_error(valid_holdout[stat], pred)))
        baseline_mae = float(mean_absolute_error(valid_baseline[stat], valid_baseline[baseline_col])) if len(valid_baseline) else None
        metrics[stat] = {"model_mae": mae, "model_rmse": rmse, "baseline_last10_mae": baseline_mae,
                         "n_holdout": int(len(valid_holdout))}
        print(f"{stat:5} holdout MAE={mae:.3f}, RMSE={rmse:.3f}, last10 baseline MAE={baseline_mae}")
        # The evaluation model never sees the holdout. Refit separately for
        # deployment *after* recording that honest out-of-sample evaluation.
        final_train = df[df.SEASON.isin(TRAIN_SEASONS + [HOLDOUT_SEASON])].dropna(subset=[stat])
        final_model = make_model()
        final_model.fit(final_train[FEATURES], final_train[stat])
        prod_models[stat] = final_model
    HOLDOUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    prediction_rows.to_parquet(HOLDOUT_PATH, index=False)
    bundle = {
        "models": prod_models,
        "features": FEATURES,
        "targets": TARGETS,
        "training_seasons": TRAIN_SEASONS + [HOLDOUT_SEASON],
        "holdout_season": HOLDOUT_SEASON,
        "holdout_metrics": metrics,
        "saved_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_type": "HistGradientBoostingRegressor",
    }
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, MODEL_PATH)
    print(f"Saved out-of-sample holdout predictions: {HOLDOUT_PATH}")
    print(f"Saved FOUR production player models: {MODEL_PATH}")
    print("Not yet a betting model: needs current pregame features, availability data, calibration, and execution-cost validation.")


if __name__ == "__main__":
    main()
