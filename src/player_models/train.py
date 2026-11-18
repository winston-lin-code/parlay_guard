
"""Train and evaluate NBA player-stat prediction models."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "processed" / "player_training_data.csv"
MODEL_DIR = ROOT / "models" / "player_stats"

TARGETS = ["PTS", "REB", "AST", "FG3M", "STL", "BLK"]

TRAIN_END = "2025-10-01"
TEST_START = "2026-10-01"


def metrics(actual, predicted):
    return {
        "mae": round(float(mean_absolute_error(actual, predicted)), 4),
        "rmse": round(
            float(np.sqrt(mean_squared_error(actual, predicted))),
            4,
        ),
    }


def main():
    df = pd.read_csv(DATA_PATH)
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])

    # Match the feature columns generated in features.py.
    feature_cols = [
        "IS_HOME",
        "REST_DAYS",
        "GAMES_PLAYED_BEFORE",
    ]
    feature_cols += sorted(
        col for col in df.columns if "_LAST_" in col
    )

    # Chronological train / validation / test split.
    train = df[df["GAME_DATE"] < TRAIN_END].copy()

    validation = df[
        (df["GAME_DATE"] >= TRAIN_END)
        & (df["GAME_DATE"] < TEST_START)
    ].copy()

    test = df[df["GAME_DATE"] >= TEST_START].copy()

    print("Training rows:", len(train))
    print("Validation rows:", len(validation))
    print("Test rows:", len(test))
    print("Features:", len(feature_cols))

    if min(len(train), len(validation), len(test)) == 0:
        raise RuntimeError("One or more data splits are empty")

    if train[feature_cols].isna().any().any():
        raise RuntimeError("Missing training features")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    results = {}

    for target in TARGETS:
        print(f"\n========== {target} ==========")

        model = CatBoostRegressor(
            iterations=1000,
            depth=6,
            learning_rate=0.04,
            loss_function="RMSE",
            l2_leaf_reg=5,
            random_seed=42,
            thread_count=-1,
            verbose=False,
        )

        model.fit(
            train[feature_cols],
            train[target],
            eval_set=(
                validation[feature_cols],
                validation[target],
            ),
            early_stopping_rounds=80,
            verbose=False,
        )

        predictions = model.predict(test[feature_cols])

        # Stats cannot be negative.
        predictions = np.maximum(predictions, 0)

        # Historical baseline: previous 10 appearances.
        baseline_col = f"{target}_LAST_10"
        baseline = test[baseline_col].to_numpy()

        model_metrics = metrics(test[target], predictions)
        baseline_metrics = metrics(test[target], baseline)

        print("CatBoost:", model_metrics)
        print("10-game baseline:", baseline_metrics)

        improvement = (
            baseline_metrics["mae"] - model_metrics["mae"]
        )

        print("MAE improvement:", round(improvement, 4))

        model.save_model(
            str(MODEL_DIR / f"{target.lower()}_model.cbm")
        )

        results[target] = {
            "catboost": model_metrics,
            "baseline": baseline_metrics,
            "mae_improvement": round(improvement, 4),
            "best_iteration": model.best_iteration_,
        }

    metadata = {
        "features": feature_cols,
        "train_end_exclusive": TRAIN_END,
        "test_start": TEST_START,
        "train_rows": len(train),
        "validation_rows": len(validation),
        "test_rows": len(test),
        "results": results,
    }

    with open(
        MODEL_DIR / "training_results.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(metadata, file, indent=2)

    print("\nTraining complete.")
    print("Models saved to:", MODEL_DIR)


if __name__ == "__main__":
    main()
