
"""Train V2 NBA player-stat models and compare against V1."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = ROOT / "data/processed/player_training_data_v2.csv"
V1_MODEL_DIR = ROOT / "models/player_stats"
V2_MODEL_DIR = ROOT / "models/player_stats_v2"

TARGETS = ["PTS", "REB", "AST", "FG3M", "STL", "BLK"]

TRAIN_END = "2025-10-01"
TEST_START = "2026-10-01"


def metrics(actual, predicted):
    return {
        "mae": round(
            float(mean_absolute_error(actual, predicted)), 4
        ),
        "rmse": round(
            float(np.sqrt(mean_squared_error(actual, predicted))), 4
        ),
        "bias": round(
            float(np.mean(np.asarray(predicted) - np.asarray(actual))), 4
        ),
    }


def main():
    df = pd.read_csv(DATA_PATH)
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])

    v1_metadata_path = V1_MODEL_DIR / "training_results.json"

    with open(v1_metadata_path, encoding="utf-8") as f:
        v1_metadata = json.load(f)

    v1_features = v1_metadata["features"]

    v2_features = [
        "IS_HOME",
        "REST_DAYS",
        "GAMES_PLAYED_BEFORE",
        "CURRENT_SEASON_GAMES_PLAYED",
    ]

    v2_features += sorted(
        c for c in df.columns
        if "_LAST_" in c
        or "_STD_" in c
        or "_TREND_" in c
        or c.endswith("_SEASON_AVG")
    )

    assert len(v2_features) == 78, (
        f"Expected 78 features, got {len(v2_features)}"
    )

    train = df[df["GAME_DATE"] < TRAIN_END].copy()
    validation = df[
        (df["GAME_DATE"] >= TRAIN_END)
        & (df["GAME_DATE"] < TEST_START)
    ].copy()
    test = df[df["GAME_DATE"] >= TEST_START].copy()

    print("Training rows:", len(train))
    print("Validation rows:", len(validation))
    print("Test rows:", len(test))
    print("V2 features:", len(v2_features))

    V2_MODEL_DIR.mkdir(parents=True, exist_ok=True)

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
            train[v2_features],
            train[target],
            eval_set=(
                validation[v2_features],
                validation[target],
            ),
            early_stopping_rounds=80,
            verbose=False,
        )

        v2_pred = np.maximum(
            model.predict(test[v2_features]), 0
        )

        v1_model = CatBoostRegressor()
        v1_model.load_model(
            str(V1_MODEL_DIR / f"{target.lower()}_model.cbm")
        )

        v1_pred = np.maximum(
            v1_model.predict(test[v1_features]), 0
        )

        baseline_pred = test[
            f"{target}_LAST_10"
        ].to_numpy()

        v2_metrics = metrics(test[target], v2_pred)
        v1_metrics = metrics(test[target], v1_pred)
        baseline_metrics = metrics(test[target], baseline_pred)

        print("V2 CatBoost:    ", v2_metrics)
        print("V1 CatBoost:    ", v1_metrics)
        print("10-game average:", baseline_metrics)

        print(
            "V2 vs V1 MAE improvement:",
            round(v1_metrics["mae"] - v2_metrics["mae"], 4)
        )

        model.save_model(
            str(V2_MODEL_DIR / f"{target.lower()}_model.cbm")
        )

        results[target] = {
            "v2": v2_metrics,
            "v1": v1_metrics,
            "baseline": baseline_metrics,
            "best_iteration": model.best_iteration_,
        }

    metadata = {
        "features": v2_features,
        "train_end_exclusive": TRAIN_END,
        "test_start": TEST_START,
        "train_rows": len(train),
        "validation_rows": len(validation),
        "test_rows": len(test),
        "results": results,
    }

    with open(
        V2_MODEL_DIR / "training_results.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(metadata, f, indent=2)

    print("\nV2 training complete.")
    print("Models saved to:", V2_MODEL_DIR)


if __name__ == "__main__":
    main()
