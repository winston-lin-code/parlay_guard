
"""Calibrate NBA player-stat over/under probabilities.

Run:
    python -m src.player_models.calibrate

Output:
    models/player_stats_v2/calibration.json

Method:
    1. Train temporary models on historical chronological folds.
    2. Predict future games in each fold.
    3. Calculate out-of-fold prediction errors.
    4. Fit a regularized empirical error distribution.
    5. Evaluate over/under probabilities on held-out folds.

Important:
    Calibration evaluates probabilities at synthetic half-point lines.
    It does not evaluate historical sportsbook or Polymarket prices.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.metrics import brier_score_loss, log_loss


ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = (
    ROOT / "data" / "processed" / "player_training_data_v2.csv"
)

MODEL_DIR = ROOT / "models" / "player_stats_v2"

OUTPUT = MODEL_DIR / "calibration.json"

TARGETS = ["PTS", "REB", "AST", "FG3M", "STL", "BLK"]

# Chronological folds. Each model trains only on games
# before the validation period.
FOLDS = [
    ("2023-10-01", "2024-04-30"),
    ("2024-10-01", "2025-04-30"),
    ("2025-10-01", "2026-04-30"),
]

# Keep the 2026-27 season out of calibration development.
CUTOFF = "2026-10-01"

# Minimum residual observations for a local scoring group.
MIN_GROUP_SIZE = 200

# Shrink local residual distributions toward the global
# distribution, especially for sparse scoring groups.
SHRINKAGE = 500

# Half-point lines relative to the predicted expected value.
LINE_OFFSETS = [-8, -5, -3, -1, 0, 1, 3, 5, 8]


def make_model():
    return CatBoostRegressor(
        iterations=700,
        depth=6,
        learning_rate=0.04,
        loss_function="RMSE",
        l2_leaf_reg=5,
        random_seed=42,
        thread_count=-1,
        verbose=False,
    )


def scoring_group(predicted, target):
    """Group by expected stat total, not actual outcome."""

    if target == "PTS":
        bins = [-np.inf, 10, 20, 30, np.inf]
    elif target in ["REB", "AST"]:
        bins = [-np.inf, 3, 6, 10, np.inf]
    elif target == "FG3M":
        bins = [-np.inf, 1, 2, 4, np.inf]
    else:
        bins = [-np.inf, 0.5, 1.5, np.inf]

    return int(np.searchsorted(bins[1:], predicted, side="right"))


def empirical_over_probability(
    predicted,
    line,
    local_errors,
    global_errors,
):
    """Estimate P(actual > line) from historical residuals.

    Error is defined as actual - predicted.
    """

    threshold = line - predicted

    global_prob = float(
        np.mean(global_errors > threshold)
    )

    if len(local_errors) < MIN_GROUP_SIZE:
        return global_prob

    local_prob = float(
        np.mean(local_errors > threshold)
    )

    weight = len(local_errors) / (
        len(local_errors) + SHRINKAGE
    )

    probability = (
        weight * local_prob
        + (1 - weight) * global_prob
    )

    return float(np.clip(probability, 0.001, 0.999))


def create_oof_predictions(df, features, target):
    """Generate predictions from forward chronological folds."""

    frames = []

    for start, end in FOLDS:
        train = df[df["GAME_DATE"] < start]

        validation = df[
            (df["GAME_DATE"] >= start)
            & (df["GAME_DATE"] <= end)
        ]

        if train.empty or validation.empty:
            print(
                f"Skipping fold {start} to {end}: empty split"
            )
            continue

        print(
            f"  Fold {start} to {end}: "
            f"train={len(train):,}, "
            f"validation={len(validation):,}"
        )

        model = make_model()

        model.fit(
            train[features],
            train[target],
            verbose=False,
        )

        predictions = np.maximum(
            model.predict(validation[features]),
            0,
        )

        frame = validation[
            ["GAME_DATE", "PLAYER_ID", target]
        ].copy()

        frame["PREDICTED"] = predictions
        frame["ACTUAL"] = frame[target]
        frame["ERROR"] = (
            frame["ACTUAL"] - frame["PREDICTED"]
        )

        frame["GROUP"] = [
            scoring_group(p, target)
            for p in predictions
        ]

        frames.append(frame)

    if not frames:
        raise RuntimeError(
            f"No out-of-fold predictions for {target}"
        )

    return pd.concat(frames, ignore_index=True)


def half_point_line(value):
    """Round a nonnegative expected total to a half-point line."""
    return max(0.5, np.floor(value) + 0.5)


def evaluate_calibration(oof, target):
    """Forward evaluation: calibrate on earlier folds,
    evaluate on the final fold.

    Synthetic lines are used solely for probability testing.
    """

    evaluation_start = pd.Timestamp(FOLDS[-1][0])

    calibration = oof[
        oof["GAME_DATE"] < evaluation_start
    ]

    evaluation = oof[
        oof["GAME_DATE"] >= evaluation_start
    ]

    if calibration.empty or evaluation.empty:
        return None

    global_errors = calibration["ERROR"].to_numpy()

    grouped = {
        int(group): rows["ERROR"].to_numpy()
        for group, rows in calibration.groupby("GROUP")
    }

    probabilities = []
    outcomes = []

    for row in evaluation.itertuples():
        predicted = float(row.PREDICTED)
        actual = float(row.ACTUAL)

        group = int(row.GROUP)

        local_errors = grouped.get(
            group, np.array([])
        )

        for offset in LINE_OFFSETS:
            line = half_point_line(predicted + offset)

            probability = empirical_over_probability(
                predicted,
                line,
                local_errors,
                global_errors,
            )

            outcome = int(actual > line)

            probabilities.append(probability)
            outcomes.append(outcome)

    probabilities = np.asarray(probabilities)
    outcomes = np.asarray(outcomes)

    return {
        "samples": int(len(outcomes)),
        "brier_score": round(
            float(brier_score_loss(
                outcomes, probabilities
            )), 5
        ),
        "log_loss": round(
            float(log_loss(
                outcomes,
                probabilities,
                labels=[0, 1],
            )), 5
        ),
        "actual_over_rate": round(
            float(outcomes.mean()), 5
        ),
        "mean_predicted_over_probability": round(
            float(probabilities.mean()), 5
        ),
    }


def build_residual_artifact(oof):
    """Save residuals for probability estimation.

    Store observations individually rather than assuming
    a normal distribution.
    """

    return {
        "global_errors": (
            oof["ERROR"].round(4).tolist()
        ),
        "groups": {
            str(int(group)): (
                rows["ERROR"].round(4).tolist()
            )
            for group, rows in oof.groupby("GROUP")
        },
        "sample_count": int(len(oof)),
        "residual_mean": round(
            float(oof["ERROR"].mean()), 4
        ),
        "residual_std": round(
            float(oof["ERROR"].std()), 4
        ),
    }


def main():
    df = pd.read_csv(
        DATA_PATH,
        dtype={"PLAYER_ID": str},
    )

    df["GAME_DATE"] = pd.to_datetime(
        df["GAME_DATE"]
    )

    df = df[
        df["GAME_DATE"] < CUTOFF
    ].copy()

    metadata_path = (
        MODEL_DIR / "training_results.json"
    )

    with open(
        metadata_path,
        encoding="utf-8",
    ) as file:
        metadata = json.load(file)

    features = metadata["features"]

    missing = [
        col for col in features
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing features: {missing}"
        )

    print("Calibration dataset rows:", len(df))
    print("Features:", len(features))

    artifact = {
        "version": "v2",
        "method": "empirical_residual_shrinkage",
        "error_definition": "actual_minus_predicted",
        "min_group_size": MIN_GROUP_SIZE,
        "shrinkage": SHRINKAGE,
        "cutoff_exclusive": CUTOFF,
        "folds": FOLDS,
        "targets": {},
    }

    for target in TARGETS:
        print(f"\n========== {target} ==========")

        oof = create_oof_predictions(
            df, features, target
        )

        evaluation = evaluate_calibration(
            oof, target
        )

        residuals = build_residual_artifact(oof)

        artifact["targets"][target] = {
            "evaluation": evaluation,
            **residuals,
        }

        print(
            "Residual samples:",
            residuals["sample_count"],
        )
        print(
            "Residual mean:",
            residuals["residual_mean"],
        )
        print(
            "Residual std:",
            residuals["residual_std"],
        )
        print(
            "Calibration evaluation:",
            evaluation,
        )

    MODEL_DIR.mkdir(
        parents=True, exist_ok=True
    )

    with open(
        OUTPUT,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(artifact, file, indent=2)

    print("\nCalibration complete.")
    print("Saved to:", OUTPUT)


if __name__ == "__main__":
    main()
