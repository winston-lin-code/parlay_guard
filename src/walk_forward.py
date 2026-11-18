"""Expanding-window NBA model comparison, excluding the final held-out test period.

Run from the project root: python -u -m src.walk_forward
"""
import json
import time

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .common import DATA, MODELS
from .train import NUMERIC, CATEGORICAL


def make_model(include_team_identity, include_point_diff=False, elo_mode=None):
    
    numeric = [f for f in NUMERIC if include_point_diff or "POINT_DIFF_LAST_10" not in f]
    if elo_mode == "ratings":
        numeric += ["HOME_ELO", "AWAY_ELO"]
    elif elo_mode == "difference":
        numeric += ["ELO_DIFF"]
    elif elo_mode is not None:
        raise ValueError(f"Unknown Elo mode: {elo_mode}")
    transformers = [
        ("numeric", Pipeline([
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("scaler", StandardScaler()),
                ]), numeric),
    ]
    inputs = list(numeric)
    if include_team_identity:
        transformers.append((
            "teams", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            CATEGORICAL,
        ))
        inputs += list(CATEGORICAL)
    model = Pipeline([
        ("preprocessor", ColumnTransformer(transformers, sparse_threshold=0)),
        ("classifier", GradientBoostingClassifier(
            n_estimators=100, learning_rate=0.05, max_depth=2, random_state=42,
        )),
    ])
    return model, inputs


def score(y, p):
    p = np.clip(np.asarray(p), 1e-7, 1 - 1e-7)
    return {
        "accuracy": float(accuracy_score(y, p >= 0.5)),
        "brier": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "roc_auc": float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else None,
    }


def main():
    path = DATA / "nba_model_dataset.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run python -m src.pipeline first.")
    df = pd.read_csv(path, dtype={"GAME_ID": str})
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    df = df.sort_values(["GAME_DATE", "GAME_ID"]).reset_index(drop=True)
    dates = np.array(sorted(df["GAME_DATE"].unique()))
    if len(dates) < 30:
        raise ValueError("Not enough unique game dates for walk-forward validation")

    # Same 85% cutoff as experiments.py; never inspect the final 15% of dates.
    boundaries = [int(len(dates) * fraction) for fraction in (0.40, 0.55, 0.70, 0.85)]
    if len(set(boundaries)) != 4:
        raise ValueError("Date boundaries overlap; insufficient historical dates")
    test_start = pd.Timestamp(dates[boundaries[-1]])
    print(f"Final held-out test starts {test_start.date()} (never evaluated)", flush=True)

    required = ["HOME_POINT_DIFF_LAST_10", "AWAY_POINT_DIFF_LAST_10"]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(
            f"Missing columns {missing}. Run python -m src.feature_engineering "
            "and python -m src.merge_features first."
        )

    elo_columns = ["HOME_ELO", "AWAY_ELO", "ELO_DIFF"]
    if any(col not in df.columns for col in elo_columns):
        raise ValueError("Missing Elo features. Run python -m src.merge_features first.")

    rows = []
    for fold in range(3):
        train_end = pd.Timestamp(dates[boundaries[fold]])
        valid_end = pd.Timestamp(dates[boundaries[fold + 1]])
        train = df[df["GAME_DATE"] < train_end]
        valid = df[(df["GAME_DATE"] >= train_end) & (df["GAME_DATE"] < valid_end)]
        if train.empty or valid.empty or train["HOME_WIN"].nunique() < 2:
            raise ValueError(f"Invalid train/validation data in fold {fold + 1}")

        print(f"\nFold {fold + 1}/3 | train: {len(train)} games before {train_end.date()} "
              f"| validate: {len(valid)} games through {(valid_end - pd.Timedelta(days=1)).date()}",
              flush=True)
        for name, use_teams, use_point_diff, elo_mode in (
            ("gradient_boosting", True, False, None),
            ("gb_no_team_identity", False, False, None),
            ("gb_with_point_diff", True, True, None),
            ("gb_no_team_with_point_diff", False, True, None),
            ("gb_elo_only", False, False, "ratings"),
            ("gb_with_point_diff_and_elo", True, True, "ratings"),
            ("gb_no_team_point_diff_elo", False, True, "ratings"),
            ("gb_no_team_point_diff_elo_diff", False, True, "difference"),
        ):
            model, inputs = make_model(use_teams, use_point_diff, elo_mode)
            start = time.perf_counter()
            model.fit(train[inputs], train["HOME_WIN"].astype(int))
            probabilities = model.predict_proba(valid[inputs])[:, 1]
            metrics = score(valid["HOME_WIN"].astype(int), probabilities)
            row = {
                "fold": fold + 1, "model": name,
                "train_start": str(train["GAME_DATE"].min().date()),
                "train_end_exclusive": str(train_end.date()),
                "validation_start": str(train_end.date()),
                "validation_end_exclusive": str(valid_end.date()),
                "train_games": int(len(train)), "validation_games": int(len(valid)),
                **metrics,
            }
            rows.append(row)
            print(f"  {name:22s} log_loss={metrics['log_loss']:.4f} "
                  f"brier={metrics['brier']:.4f} accuracy={metrics['accuracy']:.2%} "
                  f"({time.perf_counter() - start:.2f}s)", flush=True)

    details = pd.DataFrame(rows)
    summary = details.groupby("model", as_index=False).agg(
        folds=("fold", "count"),
        mean_log_loss=("log_loss", "mean"),
        std_log_loss=("log_loss", "std"),
        mean_brier=("brier", "mean"),
        mean_accuracy=("accuracy", "mean"),
        mean_roc_auc=("roc_auc", "mean"),
    ).sort_values("mean_log_loss")
    print("\n===== WALK-FORWARD SUMMARY (equal weight per fold) =====")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.5f}"))

    print("\n===== POINT DIFFERENTIAL IMPACT (negative delta is better) =====")
    comparison = details.pivot(index="fold", columns="model", values="log_loss")
    for original, enhanced in (
        ("gradient_boosting", "gb_with_point_diff"),
        ("gb_no_team_identity", "gb_no_team_with_point_diff"),
    ):
        delta_pair = comparison[enhanced] - comparison[original]
        print(f"{enhanced} vs {original}: "
              f"mean delta={delta_pair.mean():+.5f}; "
              f"improved {int((delta_pair < 0).sum())}/3 folds")
        for fold_number, change in delta_pair.items():
            print(f"  Fold {fold_number}: {change:+.5f}")

    print("\n===== ELO IMPACT (negative delta is better) =====")
    for enhanced in ("gb_no_team_point_diff_elo", "gb_no_team_point_diff_elo_diff"):
        differences = comparison[enhanced] - comparison["gb_no_team_with_point_diff"]
        print(f"{enhanced}: mean delta={differences.mean():+.5f}; "
              f"improved {int((differences < 0).sum())}/3 folds")
        for fold_number, change in differences.items():
            print(f"  Fold {fold_number}: {change:+.5f}")

    # Paired comparison: same dates and games in each fold for both models.
    paired = details.pivot(index="fold", columns="model", values="log_loss")
    delta = paired["gb_no_team_identity"] - paired["gradient_boosting"]
    print("\nPer-fold log-loss difference (no identity minus original; negative is better):")
    for fold, difference in delta.items():
        print(f"  Fold {fold}: {difference:+.5f}")
    print(f"No-identity wins: {int((delta < 0).sum())} of {len(delta)} folds")
    print("Held-out test period was NOT evaluated.")

    MODELS.mkdir(parents=True, exist_ok=True)
    details.to_csv(MODELS / "walk_forward_folds.csv", index=False)
    summary.to_csv(MODELS / "walk_forward_summary.csv", index=False)
    report = {
        "test_start": str(test_start.date()),
        "summary": summary.to_dict(orient="records"),
        "folds": rows,
        "no_identity_log_loss_delta_by_fold": [float(v) for v in delta],
    }
    (MODELS / "walk_forward_results.json").write_text(json.dumps(report, indent=2))
    print("\nSaved models/walk_forward_folds.csv, models/walk_forward_summary.csv, "
          "and models/walk_forward_results.json")


if __name__ == "__main__":
    main()
