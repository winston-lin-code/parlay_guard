"""ONE-TIME final holdout evaluation for the selected Parlay Guard Elo model.

Run from project root:
    python -u -m src.final_validation

Locked model selection (from three development walk-forward folds):
    make_model(include_team_identity=False,
               include_point_diff=True,
               elo_mode="difference")
    Uncalibrated probabilities.

Train: every game strictly before 2025-12-29.
Test:  every game from 2025-12-29 onward (expected 758 games in the
       frozen 4,920-game historical dataset).

IMPORTANT: Once this holdout is inspected, it is no longer a fresh test set.
Do not tune models, thresholds or calibrators on the results and call another
run an independent final validation.
"""

import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import confusion_matrix

from .common import DATA, MODELS
from .elo import HOME_ADVANTAGE
from .walk_forward import make_model, score

HOLDOUT_START = pd.Timestamp("2025-12-29")
EXPECTED_TOTAL_GAMES = 4920
EXPECTED_HOLDOUT_GAMES = 758
CALIBRATION_BINS = 10
BOOTSTRAP_REPLICATES = 2000
BOOTSTRAP_SEED = 42


def read_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run python -u -m src.pipeline first.")
    df = pd.read_csv(path, dtype={"GAME_ID": str})
    required = {"GAME_ID", "GAME_DATE", "HOME_TEAM", "AWAY_TEAM", "HOME_WIN", "ELO_DIFF"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing columns {sorted(missing)}; rebuild model dataset first.")
    if df[list(required)].isna().any().any():
        raise ValueError("Null values in required columns; inspect the model dataset.")
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"], errors="raise").dt.normalize()
    df["HOME_WIN"] = pd.to_numeric(df["HOME_WIN"], errors="raise")
    if not df["HOME_WIN"].isin([0, 1]).all():
        raise ValueError("HOME_WIN must contain only 0/1.")
    if df["GAME_ID"].duplicated().any():
        raise ValueError("Duplicate GAME_ID values; expected one row per NBA game.")
    if (df["HOME_TEAM"] == df["AWAY_TEAM"]).any():
        raise ValueError("Some games have the same home and away team.")
    return df.sort_values(["GAME_DATE", "GAME_ID"]).reset_index(drop=True)


def validate_frozen_split(df: pd.DataFrame):
    """Ensure the holdout is the exact test set we intentionally reserved."""
    dates = np.array(sorted(df["GAME_DATE"].unique()))
    if len(dates) < 30:
        raise ValueError("Too few unique dates for the original date-based split.")
    original_cutoff = pd.Timestamp(dates[int(len(dates) * 0.85)])
    if original_cutoff != HOLDOUT_START:
        raise ValueError(
            "Historical 85%-of-dates cutoff changed: "
            f"{original_cutoff.date()}, expected {HOLDOUT_START.date()}. "
            "Use the same frozen nba_model_dataset.csv as the walk-forward runs."
        )
    train = df.loc[df["GAME_DATE"] < HOLDOUT_START].copy()
    test = df.loc[df["GAME_DATE"] >= HOLDOUT_START].copy()
    if len(df) != EXPECTED_TOTAL_GAMES or len(test) != EXPECTED_HOLDOUT_GAMES:
        raise ValueError(
            f"Expected {EXPECTED_TOTAL_GAMES:,} total and "
            f"{EXPECTED_HOLDOUT_GAMES:,} held-out games, got "
            f"{len(df):,} total and {len(test):,} held-out. "
            "The dataset differs from the frozen development dataset."
        )
    if train.empty or test.empty or train["HOME_WIN"].nunique() != 2:
        raise ValueError("Invalid train/test split or only one class in training.")
    if set(train.GAME_ID) & set(test.GAME_ID):
        raise AssertionError("Train/test share GAME_ID values.")
    if train["GAME_DATE"].max() >= test["GAME_DATE"].min():
        raise AssertionError("Train/test dates overlap.")
    return train, test


def calibration_table(y, p, bins=10) -> pd.DataFrame:
    """Equal-width bins with standard observed and predicted win frequencies."""
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    bin_index = np.minimum((p * bins).astype(int), bins - 1)
    result = []
    for b in range(bins):
        mask = bin_index == b
        count = int(mask.sum())
        result.append({
            "bin": b + 1,
            "lower_inclusive": b / bins,
            "upper_exclusive_except_last": (b + 1) / bins,
            "games": count,
            "avg_predicted_home_win": float(p[mask].mean()) if count else np.nan,
            "observed_home_win_rate": float(y[mask].mean()) if count else np.nan,
            "absolute_gap": (float(abs(y[mask].mean() - p[mask].mean()))
                             if count else np.nan),
        })
    return pd.DataFrame(result)


def ece_from_calibration(table: pd.DataFrame) -> float:
    valid = table.loc[table["games"] > 0]
    return float(np.sum(valid["games"] * valid["absolute_gap"]) / table["games"].sum())


def bootstrap_intervals(y, predictions: dict, reps=BOOTSTRAP_REPLICATES):
    """Approximate paired 95% intervals, resampling held-out games.

    These describe uncertainty conditional on the fitted model; games are not
    perfectly independent (shared teams/dates), so do not interpret as exact
    significance tests.
    """
    y = np.asarray(y, dtype=float)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    indices = rng.integers(0, len(y), size=(reps, len(y)))
    log_loss_rows = {}
    brier_rows = {}
    accuracy_rows = {}
    for name, raw_p in predictions.items():
        p = np.clip(np.asarray(raw_p, dtype=float), 1e-7, 1 - 1e-7)
        row_ll = -(y * np.log(p) + (1 - y) * np.log1p(-p))
        row_brier = (y - p) ** 2
        row_accuracy = ((p >= 0.5) == y).astype(float)
        log_loss_rows[name] = row_ll
        brier_rows[name] = row_brier
        accuracy_rows[name] = row_accuracy

    def interval(values):
        return [float(v) for v in np.quantile(values, [0.025, 0.975])]

    intervals = {}
    for name in predictions:
        intervals[name] = {
            "log_loss_95pct_approx": interval(log_loss_rows[name][indices].mean(axis=1)),
            "brier_95pct_approx": interval(brier_rows[name][indices].mean(axis=1)),
            "accuracy_95pct_approx": interval(accuracy_rows[name][indices].mean(axis=1)),
        }
    paired = {}
    for baseline in ("historical_home_rate", "elo_formula"):
        delta = log_loss_rows["selected_elo_gb"] - log_loss_rows[baseline]
        paired[baseline] = {
            "mean_log_loss_delta": float(delta.mean()),
            "approx_95pct_interval": interval(delta[indices].mean(axis=1)),
        }
    return intervals, paired


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Recreate output files from the same frozen model/dataset. "
             "Do not use after choosing changes based on this test.",
    )
    args = parser.parse_args()
    model_path = MODELS / "elo_final_holdout_model.joblib"
    report_path = MODELS / "elo_final_holdout_report.json"
    predictions_path = MODELS / "elo_final_holdout_predictions.csv"
    calibration_path = MODELS / "elo_final_holdout_calibration.csv"
    comparison_path = MODELS / "elo_final_holdout_comparison.csv"
    if report_path.exists() and not args.overwrite:
        raise SystemExit(
            f"Final holdout already evaluated: {report_path}\n"
            "Inspect the existing report. If reproducing the EXACT same "
            "experiment, use --overwrite. Do not tune based on the holdout."
        )

    data_path = DATA / "nba_model_dataset.csv"
    df = read_data(data_path)
    train, test = validate_frozen_split(df)
    model, inputs = make_model(
        include_team_identity=False,
        include_point_diff=True,
        elo_mode="difference",
    )
    missing = sorted(set(inputs).difference(df.columns))
    if missing:
        raise ValueError(f"Model input features missing: {missing}. Rebuild features first.")
    print("===== PARLAY GUARD — ONE-TIME FINAL HOLDOUT =====", flush=True)
    print(f"Dataset: {len(df):,} NBA games across {df.GAME_DATE.nunique():,} dates", flush=True)
    print(f"Training: {len(train):,} games before {HOLDOUT_START.date()}", flush=True)
    print(f"Holdout: {len(test):,} games from {HOLDOUT_START.date()} "
          f"through {test.GAME_DATE.max().date()}", flush=True)
    print("Model: GradientBoostingClassifier(n_estimators=100, "
          "learning_rate=0.05, max_depth=2, random_state=42)", flush=True)
    print("Configuration: no team identity, point differential included, "
          "Elo difference, RAW probabilities (no calibration)", flush=True)
    print(f"Features ({len(inputs)}): {', '.join(inputs)}", flush=True)
    print(f"Training home-win rate: {train.HOME_WIN.mean():.2%}", flush=True)
    print("Fitting once on all pre-holdout games...", flush=True)

    model.fit(train[inputs], train["HOME_WIN"].astype(int))
    y = test["HOME_WIN"].astype(int).to_numpy()
    selected_probs = model.predict_proba(test[inputs])[:, 1]
    prior_rate = float(train["HOME_WIN"].mean())
    constant_probs = np.full(len(test), prior_rate, dtype=float)
    # Pure NBA Elo probability baseline, with the game's PRE-game Elo rating
    # difference and the same fixed 65-point home advantage in src.elo.
    elo_formula_probs = 1 / (1 + 10 ** (-(
        test["ELO_DIFF"].to_numpy(dtype=float) + HOME_ADVANTAGE) / 400))
    probabilities = {
        "historical_home_rate": constant_probs,
        "elo_formula": elo_formula_probs,
        "selected_elo_gb": selected_probs,
    }
    comparison = []
    for name, p in probabilities.items():
        m = score(y, p)
        cal = calibration_table(y, p, bins=CALIBRATION_BINS)
        comparison.append({"model": name, "games": len(test), **m,
                           "ece_10_bins": ece_from_calibration(cal)})
    comparison_df = pd.DataFrame(comparison).sort_values("log_loss")
    print("\n===== FINAL HOLDOUT COMPARISON =====", flush=True)
    print(comparison_df.to_string(index=False, float_format=lambda v: f"{v:.5f}"), flush=True)

    intervals, paired = bootstrap_intervals(y, probabilities)
    print("\n===== APPROXIMATE 95% BOOTSTRAP INTERVALS =====", flush=True)
    selected_interval = intervals["selected_elo_gb"]
    print("  Selected Elo GB log loss: "
          f"[{selected_interval['log_loss_95pct_approx'][0]:.5f}, "
          f"{selected_interval['log_loss_95pct_approx'][1]:.5f}]", flush=True)
    for baseline, result in paired.items():
        low, high = result["approx_95pct_interval"]
        print(f"  Selected minus {baseline} log loss (negative is better): "
              f"{result['mean_log_loss_delta']:+.5f} "
              f"(approx 95% interval [{low:+.5f}, {high:+.5f}])", flush=True)
    print("  Caution: resampling treats games as independent; approximate only.", flush=True)

    calibration = calibration_table(y, selected_probs, CALIBRATION_BINS)
    print("\n===== SELECTED MODEL RELIABILITY TABLE =====", flush=True)
    print(calibration.to_string(index=False, float_format=lambda v: f"{v:.4f}"), flush=True)

    confusion = confusion_matrix(y, selected_probs >= 0.5, labels=[0, 1])
    print("\n===== SELECTED MODEL CONFUSION MATRIX =====", flush=True)
    print("Rows: actual [away win, home win]; columns: predicted [away win, home win]", flush=True)
    print(confusion, flush=True)

    out = test[["GAME_ID", "GAME_DATE", "HOME_TEAM", "AWAY_TEAM",
                "HOME_WIN", "ELO_DIFF"]].copy()
    out["GAME_DATE"] = out["GAME_DATE"].dt.strftime("%Y-%m-%d")
    out["p_home_win_selected_elo_gb"] = selected_probs
    out["p_home_win_elo_formula"] = elo_formula_probs
    out["p_home_win_historical_rate"] = constant_probs
    out["predicted_home_win"] = (selected_probs >= 0.5).astype(int)
    out["correct"] = (out["predicted_home_win"] == y).astype(int)

    report = {
        "experiment": "One-time final holdout, selected Elo gradient boosting, raw probabilities",
        "train_end_exclusive": str(HOLDOUT_START.date()),
        "test_start_inclusive": str(HOLDOUT_START.date()),
        "test_end_inclusive": str(test.GAME_DATE.max().date()),
        "games_total": int(len(df)),
        "train_games": int(len(train)),
        "test_games": int(len(test)),
        "train_home_win_rate": prior_rate,
        "test_home_win_rate": float(y.mean()),
        "data_file_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
        "selected_model": {
            "feature_columns": inputs,
            "model_params": model.named_steps["classifier"].get_params(),
            "calibration": "none; raw",
        },
        "summary": comparison_df.to_dict(orient="records"),
        "approx_bootstrap_95pct_intervals": intervals,
        "paired_log_loss_comparisons": paired,
        "confusion_matrix_away_home": confusion.tolist(),
        "package_versions": {
            "numpy": np.__version__, "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "notes": [
            "Selected from pre-holdout walk-forward validation; no further tuning on this test.",
            "Historical held-out games are evaluated sequentially using pregame features. "
            "Pregame Elo and rolling stats for a later test game may use earlier test-game results, "
            "but the classifier is fitted ONLY on pre-2025-12-29 examples.",
            "10-bin ECE depends on bin choices and may be noisy with 758 games.",
            "Bootstrap uncertainty treats games as independent; approximate only.",
        ],
    }
    MODELS.mkdir(parents=True, exist_ok=True)
    out.to_csv(predictions_path, index=False)
    calibration.to_csv(calibration_path, index=False)
    comparison_df.to_csv(comparison_path, index=False)
    report_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    joblib.dump({
        "model": model, "inputs": inputs,
        "training_end_exclusive": str(HOLDOUT_START.date()),
        "calibration": "none",
        "data_file_sha256": report["data_file_sha256"],
        "holdout_report_path": str(report_path),
    }, model_path)
    print("\nSaved:", flush=True)
    for path in (report_path, comparison_path, calibration_path,
                 predictions_path, model_path):
        print(f"  {path}", flush=True)
    print("\nFINAL HOLDOUT HAS NOW BEEN USED. Do NOT tune on these results "
          "and call it an untouched test again.", flush=True)


if __name__ == "__main__":
    main()
