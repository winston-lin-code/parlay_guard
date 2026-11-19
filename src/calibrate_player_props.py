"""Chronological, leakage-aware baseline calibration of NBA points prop probabilities.

Input: data/processed/player_predictions_2025-26.parquet
Output: data/processed/player_points_probability_backtest.parquet

Uses earlier game dates to estimate empirical residual CDFs, evaluates later dates.
Synthetic half-point lines are for calibration diagnostics only, not betting EV.
"""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss


def empirical_over_probability(residuals: np.ndarray, thresholds: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    """P(residual > line - prediction), smoothed to avoid 0 and 1."""
    sorted_res = np.sort(np.asarray(residuals, dtype=float))
    if len(sorted_res) == 0:
        raise ValueError("Cannot calibrate using zero residuals")
    count_above = len(sorted_res) - np.searchsorted(sorted_res, thresholds, side="right")
    return (count_above + alpha) / (len(sorted_res) + 2 * alpha)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/processed/player_predictions_2025-26.parquet")
    parser.add_argument("--output", default="data/processed/player_points_probability_backtest.parquet")
    parser.add_argument("--train-fraction", type=float, default=0.7)
    args = parser.parse_args()
    if not 0 < args.train_fraction < 1:
        parser.error("--train-fraction must be between 0 and 1")

    df = pd.read_parquet(args.input)
    required = ["GAME_ID", "GAME_DATE", "PLAYER_ID", "PLAYER_NAME", "PTS", "PRED_PTS"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    df = df.dropna(subset=required).copy()
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    df = df.sort_values(["GAME_DATE", "GAME_ID", "PLAYER_ID"]).reset_index(drop=True)
    if df.duplicated(["GAME_ID", "PLAYER_ID"]).any():
        raise ValueError("Duplicate GAME_ID / PLAYER_ID records")
    dates = np.sort(df["GAME_DATE"].unique())
    if len(dates) < 3:
        raise ValueError("At least three distinct dates required")
    cutoff = dates[int(len(dates) * args.train_fraction)]
    train = df[df.GAME_DATE < cutoff].copy()
    test = df[df.GAME_DATE >= cutoff].copy()
    residuals = (train.PTS - train.PRED_PTS).to_numpy(dtype=float)

    # A half-point threshold close to the predicted mean, NOT a sportsbook line.
    # Use the same rule for calibration and evaluation. The projection itself is
    # based on a model trained on seasons before 2025-26.

# Generate multiple synthetic lines per player-game.
# These are NOT actual sportsbook lines.
    offsets = [-5, -3, -1, 0, 1, 3, 5]

    test = pd.concat(
        [
            test.assign(
                    OFFSET = offset,
                LINE=np.maximum(
                    0.5,
                    np.floor(test["PRED_PTS"] + offset) + 0.5
                )
            )
            for offset in offsets
        ],
        ignore_index=True
    )

    # Remove duplicate lines, which can occur near zero.
    test = test.drop_duplicates(
        subset=["GAME_ID", "PLAYER_ID", "LINE"]
    ).copy()

    test["OVER_HIT"] = (test["PTS"] > test["LINE"]).astype(int)

    test["P_OVER"] = empirical_over_probability(
        residuals, (test.LINE - test.PRED_PTS).to_numpy(dtype=float)
    )
    test["P_UNDER"] = 1 - test.P_OVER
    baseline_by_offset = {}

    for offset in offsets:
        train_line = np.maximum(
            
            0.5,
            np.floor(train["PRED_PTS"] + offset) + 0.5
        )

        baseline_by_offset[offset] = (
            train["PTS"] > train_line
        ).mean()

    test["BASELINE_P_OVER"] = test["OFFSET"].map(
        baseline_by_offset
)
    print("\n=== BASELINE BY OFFSET ===")

    for offset in offsets:
        print(
            f"Offset {offset:+3}: "
            f"{baseline_by_offset[offset]:.2%}"
        )

    print("\n=== BRIER BY OFFSET ===")

    for offset, group in test.groupby("OFFSET"):
        print(
            f"Offset {offset:+3}: "
            f"Calibrated={brier_score_loss(group['OVER_HIT'], group['P_OVER']):.5f} | "
            f"Baseline={brier_score_loss(group['OVER_HIT'], group['BASELINE_P_OVER']):.5f}"
        )
    # test["BASELINE_P_OVER"] = (train.PTS > (np.floor(train.PRED_PTS) + 0.5)).mean()

    y = test.OVER_HIT.to_numpy()
    p = test.P_OVER.to_numpy()
    baseline = test.BASELINE_P_OVER.to_numpy()
    print("=== POINTS PROBABILITY CALIBRATION ===")
    print(f"Calibration rows: {len(train):,}")
    print(f"Evaluation rows:  {len(test):,}")
    print(f"Cutoff date:      {pd.Timestamp(cutoff).date()}")
    print(f"Line source:      SYNTHETIC (offsets={offsets})")
    print(f"Actual over rate: {y.mean():.2%}")
    print(f"Mean predicted:   {p.mean():.2%}")
    print(f"Brier calibrated: {brier_score_loss(y, p):.5f}")
    print(f"Brier constant:   {brier_score_loss(y, baseline):.5f}")
    print(f"Log loss:         {log_loss(y, p, labels=[0, 1]):.5f}")
    test["BIN"] = pd.cut(test.P_OVER, bins=np.linspace(0, 1, 11), include_lowest=True)
    summary = test.groupby("BIN", observed=True).agg(
        count=("OVER_HIT", "size"),
        mean_prediction=("P_OVER", "mean"),
        observed_rate=("OVER_HIT", "mean"),
    )
    print("\n=== CALIBRATION BY PROBABILITY BIN ===")
    print(summary.to_string(float_format=lambda x: f"{x:.3f}"))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    test.drop(columns=["BIN"]).to_parquet(output, index=False)
    print(f"\nSaved: {output}")
    print("Note: No sportsbook prices or real betting lines were used; this is NOT an EV backtest.")


if __name__ == "__main__":
    main()
