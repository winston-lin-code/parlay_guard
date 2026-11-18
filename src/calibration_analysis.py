"""Out-of-time calibration diagnostics for the chosen NBA model.

Run: python -u -m src.calibration_analysis
No fitting of calibrators or use of the held-out final test period.
"""
import json
import time

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve

from .common import DATA, MODELS
from .walk_forward import make_model, score

MODEL_NAME = "gb_no_team_with_point_diff"
N_BINS = 10


def bin_statistics(frame, bins=N_BINS):
    edges = np.linspace(0, 1, bins + 1)
    group = pd.cut(frame["predicted_probability"], bins=edges, include_lowest=True)
    summary = frame.groupby(group, observed=False).agg(
        games=("actual_home_win", "size"),
        mean_predicted=("predicted_probability", "mean"),
        observed_win_rate=("actual_home_win", "mean"),
    ).reset_index(drop=True)
    summary.insert(0, "probability_bin", [f"{edges[i]:.1f}-{edges[i+1]:.1f}" for i in range(bins)])
    summary["absolute_gap"] = (summary["mean_predicted"] - summary["observed_win_rate"]).abs()
    return summary


def main():
    path = DATA / "nba_model_dataset.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Rebuild your feature dataset first.")
    df = pd.read_csv(path, dtype={"GAME_ID": str})
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    df = df.sort_values(["GAME_DATE", "GAME_ID"]).reset_index(drop=True)
    dates = np.array(sorted(df["GAME_DATE"].unique()))
    if len(dates) < 30:
        raise ValueError("Insufficient game dates")
    boundaries = [int(len(dates) * fraction) for fraction in (0.40, 0.55, 0.70, 0.85)]
    if len(set(boundaries)) != 4:
        raise ValueError("Overlapping fold boundaries")
    test_start = pd.Timestamp(dates[boundaries[-1]])
    print(f"Final held-out test starts {test_start.date()} (not evaluated)", flush=True)

    rows = []
    fold_metrics = []
    for fold in range(3):
        start_date = pd.Timestamp(dates[boundaries[fold]])
        end_date = pd.Timestamp(dates[boundaries[fold + 1]])
        train = df[df["GAME_DATE"] < start_date]
        valid = df[(df["GAME_DATE"] >= start_date) & (df["GAME_DATE"] < end_date)]
        model, inputs = make_model(include_team_identity=False, include_point_diff=True)
        if any(column not in df.columns for column in inputs):
            raise ValueError("Missing model feature columns. Rebuild features first.")
        t0 = time.perf_counter()
        model.fit(train[inputs], train["HOME_WIN"].astype(int))
        probs = model.predict_proba(valid[inputs])[:, 1]
        actual = valid["HOME_WIN"].astype(int).to_numpy()
        metrics = score(actual, probs)
        fold_metrics.append({"fold": fold + 1, "games": len(valid), **metrics})
        for game_id, date, y, p in zip(valid["GAME_ID"], valid["GAME_DATE"], actual, probs):
            rows.append({
                "fold": fold + 1,
                "GAME_ID": game_id,
                "GAME_DATE": str(date.date()),
                "actual_home_win": int(y),
                "predicted_probability": float(p),
            })
        print(f"Fold {fold+1}/3: {len(valid)} games | log loss {metrics['log_loss']:.4f} "
              f"| brier {metrics['brier']:.4f} | {time.perf_counter()-t0:.2f}s", flush=True)

    predictions = pd.DataFrame(rows)
    if predictions["GAME_ID"].duplicated().any():
        raise ValueError("A game was evaluated in more than one validation fold")
    bins = bin_statistics(predictions)
    nonempty = bins[bins["games"] > 0].copy()
    ece = float((nonempty["games"] * nonempty["absolute_gap"]).sum() / len(predictions))
    overall = score(predictions["actual_home_win"], predictions["predicted_probability"])
    print("\n===== POOLED OUT-OF-TIME CALIBRATION =====")
    print(f"Games: {len(predictions)} | Brier: {overall['brier']:.5f} "
          f"| log loss: {overall['log_loss']:.5f} | ECE (10 bins): {ece:.5f}")
    print("\n===== RELIABILITY TABLE =====")
    print(bins.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print("\nNOTE: ECE depends on bin choice. Sparse bins are noisy; inspect counts.")
    print("These are raw probabilities; no calibrator was fitted on validation outcomes.")

    MODELS.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(MODELS / "calibration_predictions.csv", index=False)
    bins.to_csv(MODELS / "calibration_bins.csv", index=False)
    pd.DataFrame(fold_metrics).to_csv(MODELS / "calibration_folds.csv", index=False)
    (MODELS / "calibration_report.json").write_text(json.dumps({
        "model": MODEL_NAME,
        "held_out_test_start": str(test_start.date()),
        "pooled_validation_games": len(predictions),
        "pooled_metrics": overall,
        "ece_10_bins": ece,
        "fold_metrics": fold_metrics,
    }, indent=2))
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.plot([0, 1], [0, 1], "--", color="gray", label="Perfect calibration")
        ax.plot(nonempty["mean_predicted"], nonempty["observed_win_rate"],
                "o-", label=f"{MODEL_NAME} (out-of-time)")
        for _, row in nonempty.iterrows():
            ax.annotate(str(int(row["games"])),
                        (row["mean_predicted"], row["observed_win_rate"]),
                        xytext=(4, 4), textcoords="offset points", fontsize=8)
        ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="Mean predicted home-win probability",
               ylabel="Observed home-win frequency", title="NBA model reliability (numbers = bin game counts)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(MODELS / "calibration_reliability.png", dpi=170)
        plt.close(fig)
        print("\nSaved reliability plot: models/calibration_reliability.png")
    except ImportError:
        print("\nOptional: pip install matplotlib to save a reliability plot")
    print("Saved predictions, bins, fold metrics and report to models/.")
    print("Final held-out test period was NOT evaluated.")


if __name__ == "__main__":
    main()
