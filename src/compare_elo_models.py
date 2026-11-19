
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

DATA = "data/processed/live_games/snapshots.parquet"

df = pd.read_parquet(DATA)

base_features = [
    "CURRENT_MARGIN",
    "CURRENT_TOTAL",
    "ELAPSED_SECONDS",
    "PERIOD",
]

elo_features = base_features + [
    "PREGAME_ELO_DIFF",
    "PREGAME_HOME_WIN_PROB",
]

train = df[df["SEASON"] == "2024-25"].copy()
test = df[df["SEASON"] == "2025-26"].copy()

results = []

for name, features in [
    ("Without Elo", base_features),
    ("With Elo", elo_features),
]:
    model = HistGradientBoostingRegressor(
        max_iter=180,
        max_leaf_nodes=15,
        min_samples_leaf=30,
        learning_rate=0.05,
        l2_regularization=0.2,
        random_state=42,
    )

    model.fit(train[features], train["REMAINING_MARGIN"])

    test["PRED_MARGIN"] = (
        test["CURRENT_MARGIN"] + model.predict(test[features])
    )

    for stage, group in test.groupby("STAGE"):
        mae = mean_absolute_error(
            group["FINAL_MARGIN"],
            group["PRED_MARGIN"],
        )

        trailing = group[
            (
                (group["PREGAME_HOME_WIN_PROB"] >= 0.65)
                & (group["CURRENT_MARGIN"] < 0)
            )
            | (
                (group["PREGAME_HOME_WIN_PROB"] <= 0.35)
                & (group["CURRENT_MARGIN"] > 0)
            )
        ]

        trailing_mae = (
            mean_absolute_error(
                trailing["FINAL_MARGIN"],
                trailing["PRED_MARGIN"],
            )
            if len(trailing) else float("nan")
        )

        results.append({
            "model": name,
            "stage": stage,
            "games": len(group),
            "overall_mae": round(mae, 3),
            "favorite_trailing_mae": round(trailing_mae, 3),
        })

results = pd.DataFrame(results)

print("\n=== ELO ABLATION RESULTS ===")
print(results.to_string(index=False))

print("\n=== WITH ELO MINUS WITHOUT ELO ===")
comparison = results.pivot(
    index="stage",
    columns="model",
    values=["overall_mae", "favorite_trailing_mae"],
)

for metric in ["overall_mae", "favorite_trailing_mae"]:
    print(f"\n{metric}")
    print(
        (
            comparison[metric]["With Elo"]
            - comparison[metric]["Without Elo"]
        ).round(3)
    )

print("\nNegative difference means Elo improved MAE.")
