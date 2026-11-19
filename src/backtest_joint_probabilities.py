
import pandas as pd
import numpy as np

PATH = "data/processed/player_predictions_2025-26.parquet"

df = pd.read_parquet(PATH)
df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])

pairs = df.merge(
    df,
    on=["GAME_ID", "TEAM_ID", "GAME_DATE"],
    suffixes=("_SCORER", "_PASSER")
)
pairs = pairs[
    (pairs["PLAYER_ID_SCORER"] != pairs["PLAYER_ID_PASSER"]) &
    (pairs["PTS_LAST_10_SCORER"] >= 20) &
    (pairs["AST_LAST_10_PASSER"] >= 6)
].copy()

pairs["SCORER_OVER"] = (
    pairs["PTS_SCORER"] > pairs["PRED_PTS_SCORER"]
)

pairs["PASSER_OVER"] = (
    pairs["AST_PASSER"] > pairs["PRED_AST_PASSER"]
)

pairs["BOTH_HIT"] = (
    pairs["SCORER_OVER"] & pairs["PASSER_OVER"]
).astype(int)

# Split by unique game dates to avoid splitting a game.
dates = np.sort(pairs["GAME_DATE"].unique())
cutoff = dates[int(len(dates) * 0.70)]

train = pairs[pairs["GAME_DATE"] < cutoff].copy()
test = pairs[pairs["GAME_DATE"] >= cutoff].copy()

# Model A: independent event rates
p_scorer = train["SCORER_OVER"].mean()
p_passer = train["PASSER_OVER"].mean()

p_independent = p_scorer * p_passer

# Model B: historical joint rate
p_correlated = train["BOTH_HIT"].mean()

actual = test["BOTH_HIT"].mean()

def brier_score(y, probability):
    return np.mean((y - probability) ** 2)

print("=== JOINT PROBABILITY BACKTEST ===")
print("Training pairs:", len(train))
print("Testing pairs:", len(test))

print(f"\nIndependent prediction: {p_independent:.2%}")
print(f"Correlation-aware prediction: {p_correlated:.2%}")
print(f"Actual test joint rate: {actual:.2%}")

print("\nBrier scores (lower is better):")
print(
    "Independent:",
    round(brier_score(test["BOTH_HIT"], p_independent), 5)
)
print(
    "Correlation-aware:",
    round(brier_score(test["BOTH_HIT"], p_correlated), 5)
)
