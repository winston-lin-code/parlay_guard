
import pandas as pd

PATH = "data/processed/player_predictions_2025-26.parquet"

df = pd.read_parquet(PATH)

# Compare actual stats against model predictions.
df["PTS_OVER"] = df["PTS"] > df["PRED_PTS"]
df["FG3M_OVER"] = df["FG3M"] > df["PRED_FG3M"]

# Individual event probabilities
p_pts = df["PTS_OVER"].mean()
p_threes = df["FG3M_OVER"].mean()

# Probability if independent
p_independent = p_pts * p_threes

# Actual historical joint probability
p_joint = (
    df["PTS_OVER"] & df["FG3M_OVER"]
).mean()

# Conditional probability
p_threes_given_pts = df.loc[
    df["PTS_OVER"], "FG3M_OVER"
].mean()

# Residual correlation
correlation = df[
    ["RESIDUAL_PTS", "RESIDUAL_FG3M"]
].corr().iloc[0, 1]

print("=== SAME PLAYER: POINTS + THREES ===")
print("Player-games:", len(df))
print(f"Points above prediction: {p_pts:.2%}")
print(f"Threes above prediction: {p_threes:.2%}")
print(f"Independent joint probability: {p_independent:.2%}")
print(f"Observed joint probability: {p_joint:.2%}")
print(f"Joint difference: {p_joint - p_independent:+.2%}")
print(f"P(Threes over | Points over): {p_threes_given_pts:.2%}")
print(f"Residual correlation: {correlation:.4f}")
