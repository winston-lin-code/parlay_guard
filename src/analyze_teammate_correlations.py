
import pandas as pd

PATH = "data/processed/player_predictions_2025-26.parquet"

df = pd.read_parquet(PATH)

# Join each player with teammates from the same game.
pairs = df.merge(
    df,
    on=["GAME_ID", "TEAM_ID"],
    suffixes=("_SCORER", "_PASSER")
)

# Exclude pairing a player with themselves.
pairs = pairs[
    pairs["PLAYER_ID_SCORER"] != pairs["PLAYER_ID_PASSER"]
].copy()
pairs = pairs[
    (pairs["PTS_LAST_10_SCORER"] >= 20) &
    (pairs["AST_LAST_10_PASSER"] >= 6)
].copy()
# Did each player exceed their model prediction?
pairs["SCORER_OVER"] = (
    pairs["PTS_SCORER"] > pairs["PRED_PTS_SCORER"]
)

pairs["PASSER_OVER"] = (
    pairs["AST_PASSER"] > pairs["PRED_AST_PASSER"]
)

p_scorer = pairs["SCORER_OVER"].mean()
p_passer = pairs["PASSER_OVER"].mean()

p_independent = p_scorer * p_passer

p_joint = (
    pairs["SCORER_OVER"] & pairs["PASSER_OVER"]
).mean()

p_conditional = pairs.loc[
    pairs["PASSER_OVER"], "SCORER_OVER"
].mean()

correlation = pairs[
    ["RESIDUAL_PTS_SCORER", "RESIDUAL_AST_PASSER"]
].corr().iloc[0, 1]

print("=== TEAMMATE POINTS + ASSISTS ===")
print("Player pairs:", len(pairs))
print(f"Scorer above prediction: {p_scorer:.2%}")
print(f"Passer above prediction: {p_passer:.2%}")
print(f"Independent joint rate: {p_independent:.2%}")
print(f"Observed joint rate: {p_joint:.2%}")
print(f"Joint difference: {p_joint - p_independent:+.2%}")
print(f"P(Scorer over | Passer over): {p_conditional:.2%}")
print(f"Residual correlation: {correlation:.4f}")
