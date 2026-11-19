
import numpy as np
import pandas as pd

HISTORY = "data/processed/player_predictions_2025-26.parquet"
MARKETS = "data/market_lines.csv"

history = pd.read_parquet(HISTORY)
markets = pd.read_csv(MARKETS)

# Historical prediction errors.
residuals = np.sort(
    (history["PTS"] - history["PRED_PTS"])
    .dropna()
    .to_numpy(dtype=float)
)

def american_to_decimal(odds):
    odds = float(odds)
    if odds == 0:
        raise ValueError("American odds cannot be zero")
    if odds > 0:
        return 1 + odds / 100
    return 1 + 100 / abs(odds)

def probability_over(prediction, line):
    threshold = line - prediction
    count_above = len(residuals) - np.searchsorted(
        residuals, threshold, side="right"
    )
    return (count_above + 1) / (len(residuals) + 2)

results = []

for _, bet in markets.iterrows():
    if str(bet["stat"]).upper() != "PTS":
        continue

    side = str(bet["side"]).upper()
    if side not in ("OVER", "UNDER"):
        continue

    prediction = float(bet["prediction"])
    line = float(bet["line"])
    decimal_odds = american_to_decimal(bet["american_odds"])

    p_over = probability_over(prediction, line)
    p_win = p_over if side == "OVER" else 1 - p_over

    implied_probability = 1 / decimal_odds
    ev = p_win * decimal_odds - 1

    results.append({
        "PLAYER": bet["player_name"],
        "BET": f"{side} {line:g} PTS",
        "ODDS": bet["american_odds"],
        "MODEL_PROB": round(p_win, 4),
        "IMPLIED_PROB": round(implied_probability, 4),
        "EDGE": round(p_win - implied_probability, 4),
        "EV": round(ev, 4),
    })

ranked = pd.DataFrame(results)

if ranked.empty:
    print("No valid points markets found.")
else:
    ranked = ranked.sort_values("EV", ascending=False)
    print("\n=== PARLAY GUARD: EV RANKINGS ===")
    print(ranked.to_string(index=False))

    ranked.to_csv(
        "data/recommended_lines.csv",
        index=False
    )
    print("\nSaved: data/recommended_lines.csv")
