import pandas as pd

# Load the model-ready dataset
df = pd.read_csv(
    "data/nba_model_dataset.csv",
    parse_dates=["GAME_DATE"]
)

# Use the same test period as your trained model
test = df[df["GAME_DATE"] >= "2025-12-29"].copy()

# The baseline always predicts a home-team victory
test["BASELINE_PREDICTION"] = 1

# Calculate accuracy
baseline_accuracy = (
    test["BASELINE_PREDICTION"]
    == test["HOME_WIN"].astype(int)
).mean()

model_accuracy = 0.5949868073878628

print("Test games:", len(test))
print(f"Home-team baseline accuracy: {baseline_accuracy:.2%}")
print(f"Your model accuracy:         {model_accuracy:.2%}")
print(f"Difference:                  {model_accuracy - baseline_accuracy:+.2%}")

if model_accuracy > baseline_accuracy:
    print("\nYour model beats the home-team baseline.")
elif model_accuracy < baseline_accuracy:
    print("\nYour model underperforms the home-team baseline.")
else:
    print("\nYour model matches the home-team baseline.")