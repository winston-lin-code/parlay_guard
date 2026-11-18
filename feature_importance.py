
import joblib
import pandas as pd

model = joblib.load("models/nba_best_candidate.joblib")

preprocessor = model.named_steps["prep"]
classifier = model.named_steps["classifier"]

feature_names = preprocessor.get_feature_names_out()

importance = pd.DataFrame({
    "feature": feature_names,
    "importance": classifier.feature_importances_
})

importance = importance.sort_values(
    "importance",
    ascending=False
)

print("\n===== FEATURE IMPORTANCE =====")
print(importance.to_string(index=False))
