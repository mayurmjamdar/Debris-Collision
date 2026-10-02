import joblib
import pandas as pd

MODEL_PATH = "models/collision_model.pkl"

data = joblib.load(MODEL_PATH)

model = data["model"]
features = data["features"]

importance = pd.DataFrame({
    "feature": features,
    "importance": model.feature_importances_
})

importance = importance.sort_values(
    by="importance",
    ascending=False
)

print()
print("==============================")
print("TOP 30 FEATURE IMPORTANCE")
print("==============================")
print()

print(
    importance.head(30).to_string(index=False)
)