import joblib

MODEL_PATH = "models/high_risk_classifier.pkl"

package = joblib.load(MODEL_PATH)

print()
print("==============================")
print("SAVED CLASSIFIER")
print("==============================")

print("Risk threshold:",
      package["high_risk_threshold"])

print("Classification threshold:",
      package["classification_threshold"])

print("Number of features:",
      len(package["features"]))

print("==============================")