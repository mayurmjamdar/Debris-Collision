import os
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import GroupShuffleSplit
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    precision_score,
    recall_score,
    fbeta_score,
    confusion_matrix,
    classification_report
)


# ============================================================
# 1. LOAD DATA
# ============================================================

print("Loading ESA dataset...")

DATA_PATH = "dataset/train_data.csv"

df = pd.read_csv(DATA_PATH)

print(f"Dataset shape: {df.shape}")


# ============================================================
# 2. SORT DATA
# ============================================================

df = df.sort_values(
    by=["event_id", "time_to_tca"],
    ascending=[True, False]
)

print("Data sorted by event and time_to_tca.")


# ============================================================
# 3. FINAL RISK FOR EACH EVENT
# ============================================================

final_data = (
    df.groupby("event_id", as_index=False)
      .last()
)

final_data = final_data[
    ["event_id", "risk"]
].rename(
    columns={"risk": "final_risk"}
)

print("Final risk created.")


# ============================================================
# 4. SELECT CDMs AVAILABLE >= 2 DAYS BEFORE TCA
# ============================================================

print("Selecting CDMs available >= 2 days before TCA...")

available_data = df[
    df["time_to_tca"] >= 2.0
].copy()


# ============================================================
# 5. LATEST AVAILABLE CDM
# ============================================================

prediction_data = (
    available_data
    .sort_values(
        by=["event_id", "time_to_tca"],
        ascending=[True, True]
    )
    .groupby("event_id", as_index=False)
    .first()
)

print(
    f"Events with usable CDMs: "
    f"{prediction_data['event_id'].nunique()}"
)


# ============================================================
# 6. MERGE FEATURES WITH FINAL RISK
# ============================================================

data = prediction_data.merge(
    final_data,
    on="event_id",
    how="inner"
)

print(
    f"Training events after merge: {len(data)}"
)


# ============================================================
# 7. CREATE HIGH-RISK LABEL
# ============================================================

# risk is stored as log10(probability)
#
# Probability = 1e-6
# log10(1e-6) = -6
#
# Therefore:
# final_risk >= -6  -> HIGH RISK
# final_risk <  -6  -> LOW RISK

HIGH_RISK_THRESHOLD = -6.0

data["high_risk"] = (
    data["final_risk"] >= HIGH_RISK_THRESHOLD
).astype(int)

print()
print("==============================")
print("CLASS DISTRIBUTION")
print("==============================")

print(
    data["high_risk"]
    .value_counts()
    .rename(
        index={
            0: "Low Risk",
            1: "High Risk"
        }
    )
)

print("==============================")


# ============================================================
# 8. CREATE FEATURES
# ============================================================

drop_columns = [
    "event_id",
    "mission_id",
    "risk",
    "final_risk",
    "high_risk"
]

X = data.drop(
    columns=drop_columns,
    errors="ignore"
)

y = data["high_risk"]


# ============================================================
# 9. NUMERIC FEATURES ONLY
# ============================================================

X = X.select_dtypes(
    include=["number"]
)

feature_names = X.columns.tolist()

print(
    f"Number of features: {len(feature_names)}"
)


# ============================================================
# 10. CLEAN DATA
# ============================================================

X = X.astype(np.float64)

X = X.replace(
    [np.inf, -np.inf],
    np.nan
)

MAX_VALUE = 1e20

X = X.mask(
    X.abs() > MAX_VALUE,
    np.nan
)


# ============================================================
# 11. IMPUTATION
# ============================================================

print("Handling missing values...")

imputer = SimpleImputer(
    strategy="median"
)

X = imputer.fit_transform(X)


# ============================================================
# 12. EVENT-BASED SPLIT
# ============================================================

groups = data["event_id"].values

splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=42
)

train_idx, val_idx = next(
    splitter.split(
        X,
        y,
        groups=groups
    )
)

X_train = X[train_idx]
X_val = X[val_idx]

y_train = y.iloc[train_idx]
y_val = y.iloc[val_idx]

print(
    f"Training events: {len(train_idx)}"
)

print(
    f"Validation events: {len(val_idx)}"
)


# ============================================================
# 13. TRAIN CLASSIFIER
# ============================================================

print("Training High-Risk Random Forest Classifier...")

classifier = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    n_jobs=-1,
    class_weight="balanced"
)

classifier.fit(
    X_train,
    y_train
)

print("Classifier trained successfully.")


# ============================================================
# 14. PREDICTIONS
# ============================================================

print("Generating high-risk predictions...")

y_pred = classifier.predict(
    X_val
)


# ============================================================
# 15. EVALUATION
# ============================================================

precision = precision_score(
    y_val,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_val,
    y_pred,
    zero_division=0
)

f2 = fbeta_score(
    y_val,
    y_pred,
    beta=2,
    zero_division=0
)

cm = confusion_matrix(
    y_val,
    y_pred
)


# ============================================================
# 16. RESULTS
# ============================================================

print()
print("==============================")
print("HIGH-RISK CLASSIFIER")
print("==============================")

print(
    f"High-risk threshold : {HIGH_RISK_THRESHOLD}"
)

print(
    f"Actual high-risk events    : {y_val.sum()}"
)

print(
    f"Predicted high-risk events : {y_pred.sum()}"
)

print(
    f"Precision : {precision:.6f}"
)

print(
    f"Recall    : {recall:.6f}"
)

print(
    f"F2 Score  : {f2:.6f}"
)

print()
print("Confusion Matrix:")

print(cm)

print()
print("Classification Report:")

print(
    classification_report(
        y_val,
        y_pred,
        target_names=[
            "Low Risk",
            "High Risk"
        ],
        zero_division=0
    )
)

print("==============================")
print()


# ============================================================
# 17. SAVE CLASSIFIER
# ============================================================

os.makedirs(
    "models",
    exist_ok=True
)

CLASSIFICATION_THRESHOLD = 0.15

model_package = {
    "model": classifier,
    "imputer": imputer,
    "features": feature_names,
    "high_risk_threshold": HIGH_RISK_THRESHOLD,
    "classification_threshold": CLASSIFICATION_THRESHOLD
}

MODEL_PATH = "models/high_risk_classifier.pkl"

joblib.dump(
    model_package,
    MODEL_PATH
)

print(
    f"Classifier saved successfully to: {MODEL_PATH}"
)