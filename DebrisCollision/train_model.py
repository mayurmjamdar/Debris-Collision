import os
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import GroupShuffleSplit
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ============================================================
# 1. LOAD DATA
# ============================================================

print("Loading ESA dataset...")

DATA_PATH = "dataset/train_data.csv"

df = pd.read_csv(DATA_PATH)

print(f"Dataset shape: {df.shape}")


# ============================================================
# 2. SORT BY EVENT AND TIME TO TCA
# ============================================================

df = df.sort_values(
    by=["event_id", "time_to_tca"],
    ascending=[True, False]
)

print("Data sorted by event and time_to_tca.")


# ============================================================
# 3. CREATE FINAL TARGET
# ============================================================

# The final CDM is the row closest to TCA.
# Since time_to_tca decreases with time,
# the last row of each event is the final CDM.

final_data = (
    df.groupby("event_id", as_index=False)
      .last()
)

final_data = final_data[
    ["event_id", "risk"]
].rename(
    columns={"risk": "final_risk"}
)

print("Final risk target created.")


# ============================================================
# 4. SELECT CDMs AVAILABLE AT LEAST 2 DAYS BEFORE TCA
# ============================================================

print("Selecting CDMs available >= 2 days before TCA...")

available_data = df[
    df["time_to_tca"] >= 2.0
].copy()


# ============================================================
# 5. FIND LATEST AVAILABLE CDM FOR EACH EVENT
# ============================================================

# Among rows >= 2 days from TCA,
# the smallest time_to_tca is the latest available CDM.

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
    f"Events with usable >=2-day CDM: "
    f"{prediction_data['event_id'].nunique()}"
)


# ============================================================
# 6. MERGE FEATURES WITH FINAL TARGET
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
# 7. CREATE X AND y
# ============================================================

drop_columns = [
    "event_id",
    "mission_id",
    "risk",
    "final_risk"
]

X = data.drop(
    columns=drop_columns,
    errors="ignore"
)

y = data["final_risk"]


# ============================================================
# 8. KEEP NUMERIC FEATURES
# ============================================================

X = X.select_dtypes(
    include=["number"]
)

feature_names = X.columns.tolist()

print(
    f"Number of features: {len(feature_names)}"
)


# ============================================================
# 9. CONVERT TO FLOAT64
# ============================================================

X = X.astype(np.float64)


# ============================================================
# 10. CLEAN INVALID VALUES
# ============================================================

print("Cleaning invalid numeric values...")

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
# 11. REMOVE COMPLETELY EMPTY COLUMNS
# ============================================================

empty_columns = X.columns[
    X.isna().all()
].tolist()

if empty_columns:

    print(
        f"Removing {len(empty_columns)} empty columns."
    )

    X = X.drop(
        columns=empty_columns
    )

    feature_names = X.columns.tolist()


# ============================================================
# 12. HANDLE MISSING VALUES
# ============================================================

print("Handling missing values...")

imputer = SimpleImputer(
    strategy="median"
)

X = imputer.fit_transform(X)


# ============================================================
# 13. FINAL DATA CHECK
# ============================================================

print("Checking feature matrix...")

if not np.isfinite(X).all():

    raise ValueError(
        "Feature matrix contains invalid values."
    )

print("Feature matrix is clean.")


# ============================================================
# 14. EVENT-BASED TRAIN / VALIDATION SPLIT
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
# 15. TRAIN RANDOM FOREST
# ============================================================

print("Training Random Forest...")

model = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

model.fit(
    X_train,
    y_train
)

print("Model trained successfully.")


# ============================================================
# 16. PREDICT
# ============================================================

print("Generating predictions...")

y_pred = model.predict(
    X_val
)

# ============================================================
# HIGH-RISK EVENT EVALUATION
# ============================================================

from sklearn.metrics import (
    precision_score,
    recall_score,
    fbeta_score,
    confusion_matrix
)

HIGH_RISK_THRESHOLD = -6.0

y_val_high = (
    y_val >= HIGH_RISK_THRESHOLD
).astype(int)

y_pred_high = (
    y_pred >= HIGH_RISK_THRESHOLD
).astype(int)

precision = precision_score(
    y_val_high,
    y_pred_high,
    zero_division=0
)

recall = recall_score(
    y_val_high,
    y_pred_high,
    zero_division=0
)

f2 = fbeta_score(
    y_val_high,
    y_pred_high,
    beta=2,
    zero_division=0
)

cm = confusion_matrix(
    y_val_high,
    y_pred_high
)

print()
print("==============================")
print("HIGH-RISK EVENT EVALUATION")
print("==============================")

print(
    f"High-risk threshold : {HIGH_RISK_THRESHOLD}"
)

print(
    f"Actual high-risk events    : {y_val_high.sum()}"
)

print(
    f"Predicted high-risk events : {y_pred_high.sum()}"
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

print("==============================")
print()

# ============================================================
# 17. EVALUATE
# ============================================================

mae = mean_absolute_error(
    y_val,
    y_pred
)

rmse = np.sqrt(
    mean_squared_error(
        y_val,
        y_pred
    )
)

r2 = r2_score(
    y_val,
    y_pred
)


print()
print("==============================")
print("MODEL PERFORMANCE")
print("==============================")

print(f"MAE  : {mae:.6f}")
print(f"RMSE : {rmse:.6f}")
print(f"R2   : {r2:.6f}")

print("==============================")
print()


# ============================================================
# 18. SAVE MODEL
# ============================================================

os.makedirs(
    "models",
    exist_ok=True
)

model_package = {
    "model": model,
    "imputer": imputer,
    "features": feature_names
}

MODEL_PATH = "models/collision_model.pkl"

joblib.dump(
    model_package,
    MODEL_PATH
)

print(
    f"Model saved successfully to: {MODEL_PATH}"
)