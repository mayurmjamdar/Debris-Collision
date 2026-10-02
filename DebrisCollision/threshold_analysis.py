import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import GroupShuffleSplit
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    precision_score,
    recall_score,
    fbeta_score
)
from sklearn.ensemble import RandomForestClassifier


# ============================================================
# 1. LOAD DATA
# ============================================================

print("Loading ESA dataset...")

df = pd.read_csv("dataset/train_data.csv")

print(f"Dataset shape: {df.shape}")


# ============================================================
# 2. SORT
# ============================================================

df = df.sort_values(
    by=["event_id", "time_to_tca"],
    ascending=[True, False]
)


# ============================================================
# 3. FINAL RISK
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


# ============================================================
# 4. CDMs AVAILABLE >= 2 DAYS BEFORE TCA
# ============================================================

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


# ============================================================
# 6. MERGE
# ============================================================

data = prediction_data.merge(
    final_data,
    on="event_id",
    how="inner"
)


# ============================================================
# 7. HIGH-RISK LABEL
# ============================================================

HIGH_RISK_THRESHOLD = -6.0

data["high_risk"] = (
    data["final_risk"] >= HIGH_RISK_THRESHOLD
).astype(int)


# ============================================================
# 8. FEATURES
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

X = X.select_dtypes(
    include=["number"]
)

X = X.astype(np.float64)

X = X.replace(
    [np.inf, -np.inf],
    np.nan
)

X = X.mask(
    X.abs() > 1e20,
    np.nan
)


# ============================================================
# 9. IMPUTE
# ============================================================

imputer = SimpleImputer(
    strategy="median"
)

X = imputer.fit_transform(X)

y = data["high_risk"]


# ============================================================
# 10. SAME EVENT-BASED SPLIT
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


# ============================================================
# 11. TRAIN CLASSIFIER
# ============================================================

print("Training classifier...")

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


# ============================================================
# 12. PREDICT PROBABILITIES
# ============================================================

probabilities = classifier.predict_proba(
    X_val
)[:, 1]


# ============================================================
# 13. TEST DIFFERENT THRESHOLDS
# ============================================================

results = []

for threshold in np.arange(
    0.10,
    0.91,
    0.05
):

    predictions = (
        probabilities >= threshold
    ).astype(int)

    precision = precision_score(
        y_val,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_val,
        predictions,
        zero_division=0
    )

    f2 = fbeta_score(
        y_val,
        predictions,
        beta=2,
        zero_division=0
    )

    results.append({
        "threshold": round(threshold, 2),
        "precision": precision,
        "recall": recall,
        "f2": f2,
        "predicted_high_risk": predictions.sum()
    })


results_df = pd.DataFrame(results)


# ============================================================
# 14. DISPLAY RESULTS
# ============================================================

print()
print("==============================")
print("THRESHOLD ANALYSIS")
print("==============================")

print(
    results_df.to_string(
        index=False,
        formatters={
            "precision": "{:.4f}".format,
            "recall": "{:.4f}".format,
            "f2": "{:.4f}".format
        }
    )
)


# ============================================================
# 15. BEST F2
# ============================================================

best = results_df.loc[
    results_df["f2"].idxmax()
]

print()
print("==============================")
print("BEST F2 THRESHOLD")
print("==============================")

print(
    f"Threshold : {best['threshold']:.2f}"
)

print(
    f"Precision : {best['precision']:.4f}"
)

print(
    f"Recall    : {best['recall']:.4f}"
)

print(
    f"F2 Score  : {best['f2']:.4f}"
)

print(
    f"Predicted high-risk events : "
    f"{int(best['predicted_high_risk'])}"
)

print("==============================")