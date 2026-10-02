from pathlib import Path

import joblib
import pandas as pd
import numpy as np


# ============================================================
# MODEL PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

REGRESSION_MODEL_PATH = (
    BASE_DIR / "models" / "collision_model.pkl"
)

CLASSIFIER_MODEL_PATH = (
    BASE_DIR / "models" / "high_risk_classifier.pkl"
)


TCA_CUTOFF_DAYS = 2.0


# ============================================================
# LOAD MODELS
# ============================================================

regression_package = joblib.load(
    REGRESSION_MODEL_PATH
)

classifier_package = joblib.load(
    CLASSIFIER_MODEL_PATH
)


# ============================================================
# REGRESSION MODEL
# ============================================================

if isinstance(regression_package, dict):

    regression_model = regression_package.get(
        "model",
        regression_package.get(
            "regressor",
            regression_package
        )
    )

    regression_features = regression_package.get(
        "features",
        regression_package.get(
            "feature_names",
            None
        )
    )

else:

    regression_model = regression_package
    regression_features = None


# ============================================================
# CLASSIFIER
# ============================================================

if isinstance(classifier_package, dict):

    classifier_model = classifier_package.get(
        "model",
        classifier_package.get(
            "classifier",
            classifier_package
        )
    )

    classifier_features = classifier_package.get(
        "features",
        classifier_package.get(
            "feature_names",
            None
        )
    )

    HIGH_RISK_THRESHOLD = classifier_package.get(
        "high_risk_threshold",
        -6.0
    )

    CLASSIFICATION_THRESHOLD = classifier_package.get(
        "classification_threshold",
        0.15
    )

else:

    classifier_model = classifier_package

    classifier_features = None

    HIGH_RISK_THRESHOLD = -6.0

    CLASSIFICATION_THRESHOLD = 0.15


# ============================================================
# CLEAN FEATURES
# ============================================================

def clean_features(dataframe):

    data = dataframe.copy()

    data = data.replace(
        [np.inf, -np.inf],
        np.nan
    )

    numeric_columns = (
        data
        .select_dtypes(include=[np.number])
        .columns
    )

    for column in numeric_columns:

        data.loc[
            data[column].abs() > 1e20,
            column
        ] = np.nan

    return data


# ============================================================
# PREDICT
# ============================================================

def predict_event(event_df):

    event_df = event_df.copy()

    event_df["time_to_tca"] = pd.to_numeric(
        event_df["time_to_tca"],
        errors="coerce"
    )

    # --------------------------------------------------------
    # Only use CDMs available >= 2 days before TCA
    # --------------------------------------------------------

    candidates = event_df[
        event_df["time_to_tca"] >= TCA_CUTOFF_DAYS
    ].copy()

    if candidates.empty:

        return {
            "prediction_available": False,
            "message": (
                "No CDM available at least "
                f"{TCA_CUTOFF_DAYS} days before TCA."
            )
        }

    # --------------------------------------------------------
    # Latest available CDM before cutoff
    # --------------------------------------------------------

    prediction_row = (
        candidates
        .sort_values(
            "time_to_tca",
            ascending=True
        )
        .iloc[0]
    )

    # --------------------------------------------------------
    # REGRESSION
    # --------------------------------------------------------

    if regression_features is not None:

        regression_input = (
            pd.DataFrame([prediction_row])
            .reindex(
                columns=regression_features
            )
        )

    else:

        regression_input = pd.DataFrame(
            [prediction_row]
        )

    regression_input = clean_features(
        regression_input
    )

    predicted_risk = regression_model.predict(
        regression_input
    )[0]

    # --------------------------------------------------------
    # CLASSIFIER
    # --------------------------------------------------------

    if classifier_features is not None:

        classifier_input = (
            pd.DataFrame([prediction_row])
            .reindex(
                columns=classifier_features
            )
        )

    else:

        classifier_input = pd.DataFrame(
            [prediction_row]
        )

    classifier_input = clean_features(
        classifier_input
    )

    if hasattr(
        classifier_model,
        "predict_proba"
    ):

        probabilities = (
            classifier_model
            .predict_proba(
                classifier_input
            )
        )

        high_risk_probability = float(
            probabilities[0][1]
        )

    else:

        high_risk_probability = float(
            classifier_model
            .predict(
                classifier_input
            )[0]
        )

    # --------------------------------------------------------
    # CLASSIFICATION
    # --------------------------------------------------------

    if (
        high_risk_probability
        >= CLASSIFICATION_THRESHOLD
    ):

        classification = "HIGH"

    else:

        classification = "LOW"

    # --------------------------------------------------------
    # RETURN
    # --------------------------------------------------------

    return {

        "prediction_available": True,

        "event_id": str(
            prediction_row["event_id"]
        ),

        "time_to_tca": float(
            prediction_row["time_to_tca"]
        ),

        "predicted_risk": float(
            predicted_risk
        ),

        "high_risk_probability": (
            high_risk_probability
        ),

        "classification": classification,

        "classification_threshold": (
            CLASSIFICATION_THRESHOLD
        ),

        "prediction_cdm_time_to_tca": float(
            prediction_row["time_to_tca"]
        )
    }
    