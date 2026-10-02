import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Space Debris Collision Predictor",
    page_icon="🛰️",
    layout="wide"
)


# ============================================================
# CONSTANTS
# ============================================================

REGRESSION_MODEL_PATH = "models/collision_model.pkl"
CLASSIFIER_MODEL_PATH = "models/high_risk_classifier.pkl"

TCA_CUTOFF_DAYS = 2.0


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 38px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .subtitle {
        font-size: 17px;
        color: #777;
        margin-bottom: 25px;
    }

    .section-title {
        font-size: 25px;
        font-weight: 650;
        margin-top: 20px;
        margin-bottom: 10px;
    }

    .risk-high {
        padding: 18px;
        border-radius: 12px;
        text-align: center;
        font-size: 28px;
        font-weight: 700;
        border: 2px solid #ff4b4b;
    }

    .risk-low {
        padding: 18px;
        border-radius: 12px;
        text-align: center;
        font-size: 28px;
        font-weight: 700;
        border: 2px solid #21c354;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def find_column(columns, prefix=None, keywords=None):
    """
    Find a column using:
    - optional prefix such as t_ or c_
    - one or more keywords

    Returns the first matching column or None.
    """

    if keywords is None:
        keywords = []

    keywords = [str(k).lower() for k in keywords]

    for col in columns:

        col_lower = str(col).lower()

        if prefix is not None:
            if not col_lower.startswith(prefix.lower()):
                continue

        if all(keyword in col_lower for keyword in keywords):
            return col

    # Second pass:
    # If all keywords don't match, allow any keyword match.
    for col in columns:

        col_lower = str(col).lower()

        if prefix is not None:
            if not col_lower.startswith(prefix.lower()):
                continue

        if any(keyword in col_lower for keyword in keywords):
            return col

    return None


def get_value(row, column):
    """
    Safely retrieve a value from a pandas row.
    """

    if column is None:
        return None

    if column not in row.index:
        return None

    value = row[column]

    if pd.isna(value):
        return None

    return value


def format_value(value, decimals=4):
    """
    Format values nicely for display.
    """

    if value is None:
        return "Not available"

    if isinstance(value, (float, np.floating)):
        return f"{value:.{decimals}f}"

    if isinstance(value, (int, np.integer)):
        return str(value)

    return str(value)


def clean_features(dataframe):
    """
    Clean model input:
    - replace inf
    - replace very large values
    - preserve feature structure
    """

    data = dataframe.copy()

    data = data.replace([np.inf, -np.inf], np.nan)

    numeric_columns = data.select_dtypes(include=[np.number]).columns

    for col in numeric_columns:

        data.loc[
            data[col].abs() > 1e20,
            col
        ] = np.nan

    return data


def safe_numeric(value):
    """
    Convert value to float when possible.
    """

    try:
        return float(value)
    except:
        return np.nan


# ============================================================
# LOAD MODELS
# ============================================================

@st.cache_resource
def load_models():

    regression_package = joblib.load(REGRESSION_MODEL_PATH)
    classifier_package = joblib.load(CLASSIFIER_MODEL_PATH)

    return regression_package, classifier_package


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">🛰️ Space Debris Collision Predictor</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'ESA Collision Avoidance dataset + Machine Learning'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# LOAD MODELS
# ============================================================

try:

    regression_package, classifier_package = load_models()

except Exception as e:

    st.error(
        "Could not load the trained models."
    )

    st.code(str(e))

    st.stop()


# ============================================================
# EXTRACT MODEL INFORMATION
# ============================================================

if isinstance(regression_package, dict):

    regression_model = regression_package.get(
        "model",
        regression_package.get("regressor", regression_package)
    )

    regression_features = regression_package.get(
        "features",
        regression_package.get("feature_names", None)
    )

else:

    regression_model = regression_package
    regression_features = None


if isinstance(classifier_package, dict):

    classifier_model = classifier_package.get(
        "model",
        classifier_package.get("classifier", classifier_package)
    )

    classifier_features = classifier_package.get(
        "features",
        classifier_package.get("feature_names", None)
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
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("⚙️ Settings")

    st.write(
        "This application uses the trained ESA collision-risk models."
    )

    st.divider()

    st.write("**Prediction settings**")

    st.write(
        f"CDM cutoff: **{TCA_CUTOFF_DAYS:.0f} days before TCA**"
    )

    st.write(
        f"High-risk threshold: **{HIGH_RISK_THRESHOLD}**"
    )

    st.write(
        f"Classifier probability threshold: **{CLASSIFICATION_THRESHOLD}**"
    )

    st.divider()

    st.caption(
        "The current ML model expects the ESA dataset features. "
        "Custom orbital inputs will be added separately."
    )


# ============================================================
# FILE UPLOAD
# ============================================================

st.markdown(
    '<div class="section-title">📂 Upload Dataset</div>',
    unsafe_allow_html=True
)

uploaded_file = st.file_uploader(
    "Upload an ESA collision-avoidance CSV file",
    type=["csv"]
)


if uploaded_file is None:

    st.info(
        "Upload `test_data.csv` to run predictions."
    )

    st.stop()


# ============================================================
# READ DATA
# ============================================================

try:

    df = pd.read_csv(uploaded_file)

except Exception as e:

    st.error("Could not read the CSV file.")

    st.code(str(e))

    st.stop()


# ============================================================
# BASIC VALIDATION
# ============================================================

required_columns = [
    "event_id",
    "time_to_tca"
]

missing_required = [
    col for col in required_columns
    if col not in df.columns
]


if missing_required:

    st.error(
        f"Missing required columns: {missing_required}"
    )

    st.stop()


# ============================================================
# DATA CLEANING
# ============================================================

df = df.replace([np.inf, -np.inf], np.nan)


# ============================================================
# DATASET OVERVIEW
# ============================================================

st.markdown(
    '<div class="section-title">📊 Dataset Overview</div>',
    unsafe_allow_html=True
)

metric1, metric2, metric3, metric4 = st.columns(4)

with metric1:
    st.metric(
        "Rows",
        f"{len(df):,}"
    )

with metric2:
    st.metric(
        "Columns",
        f"{len(df.columns):,}"
    )

with metric3:
    st.metric(
        "Events",
        f"{df['event_id'].nunique():,}"
    )

with metric4:
    st.metric(
        "Missing Values",
        f"{int(df.isna().sum().sum()):,}"
    )


# ============================================================
# EVENT SELECTION
# ============================================================

st.markdown(
    '<div class="section-title">🎯 Select Conjunction Event</div>',
    unsafe_allow_html=True
)

event_ids = (
    df["event_id"]
    .dropna()
    .astype(str)
    .unique()
    .tolist()
)

event_ids = sorted(event_ids)


selected_event = st.selectbox(
    "Select event_id",
    event_ids
)


# ============================================================
# EVENT DATA
# ============================================================

event_df = df[
    df["event_id"].astype(str) == selected_event
].copy()


event_df["time_to_tca"] = pd.to_numeric(
    event_df["time_to_tca"],
    errors="coerce"
)


event_df = event_df.sort_values(
    "time_to_tca",
    ascending=False
).reset_index(drop=True)


if len(event_df) == 0:

    st.warning(
        "No data found for the selected event."
    )

    st.stop()


# Latest available row in chronological progression
latest_event_row = event_df.iloc[-1]


# ============================================================
# TARGET / CHASER INFORMATION
# ============================================================

st.markdown(
    '<div class="section-title">🛰️ Object Information</div>',
    unsafe_allow_html=True
)


# ------------------------------------------------------------
# Target columns
# ------------------------------------------------------------

target_sma = find_column(
    event_df.columns,
    "t_",
    ["sma"]
)

target_ecc = find_column(
    event_df.columns,
    "t_",
    ["ecc"]
)

target_inc = find_column(
    event_df.columns,
    "t_",
    ["inc"]
)

target_apogee = find_column(
    event_df.columns,
    "t_",
    ["apogee"]
)

target_perigee = find_column(
    event_df.columns,
    "t_",
    ["perigee"]
)

target_rcs = find_column(
    event_df.columns,
    "t_",
    ["rcs"]
)

target_area_mass = find_column(
    event_df.columns,
    "t_",
    ["area", "mass"]
)


# ------------------------------------------------------------
# Chaser columns
# ------------------------------------------------------------

chaser_sma = find_column(
    event_df.columns,
    "c_",
    ["sma"]
)

chaser_ecc = find_column(
    event_df.columns,
    "c_",
    ["ecc"]
)

chaser_inc = find_column(
    event_df.columns,
    "c_",
    ["inc"]
)

chaser_apogee = find_column(
    event_df.columns,
    "c_",
    ["apogee"]
)

chaser_perigee = find_column(
    event_df.columns,
    "c_",
    ["perigee"]
)

chaser_rcs = find_column(
    event_df.columns,
    "c_",
    ["rcs"]
)

chaser_area_mass = find_column(
    event_df.columns,
    "c_",
    ["area", "mass"]
)


# ------------------------------------------------------------
# Target panel
# ------------------------------------------------------------

target_col, chaser_col = st.columns(2)


with target_col:

    st.subheader("🎯 Target")

    target_metrics = [
        (
            "Semi-major axis",
            target_sma,
            3
        ),
        (
            "Eccentricity",
            target_ecc,
            6
        ),
        (
            "Inclination",
            target_inc,
            4
        ),
        (
            "Apogee",
            target_apogee,
            3
        ),
        (
            "Perigee",
            target_perigee,
            3
        ),
        (
            "RCS",
            target_rcs,
            4
        ),
        (
            "Area / Mass",
            target_area_mass,
            6
        )
    ]

    for label, column, decimals in target_metrics:

        value = get_value(
            latest_event_row,
            column
        )

        st.write(
            f"**{label}:** {format_value(value, decimals)}"
        )


# ------------------------------------------------------------
# Chaser panel
# ------------------------------------------------------------

with chaser_col:

    st.subheader("🛰️ Chaser")

    chaser_metrics = [
        (
            "Semi-major axis",
            chaser_sma,
            3
        ),
        (
            "Eccentricity",
            chaser_ecc,
            6
        ),
        (
            "Inclination",
            chaser_inc,
            4
        ),
        (
            "Apogee",
            chaser_apogee,
            3
        ),
        (
            "Perigee",
            chaser_perigee,
            3
        ),
        (
            "RCS",
            chaser_rcs,
            4
        ),
        (
            "Area / Mass",
            chaser_area_mass,
            6
        )
    ]

    for label, column, decimals in chaser_metrics:

        value = get_value(
            latest_event_row,
            column
        )

        st.write(
            f"**{label}:** {format_value(value, decimals)}"
        )


# ============================================================
# CLOSE APPROACH
# ============================================================

st.markdown(
    '<div class="section-title">⚠️ Close Approach</div>',
    unsafe_allow_html=True
)


miss_distance_col = find_column(
    event_df.columns,
    None,
    ["miss_distance"]
)

relative_speed_col = find_column(
    event_df.columns,
    None,
    ["relative_speed"]
)


time_to_tca = safe_numeric(
    get_value(
        latest_event_row,
        "time_to_tca"
    )
)

miss_distance = safe_numeric(
    get_value(
        latest_event_row,
        miss_distance_col
    )
)

relative_speed = safe_numeric(
    get_value(
        latest_event_row,
        relative_speed_col
    )
)


c1, c2, c3 = st.columns(3)


with c1:

    if not np.isnan(time_to_tca):

        st.metric(
            "⏱️ Time to TCA",
            f"{time_to_tca:.4f} days"
        )

    else:

        st.metric(
            "⏱️ Time to TCA",
            "N/A"
        )


with c2:

    if not np.isnan(miss_distance):

        st.metric(
            "📏 Miss Distance",
            f"{miss_distance:,.2f} m"
        )

    else:

        st.metric(
            "📏 Miss Distance",
            "N/A"
        )


with c3:

    if not np.isnan(relative_speed):

        st.metric(
            "⚡ Relative Speed",
            f"{relative_speed:,.2f} m/s"
        )

    else:

        st.metric(
            "⚡ Relative Speed",
            "N/A"
        )


# ============================================================
# RELATIVE POSITION
# ============================================================

st.markdown(
    '<div class="section-title">📐 Relative Position — RTN Frame</div>',
    unsafe_allow_html=True
)


position_r = find_column(
    event_df.columns,
    None,
    ["relative_position_r"]
)

position_t = find_column(
    event_df.columns,
    None,
    ["relative_position_t"]
)

position_n = find_column(
    event_df.columns,
    None,
    ["relative_position_n"]
)


p1, p2, p3 = st.columns(3)


with p1:

    value = safe_numeric(
        get_value(
            latest_event_row,
            position_r
        )
    )

    st.metric(
        "R — Radial",
        "N/A" if np.isnan(value)
        else f"{value:,.2f} m"
    )


with p2:

    value = safe_numeric(
        get_value(
            latest_event_row,
            position_t
        )
    )

    st.metric(
        "T — Transverse",
        "N/A" if np.isnan(value)
        else f"{value:,.2f} m"
    )


with p3:

    value = safe_numeric(
        get_value(
            latest_event_row,
            position_n
        )
    )

    st.metric(
        "N — Normal",
        "N/A" if np.isnan(value)
        else f"{value:,.2f} m"
    )


# ============================================================
# RELATIVE VELOCITY
# ============================================================

st.markdown(
    '<div class="section-title">🚀 Relative Velocity — RTN Frame</div>',
    unsafe_allow_html=True
)


velocity_r = find_column(
    event_df.columns,
    None,
    ["relative_velocity_r"]
)

velocity_t = find_column(
    event_df.columns,
    None,
    ["relative_velocity_t"]
)

velocity_n = find_column(
    event_df.columns,
    None,
    ["relative_velocity_n"]
)


v1, v2, v3 = st.columns(3)


with v1:

    value = safe_numeric(
        get_value(
            latest_event_row,
            velocity_r
        )
    )

    st.metric(
        "R — Radial",
        "N/A" if np.isnan(value)
        else f"{value:,.2f} m/s"
    )


with v2:

    value = safe_numeric(
        get_value(
            latest_event_row,
            velocity_t
        )
    )

    st.metric(
        "T — Transverse",
        "N/A" if np.isnan(value)
        else f"{value:,.2f} m/s"
    )


with v3:

    value = safe_numeric(
        get_value(
            latest_event_row,
            velocity_n
        )
    )

    st.metric(
        "N — Normal",
        "N/A" if np.isnan(value)
        else f"{value:,.2f} m/s"
    )


# ============================================================
# SELECT CDM FOR ML PREDICTION
# ============================================================

st.markdown(
    '<div class="section-title">🤖 Machine Learning Prediction</div>',
    unsafe_allow_html=True
)


# Only use CDMs available at least 2 days before TCA

prediction_candidates = event_df[
    event_df["time_to_tca"] >= TCA_CUTOFF_DAYS
].copy()


if len(prediction_candidates) == 0:

    st.warning(
        f"No CDM is available at least "
        f"{TCA_CUTOFF_DAYS:.0f} days before TCA "
        f"for this event."
    )

    st.info(
        "The ML model follows the project's "
        "2-day-before-TCA prediction rule."
    )

else:

    # Select the latest available CDM before the cutoff.
    prediction_row = (
        prediction_candidates
        .sort_values(
            "time_to_tca",
            ascending=True
        )
        .iloc[0]
    )


    prediction_time = prediction_row[
        "time_to_tca"
    ]


    st.info(
        f"Prediction uses the latest available CDM "
        f"at **{prediction_time:.4f} days before TCA**."
    )


    # --------------------------------------------------------
    # REGRESSION PREDICTION
    # --------------------------------------------------------

    regression_prediction = None

    if regression_features is not None:

        regression_input = pd.DataFrame(
            [prediction_row]
        ).reindex(
            columns=regression_features
        )

    else:

        regression_input = pd.DataFrame(
            [prediction_row]
        )


    regression_input = clean_features(
        regression_input
    )


    try:

        regression_prediction = regression_model.predict(
            regression_input
        )[0]

    except Exception as e:

        st.error(
            "Regression prediction failed."
        )

        st.code(str(e))


    # --------------------------------------------------------
    # CLASSIFICATION PREDICTION
    # --------------------------------------------------------

    high_risk_probability = None

    predicted_class = None


    if classifier_features is not None:

        classifier_input = pd.DataFrame(
            [prediction_row]
        ).reindex(
            columns=classifier_features
        )

    else:

        classifier_input = pd.DataFrame(
            [prediction_row]
        )


    classifier_input = clean_features(
        classifier_input
    )


    try:

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

            high_risk_probability = probabilities[0][1]

        else:

            predicted_class_raw = (
                classifier_model
                .predict(
                    classifier_input
                )[0]
            )

            high_risk_probability = float(
                predicted_class_raw
            )


        if (
            high_risk_probability
            >= CLASSIFICATION_THRESHOLD
        ):

            predicted_class = "HIGH"

        else:

            predicted_class = "LOW"


    except Exception as e:

        st.error(
            "Classification prediction failed."
        )

        st.code(str(e))


    # --------------------------------------------------------
    # DISPLAY RESULTS
    # --------------------------------------------------------

    st.divider()


    r1, r2, r3 = st.columns(3)


    with r1:

        if regression_prediction is not None:

            st.metric(
                "Predicted Risk",
                f"{regression_prediction:.4f}"
            )

        else:

            st.metric(
                "Predicted Risk",
                "N/A"
            )


    with r2:

        if high_risk_probability is not None:

            st.metric(
                "High-Risk Probability",
                f"{high_risk_probability * 100:.2f}%"
            )

        else:

            st.metric(
                "High-Risk Probability",
                "N/A"
            )


    with r3:

        if predicted_class is not None:

            st.metric(
                "Classification",
                predicted_class
            )

        else:

            st.metric(
                "Classification",
                "N/A"
            )


    # --------------------------------------------------------
    # RISK BANNER
    # --------------------------------------------------------

    if predicted_class == "HIGH":

        st.markdown(
            """
            <div class="risk-high">
                🚨 HIGH RISK
            </div>
            """,
            unsafe_allow_html=True
        )

    elif predicted_class == "LOW":

        st.markdown(
            """
            <div class="risk-low">
                🟢 LOW RISK
            </div>
            """,
            unsafe_allow_html=True
        )


    st.caption(
        "Risk is represented on the model's logarithmic risk scale. "
        "The high-risk probability is the classifier output."
    )


# ============================================================
# EVENT PROGRESSION
# ============================================================

st.markdown(
    '<div class="section-title">📈 Event Progression</div>',
    unsafe_allow_html=True
)


risk_column = find_column(
    event_df.columns,
    None,
    ["risk"]
)


if risk_column is not None:

    plot_df = event_df[
        [
            "time_to_tca",
            risk_column
        ]
    ].copy()

    plot_df["time_to_tca"] = pd.to_numeric(
        plot_df["time_to_tca"],
        errors="coerce"
    )

    plot_df[risk_column] = pd.to_numeric(
        plot_df[risk_column],
        errors="coerce"
    )

    plot_df = plot_df.dropna()


    if len(plot_df) > 1:

        plot_df = plot_df.sort_values(
            "time_to_tca"
        )


        fig, ax = plt.subplots(
            figsize=(10, 4)
        )

        ax.plot(
            plot_df["time_to_tca"],
            plot_df[risk_column],
            marker="o"
        )

        ax.set_xlabel(
            "Time to TCA (days)"
        )

        ax.set_ylabel(
            "Risk"
        )

        ax.set_title(
            "Risk Evolution Across CDMs"
        )

        ax.grid(
            alpha=0.3
        )

        st.pyplot(
            fig,
            use_container_width=True
        )

        plt.close(fig)

    else:

        st.info(
            "Not enough risk observations to plot progression."
        )

else:

    st.info(
        "Risk column not available in this uploaded dataset."
    )


# ============================================================
# EVENT TIMELINE
# ============================================================

st.markdown(
    '<div class="section-title">🕒 Event Timeline</div>',
    unsafe_allow_html=True
)


timeline_columns = [
    col for col in [
        "event_id",
        "time_to_tca",
        miss_distance_col,
        relative_speed_col,
        risk_column
    ]
    if col is not None and col in event_df.columns
]


if timeline_columns:

    timeline_df = event_df[
        timeline_columns
    ].copy()

    timeline_df = timeline_df.sort_values(
        "time_to_tca",
        ascending=False
    )

    st.dataframe(
        timeline_df,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# SELECTED CDM DETAILS
# ============================================================

st.markdown(
    '<div class="section-title">📋 CDM Details</div>',
    unsafe_allow_html=True
)


with st.expander(
    "Show selected event data"
):

    st.dataframe(
        event_df,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# FEATURE AVAILABILITY
# ============================================================

with st.expander(
    "🔍 Model Feature Diagnostics"
):

    col_a, col_b = st.columns(2)


    with col_a:

        st.write(
            "**Regression model features:**"
        )

        if regression_features is not None:

            st.write(
                len(regression_features)
            )

            missing_regression = [
                feature
                for feature in regression_features
                if feature not in df.columns
            ]

            if missing_regression:

                st.warning(
                    f"{len(missing_regression)} "
                    "regression features are missing "
                    "from this CSV."
                )

                st.write(
                    missing_regression
                )

            else:

                st.success(
                    "All regression features are present."
                )


    with col_b:

        st.write(
            "**Classifier features:**"
        )

        if classifier_features is not None:

            st.write(
                len(classifier_features)
            )

            missing_classifier = [
                feature
                for feature in classifier_features
                if feature not in df.columns
            ]

            if missing_classifier:

                st.warning(
                    f"{len(missing_classifier)} "
                    "classifier features are missing "
                    "from this CSV."
                )

                st.write(
                    missing_classifier
                )

            else:

                st.success(
                    "All classifier features are present."
                )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Space Debris Collision Prediction Project | "
    "ESA Collision Avoidance Dataset | "
    "Machine Learning Prototype"
)