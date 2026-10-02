from pathlib import Path

import pandas as pd

from fastapi import (
    FastAPI,
    UploadFile,
    File,
    HTTPException
)

from fastapi.middleware.cors import CORSMiddleware


from data_service import get_event_data
from model_service import predict_event


# ============================================================
# APP
# ============================================================

app = FastAPI(
    title="Space Debris Collision Prediction API",
    description=(
        "Backend API for the ESA Collision Avoidance "
        "Machine Learning project."
    ),
    version="1.0.0"
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,

    allow_origins=["*"],

    allow_credentials=True,

    allow_methods=["*"],

    allow_headers=["*"]
)


# ============================================================
# TEMPORARY DATA STORAGE
# ============================================================

uploaded_dataframe = None


# ============================================================
# ROOT
# ============================================================

@app.get("/")
def root():

    return {
        "system": "Space Debris Collision Predictor",
        "status": "online",
        "version": "1.0.0"
    }


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/api/health")
def health():

    return {
        "status": "healthy"
    }


# ============================================================
# UPLOAD CSV
# ============================================================

@app.post("/api/upload")
async def upload_csv(
    file: UploadFile = File(...)
):

    global uploaded_dataframe

    if not file.filename.lower().endswith(
        ".csv"
    ):

        raise HTTPException(
            status_code=400,
            detail="Only CSV files are supported."
        )

    try:

        uploaded_dataframe = pd.read_csv(
            file.file
        )

    except Exception as e:

        raise HTTPException(
            status_code=400,
            detail=f"Could not read CSV: {str(e)}"
        )


    if "event_id" not in uploaded_dataframe.columns:

        uploaded_dataframe = None

        raise HTTPException(
            status_code=400,
            detail="CSV must contain event_id."
        )


    if "time_to_tca" not in uploaded_dataframe.columns:

        uploaded_dataframe = None

        raise HTTPException(
            status_code=400,
            detail="CSV must contain time_to_tca."
        )


    return {

        "message": "CSV uploaded successfully.",

        "filename": file.filename,

        "rows": len(
            uploaded_dataframe
        ),

        "columns": len(
            uploaded_dataframe.columns
        ),

        "events": int(
            uploaded_dataframe["event_id"]
            .nunique()
        )
    }


# ============================================================
# EVENTS
# ============================================================

@app.get("/api/events")
def get_events():

    global uploaded_dataframe

    if uploaded_dataframe is None:

        raise HTTPException(
            status_code=400,
            detail="Upload a CSV first."
        )


    events = (
        uploaded_dataframe["event_id"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )


    events.sort()


    return {

        "count": len(events),

        "events": events
    }


# ============================================================
# EVENT INFORMATION
# ============================================================

@app.get("/api/event/{event_id}")
def get_event(event_id: str):

    global uploaded_dataframe

    if uploaded_dataframe is None:

        raise HTTPException(
            status_code=400,
            detail="Upload a CSV first."
        )


    result = get_event_data(
        uploaded_dataframe,
        event_id
    )


    if result is None:

        raise HTTPException(
            status_code=404,
            detail="Event not found."
        )


    return result


# ============================================================
# PREDICT EVENT
# ============================================================

@app.post("/api/predict/{event_id}")
def predict(event_id: str):

    global uploaded_dataframe

    if uploaded_dataframe is None:

        raise HTTPException(
            status_code=400,
            detail="Upload a CSV first."
        )


    event_df = uploaded_dataframe[
        uploaded_dataframe["event_id"]
        .astype(str)
        == str(event_id)
    ].copy()


    if event_df.empty:

        raise HTTPException(
            status_code=404,
            detail="Event not found."
        )


    result = predict_event(
        event_df
    )


    return result