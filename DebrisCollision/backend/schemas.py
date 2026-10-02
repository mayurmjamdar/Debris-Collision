from typing import Optional
from pydantic import BaseModel


class PredictionResponse(BaseModel):
    event_id: str

    time_to_tca: Optional[float] = None
    miss_distance: Optional[float] = None
    relative_speed: Optional[float] = None

    predicted_risk: Optional[float] = None
    high_risk_probability: Optional[float] = None

    classification: Optional[str] = None