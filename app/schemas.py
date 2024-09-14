from pydantic import BaseModel, Field
from typing import List, Optional, Any

class PredictionInput(BaseModel):
    features: List[float] = Field(..., min_items=4, max_items=4)

class PredictionOutput(BaseModel):
    prediction: int
    probability: float

class ModelInfo(BaseModel):
    version: int
    path: str
    accuracy: float

class ModelMetadata(BaseModel):
    model_type: str
    feature_importance: Optional[dict[str, float]]
    model_parameters: dict[str, Any]

class ModelUpdate(BaseModel):
    message: str
    accuracy: float