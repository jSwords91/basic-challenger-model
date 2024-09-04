from pydantic import BaseModel
from typing import List

class PredictionInput(BaseModel):
    features: List[float]

class PredictionOutput(BaseModel):
    prediction: int

class ModelInfo(BaseModel):
    version: int
    path: str
    accuracy: float

class ModelUpdate(BaseModel):
    message: str
    accuracy: float