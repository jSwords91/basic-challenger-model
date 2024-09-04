from fastapi import FastAPI
from .model import ModelManager
from .schemas import PredictionInput, PredictionOutput, ModelUpdate
from .logger import main_logger
from .config import settings
import time

app = FastAPI(title=settings.app_name, description=settings.app_description)
model_manager = ModelManager()

@app.on_event("startup")
async def startup_event():
    main_logger.info("Starting up the application")
    model_manager.load_or_train_model()

@app.post("/predict", response_model=PredictionOutput)
async def make_prediction(input_data: PredictionInput) -> PredictionOutput:
    main_logger.info(f"Received prediction request with features: {input_data.features}")
    prediction = model_manager.predict(input_data.features)
    main_logger.info(f"Prediction: {prediction}")
    return PredictionOutput(prediction=prediction)

@app.get("/train", response_model=ModelUpdate)
async def train() -> dict:
    start_time = time.time()
    main_logger.info("Received request to train model")
    is_new_champion, accuracy = model_manager.challenger_process()
    end_time = time.time()
    if is_new_champion:
        message = "New champion model deployed"
    else:
        message = "Current model remains champion"
    main_logger.info(f"{message}. Accuracy: {accuracy}, Time taken: {end_time - start_time:.4f} seconds")
    return ModelUpdate(message=message, accuracy=accuracy)

