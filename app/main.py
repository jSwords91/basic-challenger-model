from fastapi import FastAPI, HTTPException
from .model import ModelManager
from .schemas import PredictionInput, PredictionOutput, ModelUpdate
from .logger import main_logger

app = FastAPI()
model_manager = ModelManager()

@app.on_event("startup")
async def startup_event():
    main_logger.info("Starting up the application")
    model_manager.load_or_train_model()

@app.post("/predict", response_model=PredictionOutput)
async def make_prediction(input_data: PredictionInput):
    main_logger.info(f"Received prediction request with features: {input_data.features}")
    try:
        prediction = model_manager.predict(input_data.features)
        return prediction
    except Exception as e:
        main_logger.error(f"Error making prediction: {str(e)}")
        raise HTTPException(status_code=500, detail="Error making prediction")

@app.post("/update_model", response_model=ModelUpdate)
async def update_model():
    main_logger.info("Received request to update model")
    try:
        updated, accuracy = model_manager.challenger_process()
        if updated:
            message = "Model updated successfully"
        else:
            message = "Current model remains champion"
        return ModelUpdate(message=message, accuracy=accuracy)
    except Exception as e:
        main_logger.error(f"Error updating model: {str(e)}")
        raise HTTPException(status_code=500, detail="Error updating model")

