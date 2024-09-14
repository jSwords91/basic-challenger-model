import pytest
from app.model import ModelManager, DataHandler
from app.schemas import PredictionOutput

@pytest.fixture
def model_manager():
    return ModelManager()

def test_data_handler():
    data_handler = DataHandler()
    X, y = data_handler.load_data()
    assert len(X) == 150  # Iris dataset has 150 samples
    assert len(y) == 150
    assert X.shape[1] == 4  # 4 features per sample

def test_model_training(model_manager):
    model_manager.load_or_train_model()
    assert hasattr(model_manager.model, 'predict')
    assert hasattr(model_manager.model, 'predict_proba')

def test_prediction(model_manager):
    model_manager.load_or_train_model()
    features = [5.1, 3.5, 1.4, 0.2]  # Example Iris setosa features
    prediction = model_manager.predict(features)
    assert isinstance(prediction, PredictionOutput)
    assert 0 <= prediction.prediction <= 2  # Prediction should be one of the Iris classes
    assert 0 <= prediction.probability <= 1  # Probability should be between 0 and 1

def test_challenger_process(model_manager):
    updated, accuracy = model_manager.challenger_process()
    assert isinstance(updated, bool)
    assert 0 <= accuracy <= 1
