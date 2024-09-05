import pytest
from app.model import ModelManager
from app.schemas import ModelInfo

import os
import joblib

@pytest.fixture
def model_manager():
    return ModelManager()

def test_load_data(model_manager):
    X, y = model_manager.load_data()
    assert len(X) == 150  # Iris dataset has 150 samples
    assert len(y) == 150
    assert all(len(features) == 4 for features in X)  # 4 features per sample
    assert all(label in [0, 1, 2] for label in y)  # 3 classes in Iris dataset

def test_split_data(model_manager):
    X, y = model_manager.load_data()
    X_train, X_test, y_train, y_test = model_manager.split_data(X, y)
    assert len(X_train) == 15  # 10% of 150 is 15
    assert len(X_test) == 135
    assert len(y_train) == 15
    assert len(y_test) == 135

def test_train_model(model_manager):
    accuracy, model = model_manager.train_model()
    assert 0 <= accuracy <= 1  # Accuracy should be between 0 and 1
    assert hasattr(model, 'predict')  # Trained model should have a predict method

def test_predict(model_manager):
    model_manager.train_model()  # Ensure model is trained
    prediction = model_manager.predict([5.1, 3.5, 1.4, 0.2])  # Example Iris setosa features
    assert prediction in [0, 1, 2]  # Prediction should be one of the Iris classes
