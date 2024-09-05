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

def test_save_and_load_model(model_manager):
    # Train and save model
    accuracy, _ = model_manager.train_model()
    model_path = "models/model_v0.joblib"
    model_manager.save_model(accuracy, version=0)
    
    # Load model
    loaded_model_manager = ModelManager()
    loaded_model_manager.model = joblib.load(model_path)
    ## Delete the model
    os.remove(model_path)
    
    # Compare predictions
    test_features = [5.1, 3.5, 1.4, 0.2]
    original_prediction = model_manager.predict(test_features)
    loaded_prediction = loaded_model_manager.predict(test_features)
    assert original_prediction == loaded_prediction

def test_challenger_process(model_manager, monkeypatch):
    # Mock the load_model_info method to return None (simulating no existing model)
    monkeypatch.setattr(model_manager, 'load_model_info', lambda: None)
    
    # Mock the save_model method to do nothing
    monkeypatch.setattr(model_manager, 'save_model', lambda *args: None)
    
    is_champion, accuracy = model_manager.challenger_process()
    assert is_champion == True
    assert 0 <= accuracy <= 1

    # Now mock load_model_info to return an existing model info
    monkeypatch.setattr(model_manager, 'load_model_info', lambda: ModelInfo(version=1, path="dummy_path", accuracy=0.5))
    
    # Mock train_model to return a higher accuracy
    monkeypatch.setattr(model_manager, 'train_model', lambda: (0.6, None))
    
    is_champion, accuracy = model_manager.challenger_process()
    assert is_champion == True
    assert accuracy == 0.6

    # Mock train_model to return a lower accuracy
    monkeypatch.setattr(model_manager, 'train_model', lambda: (0.4, None))
    
    is_champion, accuracy = model_manager.challenger_process()
    assert is_champion == False
    assert accuracy == 0.5