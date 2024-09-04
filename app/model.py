from sklearn import datasets, model_selection, svm, linear_model
from sklearn.base import BaseEstimator
from typing import List, Tuple, Optional
from .logger import model_logger
from .config import settings
from .schemas import ModelInfo
from .utils import log_time
import joblib
import os
import json


class ModelManager:
    def __init__(self, model: BaseEstimator = linear_model.LogisticRegression()):
        self.model = model

    @log_time
    def load_data(self) -> Tuple[List[List[float]], List[int]]:
        model_logger.info("Loading iris dataset")
        iris = datasets.load_iris()
        return iris.data.tolist(), iris.target.tolist()

    def split_data(self, X: List[List[float]], y: List[int]) -> Tuple[List[List[float]], List[List[float]], List[int], List[int]]:
        return model_selection.train_test_split(X, y, train_size=0.10)  # make it hard to train for this example

    def fit_model(self, X_train: List[List[float]], y_train: List[int]):
        self.model.fit(X_train, y_train)

    def score_model(self, X_test: List[List[float]], y_test: List[int]) -> float:
        return self.model.score(X_test, y_test)

    @log_time
    def train_model(self) -> Tuple[float, BaseEstimator]:
        X, y = self.load_data()
        X_train, X_test, y_train, y_test = self.split_data(X, y)
        self.fit_model(X_train, y_train)
        accuracy = self.score_model(X_test, y_test)
        return accuracy, self.model

    @log_time
    def predict(self, features: List[float]) -> int:
        prediction = self.model.predict([features])[0]
        model_logger.info(f"Prediction made. Input: {features}, Prediction: {prediction}")
        return prediction

    def load_or_train_model(self):
        model_info = self.load_model_info()
        if model_info and os.path.exists(model_info.path):
            model_logger.info(f"Loading existing model (version {model_info.version}) from {model_info.path}")
            self.model = joblib.load(model_info.path)
            model_logger.info("Model loaded successfully")
        else:
            model_logger.info("No existing model found. Training a new model.")
            accuracy, _ = self.train_model()
            self.save_model(accuracy, 1)

    def save_model(self, accuracy: float, version: int):
        model_dir = os.path.dirname(settings.model_path)
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, f"model_v{version}.joblib")
        joblib.dump(self.model, model_path)
        model_logger.info(f"Model (version {version}) saved to {model_path}")
        self.save_model_info(ModelInfo(version=version, path=model_path, accuracy=accuracy))

    def load_model_info(self) -> Optional[ModelInfo]:
        if os.path.exists(settings.model_info_path):
            with open(settings.model_info_path, 'r') as f:
                data = json.load(f)
                return ModelInfo(**data)
        return None

    def save_model_info(self, model_info: ModelInfo):
        with open(settings.model_info_path, 'w') as f:
            json.dump(model_info.dict(), f)

    def challenger_process(self) -> Tuple[bool, float]:
        model_logger.info("Starting challenger process")
        current_model_info = self.load_model_info()
        if not current_model_info:
            model_logger.info("No existing model found. Training a new model.")
            accuracy, _ = self.train_model()
            self.save_model(accuracy, 1)
            return True, accuracy

        current_accuracy = current_model_info.accuracy
        new_accuracy, _ = self.train_model()

        if new_accuracy > current_accuracy:
            model_logger.info(f"New model outperforms current model. New accuracy: {new_accuracy}, Old accuracy: {current_accuracy}")
            new_version = current_model_info.version + 1
            self.save_model(new_accuracy, new_version)
            return True, new_accuracy
        else:
            model_logger.info(f"Current model remains champion. New accuracy: {new_accuracy}, Current accuracy: {current_accuracy}")
            return False, current_accuracy