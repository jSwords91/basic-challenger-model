from sklearn import datasets, model_selection, linear_model
from sklearn.base import BaseEstimator
from typing import Tuple, Optional, List, Dict, Any
from .logger import model_logger
from .config import settings
from .schemas import ModelInfo, PredictionOutput, ModelMetadata
from .utils import log_time
import joblib
import os
import json
import numpy as np

IMPROVEMENT_THRESHOLD = 0.02 # 2% improvement required to update model

class DataHandler:
    @staticmethod
    @log_time
    def load_data() -> Tuple[np.ndarray, np.ndarray]:
        model_logger.info("Loading iris dataset for the demo...")
        iris = datasets.load_iris()
        return iris.data, iris.target

    @staticmethod
    def split_data(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        return model_selection.train_test_split(X, y, test_size=0.10, random_state=42)

class ModelPersistence:
    @staticmethod
    def load_model_info() -> Optional[ModelInfo]:
        if os.path.exists(settings.model_info_path):
            with open(settings.model_info_path, 'r') as f:
                return ModelInfo(**json.load(f))
        return None

    @staticmethod
    def save_model_info(model_info: ModelInfo) -> None:
        with open(settings.model_info_path, 'w') as f:
            json.dump(model_info.dict(), f)

    @staticmethod
    def save_model(model: BaseEstimator, accuracy: float, version: int) -> ModelInfo:
        model_dir = os.path.dirname(settings.model_path)
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, f"model_v{version}.joblib")
        joblib.dump(model, model_path)
        model_logger.info(f"Model (version {version}) saved to {model_path}")
        return ModelInfo(version=version, path=model_path, accuracy=accuracy)

    @staticmethod
    def load_model(model_info: ModelInfo) -> BaseEstimator:
        model_logger.info(f"Loading existing model (version {model_info.version}) from {model_info.path}")
        return joblib.load(model_info.path)

    @staticmethod
    def save_model_metadata(model_info: ModelInfo, metadata: ModelMetadata) -> None:
        metadata_path = os.path.join(os.path.dirname(settings.model_info_path), f"model_metadata_v{model_info.version}.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata.dict(), f)
        model_logger.info(f"Model metadata (version {model_info.version}) saved to {metadata_path}")


class ModelTrainer:
    @staticmethod
    @log_time
    def train_model(model: BaseEstimator, data_handler: DataHandler) -> Tuple[float, BaseEstimator]:
        X, y = data_handler.load_data()
        X_train, X_test, y_train, y_test = data_handler.split_data(X, y)

        model.fit(X_train, y_train)

        train_acc = model.score(X_train, y_train)
        test_acc = model.score(X_test, y_test)
        model_logger.info(f"Train acc: {train_acc:.4f} || Test acc: {test_acc:.4f}")
        return test_acc, model


class ModelManager:
    def __init__(self, model: BaseEstimator = linear_model.LogisticRegression(random_state=101),
                 data_handler: DataHandler = DataHandler(),
                 persistence: ModelPersistence = ModelPersistence(),
                 trainer: ModelTrainer = ModelTrainer()):
        self.model = model
        self.data_handler = data_handler
        self.persistence = persistence
        self.trainer = trainer

    @log_time
    def predict(self, features: List[float]) -> PredictionOutput:
        features_array = np.array(features).reshape(1, -1)
        prediction = int(self.model.predict(features_array)[0])
        if not hasattr(self.model, 'predict_proba'):
            model_logger.error("Model does not have predict_proba method")
            raise ValueError("Model does not have predict_proba method")
        probability = float(self.model.predict_proba(features_array).max())
        model_logger.info(f"Prediction made || Input: {features} || Prediction: {prediction} || Probability: {probability:.4f}")
        return PredictionOutput(prediction=prediction, probability=probability)

    def load_or_train_model(self) -> None:
        model_info = self.persistence.load_model_info()
        if model_info and os.path.exists(model_info.path):
            self.model = self.persistence.load_model(model_info)
            model_logger.info("Model loaded successfully")
        else:
            model_logger.info("No existing model found. Training a new model.")
            accuracy, self.model = self.trainer.train_model(self.model, self.data_handler)
            model_info = self.persistence.save_model(self.model, accuracy, 1)
            self.persistence.save_model_info(model_info)
            self.save_model_metadata(model_info)

    def challenger_process(self) -> Tuple[bool, float]:
        model_logger.info("Starting challenger process")
        current_model_info = self.persistence.load_model_info()
        if not current_model_info:
            return self._handle_no_existing_model()

        return self._compare_and_update_model(current_model_info)

    def _handle_no_existing_model(self) -> Tuple[bool, float]:
        model_logger.info("No existing model found. Training a new model.")
        accuracy, self.model = self.trainer.train_model(self.model, self.data_handler)
        model_info = self.persistence.save_model(self.model, accuracy, 1)
        self.persistence.save_model_info(model_info)
        self.save_model_metadata(model_info)
        return True, accuracy

    def _compare_and_update_model(self, current_model_info: ModelInfo) -> Tuple[bool, float]:
        current_accuracy = current_model_info.accuracy
        new_accuracy, new_model = self.trainer.train_model(self.model, self.data_handler)

        if new_accuracy > current_accuracy * (1 + IMPROVEMENT_THRESHOLD):
            return self._update_model(new_accuracy, new_model, current_model_info.version)
        else:
            model_logger.info(f"Current model remains champion || New acc: {new_accuracy:.4f} || Current acc: {current_accuracy:.4f}")
            return False, current_accuracy

    def _update_model(self, new_accuracy: float, new_model: BaseEstimator, current_version: int) -> Tuple[bool, float]:
        model_logger.info(f"New model outperforms current model || New acc: {new_accuracy:.4f} || Prev acc: {current_version:.4f}")
        new_version = current_version + 1
        model_info = self.persistence.save_model(new_model, new_accuracy, new_version)
        self.persistence.save_model_info(model_info)
        self.save_model_metadata(model_info)
        self.model = new_model
        return True, new_accuracy

    def save_model_metadata(self, model_info: ModelInfo) -> None:
        metadata = self.get_model_metadata()
        self.persistence.save_model_metadata(model_info, metadata)

    def get_model_metadata(self) -> ModelMetadata:
        return ModelMetadata(
            model_type=type(self.model).__name__,
            feature_importance=self._get_feature_importance(),
            model_parameters=self.model.get_params(),
        )

    def _get_feature_importance(self) -> Optional[Dict[str, float]]:
        if hasattr(self.model, 'coef_'):
            feature_importance = dict(zip(datasets.load_iris().feature_names, self.model.coef_[0]))
            return {k: float(v) for k, v in feature_importance.items()}
        return None
