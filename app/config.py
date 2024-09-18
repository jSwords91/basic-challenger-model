from pydantic import BaseModel

class Settings(BaseModel):
    app_name: str = "Iris Classifier"
    app_description: str = "Example ML project with CI/CD pipeline"
    model_path: str = "models/model.joblib"
    model_info_path: str = "models/model_info.json"


settings = Settings()