from pydantic import BaseModel, Field, validator
from typing import List

class PredictRequest(BaseModel):
    """
    Schema for single prediction request.
    Define all necessary features here.
    Replace 'feature1', 'feature2', etc., with actual feature names.
    """
    feature1: float = Field(..., ge=0.0, description="Description for feature1")
    feature2: float = Field(..., ge=0.0, description="Description for feature2")
    feature3: float = Field(..., ge=0.0, description="Description for feature3")

    @validator('*')
    def check_non_negative(cls, v, field):
        """
        Validator to ensure all features are non-negative.
        Modify or remove based on feature requirements.
        """
        if v < 0:
            raise ValueError(f"{field.name} must be non-negative")
        return v

class PredictResponse(BaseModel):
    """
    Schema for single prediction response.
    """
    predicted_rul: float = Field(..., description="Predicted Remaining Useful Life (RUL)")

class BatchPredictRequest(BaseModel):
    """
    Schema for batch prediction request.
    Contains a list of PredictRequest items.
    """
    data: List[PredictRequest] = Field(..., description="List of feature sets for batch prediction")

class BatchPredictResponse(BaseModel):
    """
    Schema for batch prediction response.
    Returns a list of predicted RULs corresponding to each input.
    """
    predictions: List[float] = Field(..., description="List of predicted Remaining Useful Life (RUL) values")
