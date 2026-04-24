"""Request / response schemas for the Predictive Maintenance API.

The feature fields here (``feature1`` / ``feature2`` / ``feature3``) are
placeholder names carried over from the project's first iteration. The
real trained model consumes the full C-MAPSS sensor + lag feature set,
so the API contract is out of sync with the model until a follow-up PR
regenerates the schema from the training columns. Documented in the
README under "Known gaps".
"""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


class PredictRequest(BaseModel):
    """Features for a single prediction."""

    feature1: float = Field(..., ge=0.0, description="Placeholder feature #1 (non-negative).")
    feature2: float = Field(..., ge=0.0, description="Placeholder feature #2 (non-negative).")
    feature3: float = Field(..., ge=0.0, description="Placeholder feature #3 (non-negative).")

    @field_validator("feature1", "feature2", "feature3")
    @classmethod
    def _non_negative(cls, v: float) -> float:
        # ``ge=0.0`` on the Field already rejects negatives at parse time,
        # but this guards against subclass overrides that drop the bound.
        if v < 0:
            raise ValueError("features must be non-negative")
        return v


class PredictResponse(BaseModel):
    """Predicted Remaining Useful Life (RUL) for a single request."""

    predicted_rul: float = Field(..., description="Predicted Remaining Useful Life (RUL).")


class BatchPredictRequest(BaseModel):
    """Batched features — one ``PredictRequest`` per row."""

    data: list[PredictRequest] = Field(..., description="List of feature rows to score.")


class BatchPredictResponse(BaseModel):
    """Predicted RULs, one per input row, in request order."""

    predictions: list[float] = Field(
        ..., description="Predicted RULs corresponding to each input row."
    )
