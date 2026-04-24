"""FastAPI serving layer for the Predictive Maintenance model.

Request flow:

1. Pydantic parses + validates the JSON body (via the ``PredictRequest`` /
   ``BatchPredictRequest`` schemas in ``schemas.py``).
2. The body is turned into a DataFrame **in memory**.
3. :func:`~src.api.utils.clean_dataframe` drops/imputes as configured.
4. :func:`~src.api.utils.validate_data` enforces required columns.
5. The loaded ``PredictiveMaintenanceModel`` produces a prediction.

Previously every request wrote two CSVs to disk, read them back, and
deleted them. That version could not survive concurrent traffic — two
requests would race on the same temp filenames — and a CSV round-trip is
meaningless overhead for a handful of features. The new path is
single-process, stateless, and returns in sub-millisecond wallclock
before any future ``/metrics`` histogram kicks in.

Startup + shutdown are wired through FastAPI's modern ``lifespan``
context manager. The older ``@app.on_event`` decorators are deprecated
upstream and were flagging ``DeprecationWarning`` on every boot.
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from typing import Optional

import pandas as pd
from fastapi import FastAPI, HTTPException, Security
from fastapi.security.api_key import APIKeyHeader

from src.api.model import PredictiveMaintenanceModel
from src.api.schemas import (
    BatchPredictRequest,
    BatchPredictResponse,
    PredictRequest,
    PredictResponse,
)
from src.api.utils import clean_dataframe, validate_data

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("prediction.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

MODEL_PATH = os.getenv("MODEL_PATH", "models/rf_model.pkl")
SCALER_PATH = os.getenv("SCALER_PATH", "models/scaler.pkl")

API_KEY = os.getenv("API_KEY")
API_KEY_NAME = "access_token"

api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

# Column names we insist on receiving. Kept in sync with PredictRequest;
# update both when the feature contract changes.
REQUIRED_FEATURES = ["feature1", "feature2", "feature3"]

# Populated by the lifespan hook; unit tests swap this via env vars +
# module reload.
_state: dict = {}


@asynccontextmanager
async def lifespan(_: FastAPI):
    if not API_KEY:
        logger.warning(
            "API_KEY is unset — the auth check will reject every request. "
            "Set API_KEY in the environment to enable the API."
        )

    scaler = SCALER_PATH if os.path.exists(SCALER_PATH) else None
    try:
        _state["model"] = PredictiveMaintenanceModel(
            model_path=MODEL_PATH, scaler_path=scaler
        )
        logger.info("Model loaded from %s (scaler=%s).", MODEL_PATH, scaler)
    except FileNotFoundError as e:
        # Surface a clear error at boot instead of silently serving 500s.
        logger.error("Model artifact missing: %s", e)
        raise
    yield
    _state.clear()
    logger.info("Predictive Maintenance API shutting down.")


app = FastAPI(
    title="Predictive Maintenance API",
    description="RUL predictions served from a trained RandomForest estimator.",
    version="1.1.0",
    lifespan=lifespan,
)


def get_api_key(api_key: Optional[str] = Security(api_key_header)) -> str:
    """Reject the request unless the header matches the configured API key."""
    if not API_KEY or api_key != API_KEY:
        logger.warning("Unauthorized request (missing or mismatched API key).")
        raise HTTPException(status_code=403, detail="Could not validate credentials.")
    return api_key


def _predict_frame(df: pd.DataFrame) -> list[float]:
    model: PredictiveMaintenanceModel = _state["model"]
    cleaned = clean_dataframe(df, handle_missing="impute", impute_strategy="median")
    validate_data(cleaned, REQUIRED_FEATURES)
    # ``PredictiveMaintenanceModel.predict`` returns a scalar for a one-row
    # frame (its ``[0]`` indexing) and an array otherwise, so we normalise
    # here and let the batch vs. single endpoints slice as they need.
    preds = model.model.predict(model.preprocess(cleaned))
    return [float(p) for p in preds]


@app.post(
    "/predict",
    response_model=PredictResponse,
    dependencies=[Security(get_api_key)],
)
def predict(request: PredictRequest) -> PredictResponse:
    if "model" not in _state:
        raise HTTPException(status_code=500, detail="Model not loaded.")
    df = pd.DataFrame([request.model_dump()])
    try:
        preds = _predict_frame(df)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    return PredictResponse(predicted_rul=preds[0])


@app.post(
    "/batch_predict",
    response_model=BatchPredictResponse,
    dependencies=[Security(get_api_key)],
)
def batch_predict(request: BatchPredictRequest) -> BatchPredictResponse:
    if "model" not in _state:
        raise HTTPException(status_code=500, detail="Model not loaded.")
    if not request.data:
        raise HTTPException(status_code=400, detail="Empty batch.")
    df = pd.DataFrame([item.model_dump() for item in request.data])
    try:
        preds = _predict_frame(df)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    return BatchPredictResponse(predictions=preds)


@app.get("/health")
def health() -> dict:
    """Liveness probe. Returns 200 as long as the app is serving."""
    return {"status": "ok", "model_loaded": "model" in _state}


@app.get("/", include_in_schema=False)
def root() -> dict:
    return {"message": "Predictive Maintenance API. See /docs for the schema."}
