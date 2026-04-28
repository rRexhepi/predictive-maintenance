"""FastAPI serving layer for the Predictive Maintenance model.

Two loading paths, picked in order:

1. **Model Registry URI** (``MODEL_URI`` env var, e.g.
   ``models:/predictive-maintenance-rul@production``).
2. **Filesystem pickles** (``MODEL_PATH`` + ``SCALER_PATH``). Used by CI,
   the Dockerfile's baked-in artifacts, and local dev before a Registry
   exists.

Both paths expose the same ``predict(df) -> np.ndarray`` interface so
nothing downstream of ``_load_predictor`` needs to know which one won.

The app also serves Prometheus ``/metrics``: per-call prediction
counter, prediction latency histogram, predicted-RUL distribution, and
PSI-based feature drift gauges. See :mod:`src.monitoring` for the drift
story.
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Protocol

import joblib
import mlflow.pyfunc
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Response, Security
from fastapi.security.api_key import APIKeyHeader

from src.api.schemas import (
    BatchPredictRequest,
    BatchPredictResponse,
    PredictRequest,
    PredictResponse,
)
from src.api.utils import clean_dataframe, validate_data
from src.monitoring import DriftMonitor, ReferenceStats, time_block

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("prediction.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

MODEL_URI = os.getenv("MODEL_URI")
MODEL_PATH = os.getenv("MODEL_PATH", "models/rf_model.pkl")
SCALER_PATH = os.getenv("SCALER_PATH", "models/scaler.pkl")
REFERENCE_STATS_PATH = Path(
    os.getenv("REFERENCE_STATS_PATH", "models/reference_stats.json")
)

API_KEY = os.getenv("API_KEY")
API_KEY_NAME = "access_token"

api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

REQUIRED_FEATURES = ["feature1", "feature2", "feature3"]

_state: dict = {}


class _Predictor(Protocol):
    source: str

    def predict(self, df: pd.DataFrame) -> np.ndarray: ...


class _FilesystemPredictor:
    def __init__(self, model_path: str, scaler_path: str):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model artifact missing at {model_path}")
        if not os.path.exists(scaler_path):
            raise FileNotFoundError(f"Scaler artifact missing at {scaler_path}")
        self._estimator = joblib.load(model_path)
        self._scaler = joblib.load(scaler_path)
        self.source = f"filesystem:{model_path}"

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        scaled = self._scaler.transform(df)
        return np.asarray(self._estimator.predict(scaled))


class _RegistryPredictor:
    def __init__(self, uri: str):
        self._model = mlflow.pyfunc.load_model(uri)
        self.source = f"registry:{uri}"

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return np.asarray(self._model.predict(df))


def _load_predictor() -> _Predictor:
    if MODEL_URI:
        logger.info("Loading model from registry URI: %s", MODEL_URI)
        return _RegistryPredictor(MODEL_URI)
    logger.info("Loading model from filesystem: %s", MODEL_PATH)
    return _FilesystemPredictor(MODEL_PATH, SCALER_PATH)


def _load_reference() -> ReferenceStats | None:
    if REFERENCE_STATS_PATH.exists():
        return ReferenceStats.load(REFERENCE_STATS_PATH)
    logger.warning(
        "Reference stats missing at %s; drift gauges will be empty.",
        REFERENCE_STATS_PATH,
    )
    return None


@asynccontextmanager
async def lifespan(_: FastAPI):
    if not API_KEY:
        logger.warning(
            "API_KEY is unset, the auth check will reject every request. "
            "Set API_KEY in the environment to enable the API."
        )

    try:
        predictor = _load_predictor()
        logger.info("Predictor ready (source=%s).", predictor.source)
    except FileNotFoundError as e:
        logger.error("Model artifact missing: %s", e)
        raise

    reference = _load_reference()
    monitor = DriftMonitor(reference)
    monitor.set_model_info(
        source=predictor.source,
        uri=MODEL_URI or "",
        path=str(MODEL_PATH) if not MODEL_URI else "",
    )
    _state["predictor"] = predictor
    _state["monitor"] = monitor
    yield
    _state.clear()
    logger.info("Predictive Maintenance API shutting down.")


app = FastAPI(
    title="Predictive Maintenance API",
    description="RUL predictions served from a trained RandomForest estimator.",
    version="1.3.0",
    lifespan=lifespan,
)


def get_api_key(api_key: str | None = Security(api_key_header)) -> str:
    if not API_KEY or api_key != API_KEY:
        logger.warning("Unauthorized request (missing or mismatched API key).")
        raise HTTPException(status_code=403, detail="Could not validate credentials.")
    return api_key


def _predict_frame(df: pd.DataFrame) -> list[float]:
    predictor: _Predictor = _state["predictor"]
    monitor: DriftMonitor = _state["monitor"]
    cleaned = clean_dataframe(df, handle_missing="impute", impute_strategy="median")
    validate_data(cleaned, REQUIRED_FEATURES)
    with time_block() as timer:
        preds = predictor.predict(cleaned)
    preds_list = [float(p) for p in preds]
    # The monitor only ever looks at the REQUIRED columns, no risk of
    # recording noise if the frame carries extra fields.
    monitor.record_prediction(cleaned[REQUIRED_FEATURES], preds_list, timer.elapsed)
    return preds_list


@app.post(
    "/predict",
    response_model=PredictResponse,
    dependencies=[Security(get_api_key)],
)
def predict(request: PredictRequest) -> PredictResponse:
    if "predictor" not in _state:
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
    if "predictor" not in _state:
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
    predictor = _state.get("predictor")
    return {
        "status": "ok",
        "model_loaded": predictor is not None,
        "source": predictor.source if predictor is not None else None,
    }


@app.get("/metrics")
def metrics() -> Response:
    monitor: DriftMonitor = _state["monitor"]
    body, content_type = monitor.render()
    return Response(content=body, media_type=content_type)


@app.get("/", include_in_schema=False)
def root() -> dict:
    return {"message": "Predictive Maintenance API. See /docs for the schema."}
