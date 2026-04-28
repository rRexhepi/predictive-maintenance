"""Train the RUL RandomForest and publish to MLflow.

Every run produces:

1. A single ``pyfunc`` artifact (scaler + estimator wrapped as
   :class:`~src.mlflow_model.PredictiveMaintenanceModel`) logged to the
   current run and registered under ``REGISTERED_MODEL_NAME``. New
   versions get the ``candidate`` alias. ``--promote`` also moves the
   ``production`` alias.
2. ``models/rf_model.pkl`` on disk for the Dockerfile / filesystem
   serving fallback.

Tracking URI and joblib fan-out both honour env-vars so CI and local
dev can opt into different backends without touching the code:

* ``MLFLOW_TRACKING_URI`` defaults to ``./mlruns`` file backend (no
  server required). Point at ``http://localhost:5001`` to use a
  long-running tracking server + registry DB.
* ``PM_N_JOBS`` defaults to ``2``. Caps sklearn's joblib worker count.
  ``-1`` (one per core) can fork-bomb a laptop.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Invoked as `python src/train_model.py`, only `src/` is on sys.path, so
# `from src.X import ...` fails. The API + tests already run with the
# repo root on the path (Dockerfile sets PYTHONPATH, pytest reads
# pyproject.toml). Prepend it here so the training entry point has the
# same guarantee without forcing callers to export PYTHONPATH.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import argparse
import os

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow import MlflowClient
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.mlflow_model import log_and_register
from src.monitoring import MONITORED_FEATURES, ReferenceStats

TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns")
N_JOBS = int(os.getenv("PM_N_JOBS", "2"))

MODELS_DIR = Path("models")
SCALER_PATH = MODELS_DIR / "scaler.pkl"
ESTIMATOR_PATH = MODELS_DIR / "rf_model.pkl"
REFERENCE_STATS_PATH = MODELS_DIR / "reference_stats.json"
REGISTERED_MODEL_NAME = "predictive-maintenance-rul"
CANDIDATE_ALIAS = "candidate"
PRODUCTION_ALIAS = "production"


def load_preprocessed_data():
    """Load the preprocessed training and validation data."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, "..", "data")

    train_path = os.path.join(data_dir, "train_preprocessed.csv")
    val_path = os.path.join(data_dir, "val_preprocessed.csv")
    if not os.path.isfile(train_path):
        raise FileNotFoundError(f"Training data not found at {train_path}")
    if not os.path.isfile(val_path):
        raise FileNotFoundError(f"Validation data not found at {val_path}")

    train_data = pd.read_csv(train_path)
    val_data = pd.read_csv(val_path)

    X_train = train_data.drop("RUL", axis=1)
    y_train = train_data["RUL"]
    X_val = val_data.drop("RUL", axis=1)
    y_val = val_data["RUL"]
    return X_train, y_train, X_val, y_val


def train_and_evaluate_model(X_train, y_train, X_val, y_val, *, promote: bool = False):
    """Train a RandomForest, log to MLflow, register as a pyfunc."""
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment("Predictive Maintenance Model Training")

    with mlflow.start_run() as run:
        params = {
            "n_estimators": 100,
            "max_depth": None,
            "min_samples_split": 2,
            "random_state": 42,
            "n_jobs": N_JOBS,
        }
        mlflow.log_params(params)

        model = RandomForestRegressor(**params)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_val)
        rmse = float(np.sqrt(mean_squared_error(y_val, y_pred)))
        mae = float(mean_absolute_error(y_val, y_pred))
        r2 = float(model.score(X_val, y_val))

        mlflow.log_metrics({"rmse": rmse, "mae": mae, "r2": r2})
        print(f"Validation RMSE: {rmse:.2f}")
        print(f"Validation MAE:  {mae:.2f}")
        print(f"Validation R2:   {r2:.3f}")

        # Filesystem estimator (kept for Dockerfile + filesystem fallback).
        MODELS_DIR.mkdir(exist_ok=True)
        joblib.dump(model, ESTIMATOR_PATH)

        # Reference-distribution snapshot consumed by the live drift
        # monitor. Matches feature names the API exposes.
        feature_frame = X_train[list(MONITORED_FEATURES)] if all(
            f in X_train.columns for f in MONITORED_FEATURES
        ) else X_train
        ReferenceStats.fit(feature_frame).save(REFERENCE_STATS_PATH)
        mlflow.log_artifact(str(REFERENCE_STATS_PATH))

        # Register a single pyfunc (scaler + estimator) so serving can
        # resolve it by URI instead of loading two pickles by path.
        if not SCALER_PATH.exists():
            raise FileNotFoundError(
                f"Scaler missing at {SCALER_PATH}. Run `python src/preprocessing.py` first."
            )
        info = log_and_register(
            scaler_path=SCALER_PATH,
            estimator_path=ESTIMATOR_PATH,
            registered_model_name=REGISTERED_MODEL_NAME,
        )

        client = MlflowClient()
        version = info.registered_model_version
        client.set_registered_model_alias(
            name=REGISTERED_MODEL_NAME, alias=CANDIDATE_ALIAS, version=version
        )
        print(
            f"Registered {REGISTERED_MODEL_NAME} v{version} "
            f"(run {run.info.run_id}) with alias @{CANDIDATE_ALIAS}."
        )
        if promote:
            client.set_registered_model_alias(
                name=REGISTERED_MODEL_NAME, alias=PRODUCTION_ALIAS, version=version
            )
            print(f"Promoted {REGISTERED_MODEL_NAME} v{version} to @{PRODUCTION_ALIAS}.")

    return model, {"rmse": rmse, "mae": mae, "r2": r2}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the RUL model and publish to MLflow.")
    parser.add_argument(
        "--promote",
        action="store_true",
        help="Also set the `production` alias on this new version.",
    )
    args = parser.parse_args()

    X_train, y_train, X_val, y_val = load_preprocessed_data()
    train_and_evaluate_model(X_train, y_train, X_val, y_val, promote=args.promote)


if __name__ == "__main__":
    main()
