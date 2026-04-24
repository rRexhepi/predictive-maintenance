"""MLflow ``pyfunc`` wrapper: scaler + RandomForest estimator as one artifact.

Serving previously loaded two pickles off disk — ``rf_model.pkl`` and
``scaler.pkl`` — and either could drift out of sync with the other. A
preprocessor trained on yesterday's data paired with today's estimator
is a silent correctness bug, not a loud crash.

Wrapping both into one :class:`mlflow.pyfunc.PythonModel` removes the
chance of mismatch: MLflow Registry promotes the whole unit, serving
loads it by URI, and there's exactly one version to track.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import mlflow.pyfunc
import pandas as pd


class PredictiveMaintenanceModel(mlflow.pyfunc.PythonModel):
    """Scaler + estimator as a single pyfunc.

    ``context.artifacts`` must contain ``scaler`` and ``estimator`` paths
    pointing at joblib pickles produced during training.
    """

    def load_context(self, context) -> None:
        self.scaler = joblib.load(context.artifacts["scaler"])
        self.estimator = joblib.load(context.artifacts["estimator"])

    def predict(self, context, model_input: pd.DataFrame, params: dict[str, Any] | None = None):
        scaled = self.scaler.transform(model_input)
        return self.estimator.predict(scaled)


def log_and_register(
    *,
    scaler_path: Path,
    estimator_path: Path,
    registered_model_name: str,
) -> mlflow.models.ModelInfo:
    """Log the pyfunc model to the current MLflow run and register it."""
    return mlflow.pyfunc.log_model(
        artifact_path="model",
        python_model=PredictiveMaintenanceModel(),
        artifacts={
            "scaler": str(scaler_path),
            "estimator": str(estimator_path),
        },
        registered_model_name=registered_model_name,
    )
