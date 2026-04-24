"""Round-trip test for the MLflow pyfunc wrapper.

Save the scaler + estimator as a single pyfunc via MLflow, load it back
via ``mlflow.pyfunc.load_model``, and assert predictions on raw rows
match what the unwrapped estimator gives. This pins the property that
matters: the served artifact can't drift apart from its scaler.
"""

from __future__ import annotations

import joblib
import mlflow
import mlflow.pyfunc
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

from src.mlflow_model import PredictiveMaintenanceModel


@pytest.fixture
def toy_artifacts(tmp_path):
    rng = np.random.default_rng(0)
    X = rng.normal(loc=1.0, scale=0.5, size=(100, 3))
    y = X @ [5.0, 3.0, -1.0] + rng.normal(scale=0.5, size=100)

    scaler = StandardScaler().fit(X)
    estimator = RandomForestRegressor(n_estimators=15, random_state=0, n_jobs=1).fit(
        scaler.transform(X), y
    )

    scaler_path = tmp_path / "scaler.pkl"
    estimator_path = tmp_path / "rf.pkl"
    joblib.dump(scaler, scaler_path)
    joblib.dump(estimator, estimator_path)
    return {
        "scaler": scaler,
        "estimator": estimator,
        "scaler_path": scaler_path,
        "estimator_path": estimator_path,
    }


def test_pyfunc_round_trip_matches_raw_estimator(toy_artifacts, tmp_path, monkeypatch):
    tracking_uri = (tmp_path / "mlruns").as_uri()
    monkeypatch.setenv("MLFLOW_TRACKING_URI", tracking_uri)
    mlflow.set_tracking_uri(tracking_uri)

    with mlflow.start_run() as run:
        info = mlflow.pyfunc.log_model(
            artifact_path="model",
            python_model=PredictiveMaintenanceModel(),
            artifacts={
                "scaler": str(toy_artifacts["scaler_path"]),
                "estimator": str(toy_artifacts["estimator_path"]),
            },
        )

    loaded = mlflow.pyfunc.load_model(info.model_uri)

    frame = pd.DataFrame(
        [
            {"feature1": 0.5, "feature2": 1.2, "feature3": 3.4},
            {"feature1": 2.0, "feature2": 0.1, "feature3": 0.0},
        ]
    )
    pyfunc_preds = loaded.predict(frame).tolist()
    direct_preds = toy_artifacts["estimator"].predict(
        toy_artifacts["scaler"].transform(frame)
    ).tolist()

    assert pyfunc_preds == direct_preds
    assert run.info.run_id
