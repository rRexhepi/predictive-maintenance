"""Shared fixtures for the API test suite.

Trains a trivially small ``RandomForestRegressor`` + ``StandardScaler``
into a session ``tmp_path`` before any tests run and exposes their
paths as environment variables so ``src.api.main`` picks them up via
its own configuration hooks. Also snapshots a ``reference_stats.json``
so the drift monitor has a baseline at startup.
"""

from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

from src.monitoring import ReferenceStats


@pytest.fixture(scope="session")
def trained_artifacts(tmp_path_factory) -> dict[str, str]:
    rng = np.random.default_rng(0)
    X = rng.normal(loc=1.0, scale=0.5, size=(200, 3))
    y = X[:, 0] * 10 + X[:, 1] * 5 - X[:, 2] * 2 + rng.normal(scale=1.0, size=200)

    scaler = StandardScaler().fit(X)
    X_scaled = scaler.transform(X)
    model = RandomForestRegressor(n_estimators=20, random_state=0, n_jobs=1).fit(X_scaled, y)

    tmp_dir = tmp_path_factory.mktemp("artifacts")
    model_path = tmp_dir / "rf_model.pkl"
    scaler_path = tmp_dir / "scaler.pkl"
    ref_path = tmp_dir / "reference_stats.json"
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    ReferenceStats.fit(
        pd.DataFrame(X, columns=["feature1", "feature2", "feature3"])
    ).save(ref_path)

    return {
        "model_path": str(model_path),
        "scaler_path": str(scaler_path),
        "reference_stats_path": str(ref_path),
    }


@pytest.fixture
def api_client(trained_artifacts, monkeypatch):
    from fastapi.testclient import TestClient

    monkeypatch.setenv("MODEL_PATH", trained_artifacts["model_path"])
    monkeypatch.setenv("SCALER_PATH", trained_artifacts["scaler_path"])
    monkeypatch.setenv("REFERENCE_STATS_PATH", trained_artifacts["reference_stats_path"])
    monkeypatch.setenv("API_KEY", "test-key-abc")

    # Reload so module-level constants capture the test env vars.
    import importlib

    import src.api.main as api_main

    importlib.reload(api_main)

    with TestClient(api_main.app) as client:
        yield client


@pytest.fixture
def api_key() -> str:
    return "test-key-abc"
