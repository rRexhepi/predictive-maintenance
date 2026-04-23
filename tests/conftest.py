"""Shared fixtures for the API test suite.

Trains a trivially small ``RandomForestRegressor`` + ``StandardScaler``
into a ``tmp_path`` before each test module and exposes their paths as
environment variables so ``src.api.main`` picks them up via its own
configuration hooks. No real dataset needed — the API behaviour tests
we care about (authn, schema validation, error paths) are independent
of model quality.
"""

from __future__ import annotations

import joblib
import numpy as np
import pytest
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler


@pytest.fixture(scope="session")
def trained_artifacts(tmp_path_factory) -> dict[str, str]:
    """Fit a tiny scaler + estimator and write them to a session tmpdir."""
    rng = np.random.default_rng(0)
    # Three placeholder features to match src/api/schemas.py.
    X = rng.normal(loc=1.0, scale=0.5, size=(200, 3))
    y = X[:, 0] * 10 + X[:, 1] * 5 - X[:, 2] * 2 + rng.normal(scale=1.0, size=200)

    scaler = StandardScaler().fit(X)
    X_scaled = scaler.transform(X)
    model = RandomForestRegressor(n_estimators=20, random_state=0, n_jobs=1).fit(X_scaled, y)

    tmp_dir = tmp_path_factory.mktemp("artifacts")
    model_path = tmp_dir / "rf_model.pkl"
    scaler_path = tmp_dir / "scaler.pkl"
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)

    return {"model_path": str(model_path), "scaler_path": str(scaler_path)}


@pytest.fixture
def api_client(trained_artifacts, monkeypatch):
    """Boot the FastAPI app against the fixture artifacts + a known key."""
    from fastapi.testclient import TestClient

    monkeypatch.setenv("MODEL_PATH", trained_artifacts["model_path"])
    monkeypatch.setenv("SCALER_PATH", trained_artifacts["scaler_path"])
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
