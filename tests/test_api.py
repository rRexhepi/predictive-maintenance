"""API behaviour tests.

Pin the contract we actually ship:

* ``/health`` is unauthenticated and returns 200 + ``model_loaded: true``
  once the lifespan hook has loaded the artifacts.
* ``/predict`` and ``/batch_predict`` require the ``access_token`` header.
  Missing or wrong key → 403. Bad payload → 422. Good payload → 200 with
  the right response shape.
* Nothing touches the filesystem per request — the old code wrote two
  CSVs for every call. This test's existence is only possible because
  that pattern is gone.
"""

from __future__ import annotations


def _valid_payload() -> dict:
    return {"feature1": 0.5, "feature2": 1.2, "feature3": 3.4}


def test_health_is_unauthenticated(api_client):
    r = api_client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    # The filesystem predictor reports its source path so on-call can
    # tell at a glance which model is actually loaded.
    assert body["source"].startswith("filesystem:")


def test_predict_happy_path(api_client, api_key):
    r = api_client.post(
        "/predict",
        headers={"access_token": api_key},
        json=_valid_payload(),
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert "predicted_rul" in body
    assert isinstance(body["predicted_rul"], float)


def test_predict_rejects_missing_api_key(api_client):
    r = api_client.post("/predict", json=_valid_payload())
    assert r.status_code == 403


def test_predict_rejects_wrong_api_key(api_client):
    r = api_client.post(
        "/predict",
        headers={"access_token": "definitely-not-the-key"},
        json=_valid_payload(),
    )
    assert r.status_code == 403


def test_predict_rejects_negative_feature(api_client, api_key):
    payload = _valid_payload() | {"feature1": -1.0}
    r = api_client.post(
        "/predict",
        headers={"access_token": api_key},
        json=payload,
    )
    # Pydantic v2 rejects ge=0 violations at parse time → 422.
    assert r.status_code == 422


def test_predict_rejects_missing_field(api_client, api_key):
    payload = {"feature1": 0.5, "feature2": 1.2}  # missing feature3
    r = api_client.post(
        "/predict",
        headers={"access_token": api_key},
        json=payload,
    )
    assert r.status_code == 422


def test_batch_predict_happy_path(api_client, api_key):
    r = api_client.post(
        "/batch_predict",
        headers={"access_token": api_key},
        json={"data": [_valid_payload(), _valid_payload() | {"feature1": 2.0}]},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert "predictions" in body
    assert len(body["predictions"]) == 2
    assert all(isinstance(p, float) for p in body["predictions"])


def test_batch_predict_rejects_empty_batch(api_client, api_key):
    r = api_client.post(
        "/batch_predict",
        headers={"access_token": api_key},
        json={"data": []},
    )
    assert r.status_code == 400


def test_predict_does_not_write_temp_files(api_client, api_key, tmp_path, monkeypatch):
    """Regression test: the old handler created temp_input.csv / temp_clean_data.csv.

    Run from an empty working directory and assert none appear after a
    prediction request. If this ever fails, someone re-introduced the
    disk-round-trip pattern.
    """
    monkeypatch.chdir(tmp_path)
    api_client.post(
        "/predict",
        headers={"access_token": api_key},
        json=_valid_payload(),
    )
    assert not (tmp_path / "temp_input.csv").exists()
    assert not (tmp_path / "temp_clean_data.csv").exists()
    assert not (tmp_path / "temp_batch_input.csv").exists()
    assert not (tmp_path / "temp_batch_clean_data.csv").exists()
