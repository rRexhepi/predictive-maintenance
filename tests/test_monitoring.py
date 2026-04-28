"""Tests for the drift + Prometheus monitoring module."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src.monitoring import DriftMonitor, ReferenceStats, compute_psi, time_block


@pytest.fixture
def reference_frame() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "feature1": rng.normal(1.0, 0.5, size=500).clip(0, None),
            "feature2": rng.gamma(2.0, 1.0, size=500),
            "feature3": rng.normal(2.0, 0.7, size=500).clip(0, None),
        }
    )


def test_reference_stats_roundtrip(tmp_path, reference_frame):
    stats = ReferenceStats.fit(reference_frame)
    assert set(stats.bin_edges) == {"feature1", "feature2", "feature3"}
    path = tmp_path / "ref.json"
    stats.save(path)
    loaded = ReferenceStats.load(path)
    assert loaded.bin_edges == stats.bin_edges
    assert loaded.bin_counts == stats.bin_counts
    assert loaded.n_training_rows == stats.n_training_rows
    # Sanity: file is valid JSON with the expected top-level keys.
    parsed = json.loads(path.read_text())
    assert set(parsed) == {"bin_edges", "bin_counts", "n_training_rows"}


def test_reference_stats_skips_constant_feature():
    df = pd.DataFrame(
        {"feature1": [5.0] * 50, "feature2": np.linspace(0, 100, 50)}
    )
    stats = ReferenceStats.fit(df, features=("feature1", "feature2"))
    assert "feature1" not in stats.bin_edges  # degenerate, can't bin
    assert "feature2" in stats.bin_edges


def test_psi_zero_for_same_distribution(reference_frame):
    stats = ReferenceStats.fit(reference_frame)
    psi = compute_psi(
        stats.bin_counts["feature1"],
        reference_frame["feature1"].tolist(),
        stats.bin_edges["feature1"],
    )
    assert psi == pytest.approx(0.0, abs=0.05)


def test_psi_flags_shifted_distribution(reference_frame):
    stats = ReferenceStats.fit(reference_frame)
    rng = np.random.default_rng(1)
    # Training distribution was ~N(1.0, 0.5) clipped at 0, so its bin
    # edges span roughly 0–3. Shift the mean toward the upper tail while
    # keeping enough support inside the reference span, otherwise every
    # actual value lands outside every bin and PSI is undefined.
    shifted = rng.normal(2.5, 0.4, size=500).tolist()
    psi = compute_psi(
        stats.bin_counts["feature1"],
        shifted,
        stats.bin_edges["feature1"],
    )
    assert psi > 0.25  # significant-drift threshold


def test_drift_monitor_records_and_renders(reference_frame):
    stats = ReferenceStats.fit(reference_frame)
    monitor = DriftMonitor(stats, buffer_size=100)

    inputs = reference_frame.sample(n=20, random_state=0).reset_index(drop=True)
    with time_block() as timer:
        pass
    monitor.record_prediction(
        inputs,
        predictions=[float(x) for x in np.linspace(20, 100, 20)],
        latency_seconds=timer.elapsed,
    )

    body, content_type = monitor.render()
    text = body.decode()

    assert "text/plain" in content_type
    assert "pm_predictions_total" in text
    assert "pm_prediction_latency_seconds" in text
    assert "pm_predicted_rul_bucket" in text
    assert "pm_feature_drift_psi" in text
    assert 'pm_input_buffer_size{feature="feature1"} 20.0' in text


def test_drift_monitor_without_reference_still_serves_metrics():
    monitor = DriftMonitor(reference=None)
    monitor.record_prediction(
        pd.DataFrame({"feature1": [1.0], "feature2": [2.0], "feature3": [3.0]}),
        predictions=[15.2],
        latency_seconds=0.01,
    )
    body, _ = monitor.render()
    text = body.decode()
    assert "pm_predictions_total" in text
    # No reference → no drift gauge values.
    assert 'pm_feature_drift_psi{feature=' not in text
