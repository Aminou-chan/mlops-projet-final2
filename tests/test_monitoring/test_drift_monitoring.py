"""Tests for data drift monitoring utilities.

Validate drift detection behavior, report generation, and summary extraction
for reference and production datasets.
"""

import numpy as np
import pandas as pd
import pytest

from src.monitoring.drift_monitoring import (
    build_report,
    drift_summary,
    generate_drift_report,
)

FEATURES = [
    "MedInc",
    "HouseAge",
    "AveRooms",
    "AveBedrms",
    "Population",
    "AveOccup",
    "Latitude",
    "Longitude",
]


@pytest.fixture
def reference_df():
    rng = np.random.default_rng(0)
    return pd.DataFrame({c: rng.normal(5, 1, 200) for c in FEATURES})


def test_no_drift_on_identical_data(reference_df):
    """Aucune dérive ne doit être détectée lorsque les données sont identiques."""
    summary = drift_summary(build_report(reference_df, reference_df.copy()))
    assert summary["count"] == 0


def test_drift_detected_on_shifted_data(reference_df):
    """Une dérive doit être détectée si la distribution du revenu change."""
    shifted = reference_df.copy()
    shifted["MedInc"] = shifted["MedInc"] + 8  # décalage net
    summary = drift_summary(build_report(reference_df, shifted))
    assert summary["count"] >= 1


def test_generate_report_creates_html(reference_df, tmp_path):
    """Vérifie que le rapport de dérive est généré au format HTML."""
    out = tmp_path / "drift.html"
    generate_drift_report(reference_df, reference_df.copy(), output_path=str(out))
    assert out.exists()
    assert out.stat().st_size > 0
