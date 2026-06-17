"""Tests for production data simulation used in drift monitoring.

These tests verify that simulated data preserves dataset structure while
applying configurable shifts that should trigger drift detection.
"""

import numpy as np
import pandas as pd
import pytest

from src.monitoring.drift_monitoring import build_report, drift_summary
from src.monitoring.simulate_production import simulate_production_data

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
def base_df():
    rng = np.random.default_rng(0)
    return pd.DataFrame({c: rng.uniform(1, 5, 200) for c in FEATURES})


def test_simulation_preserves_shape_and_columns(base_df):
    """Vérifie que la simulation conserve la forme et les colonnes du dataset."""
    sim = simulate_production_data(base_df)
    assert sim.shape == base_df.shape
    assert list(sim.columns) == list(base_df.columns)


def test_income_is_inflated(base_df):
    """Vérifie que l'inflation du revenu est appliquée par la simulation."""
    sim = simulate_production_data(base_df, income_inflation=0.20)
    assert sim["MedInc"].mean() > base_df["MedInc"].mean()


def test_geographic_shift_applied(base_df):
    """Vérifie que le décalage géographique est appliqué aux coordonnées."""
    sim = simulate_production_data(base_df, geo_shift=1.5)
    assert sim["Latitude"].mean() > base_df["Latitude"].mean()
    assert sim["Longitude"].mean() < base_df["Longitude"].mean()


def test_simulated_data_triggers_drift(base_df):
    """Vérifie que les données simulées déclenchent une détection de dérive."""
    sim = simulate_production_data(base_df)
    summary = drift_summary(build_report(base_df, sim))
    assert summary["count"] >= 1
