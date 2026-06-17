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
    sim = simulate_production_data(base_df)
    assert sim.shape == base_df.shape
    assert list(sim.columns) == list(base_df.columns)


def test_income_is_inflated(base_df):
    sim = simulate_production_data(base_df, income_inflation=0.20)
    # le revenu moyen simule doit etre nettement superieur
    assert sim["MedInc"].mean() > base_df["MedInc"].mean()


def test_geographic_shift_applied(base_df):
    sim = simulate_production_data(base_df, geo_shift=1.5)
    assert sim["Latitude"].mean() > base_df["Latitude"].mean()
    assert sim["Longitude"].mean() < base_df["Longitude"].mean()


def test_simulated_data_triggers_drift(base_df):
    sim = simulate_production_data(base_df)
    summary = drift_summary(build_report(base_df, sim))
    # la derive simulee doit etre detectee sur au moins une colonne
    assert summary["count"] >= 1
