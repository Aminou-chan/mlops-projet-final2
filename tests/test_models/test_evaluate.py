"""Tests des métriques d'évaluation de régression.

This module validates RMSE, MAE, and R2 calculations over deterministic
examples as well as randomized inputs using Hypothesis property-based tests.
"""

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from src.models.evaluate import evaluate_model


@pytest.mark.parametrize(
    "y_true, y_pred, expected_rmse, expected_mae",
    [
        ([1.0, 2.0, 3.0], [1.0, 2.0, 3.0], 0.0, 0.0),
        ([0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0], 1.0, 1.0),
        ([0.0, 0.0], [3.0, 4.0], np.sqrt(12.5), 3.5),
        ([1.0, 2.0, 3.0], [1.0, 2.0, 5.0], np.sqrt(4 / 3), 2 / 3),
    ],
)
def test_evaluate_known_cases(y_true, y_pred, expected_rmse, expected_mae):
    """Vérifie les métriques pour des cas déterministes connus."""
    metrics = evaluate_model(np.array(y_true), np.array(y_pred))
    assert metrics["rmse"] == pytest.approx(expected_rmse)
    assert metrics["mae"] == pytest.approx(expected_mae)


def test_evaluate_perfect_prediction_r2():
    """Si la prédiction est parfaite, le R2 doit être égal à 1."""
    y = np.array([1.0, 2.0, 3.0, 4.0])
    metrics = evaluate_model(y, y)
    assert metrics["r2"] == pytest.approx(1.0)


def test_evaluate_returns_three_floats():
    metrics = evaluate_model(np.array([1.0, 2.0]), np.array([1.5, 2.5]))
    assert set(metrics) == {"rmse", "mae", "r2"}
    assert all(isinstance(v, float) for v in metrics.values())


finite_floats = st.floats(
    min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False
)


@st.composite
def paired_arrays(draw):
    """Genere deux tableaux de meme longueur (y_true, y_pred)."""
    n = draw(st.integers(min_value=2, max_value=50))
    y_true = draw(st.lists(finite_floats, min_size=n, max_size=n))
    y_pred = draw(st.lists(finite_floats, min_size=n, max_size=n))
    return np.array(y_true), np.array(y_pred)


@settings(max_examples=200)
@given(arrays=paired_arrays())
def test_metrics_are_non_negative(arrays):
    """Quelles que soient les entrees, RMSE et MAE sont >= 0."""
    y_true, y_pred = arrays
    metrics = evaluate_model(y_true, y_pred)
    assert metrics["rmse"] >= 0.0
    assert metrics["mae"] >= 0.0


@settings(max_examples=200)
@given(arrays=paired_arrays())
def test_rmse_greater_or_equal_mae(arrays):
    """Propriete mathematique : RMSE >= MAE pour toute entree."""
    y_true, y_pred = arrays
    metrics = evaluate_model(y_true, y_pred)
    assert metrics["rmse"] >= metrics["mae"] - 1e-9


@settings(max_examples=100)
@given(values=st.lists(finite_floats, min_size=2, max_size=50))
def test_perfect_prediction_has_zero_error(values):
    """Si y_pred == y_true, alors RMSE = 0 et MAE = 0 (pour tout vecteur)."""
    y = np.array(values)
    metrics = evaluate_model(y, y)
    assert metrics["rmse"] == pytest.approx(0.0, abs=1e-9)
    assert metrics["mae"] == pytest.approx(0.0, abs=1e-9)
