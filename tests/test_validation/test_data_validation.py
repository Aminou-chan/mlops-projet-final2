"""Tests for raw data validation using Great Expectations results.

These unit tests confirm that valid data passes, invalid values fail,
and human-readable validation summary output is produced.
"""

import numpy as np
import pandas as pd
import pytest

from src.validation.data_validation import validate_dataframe


@pytest.fixture
def conforming_df():
    rng = np.random.default_rng(0)
    n = 100
    return pd.DataFrame(
        {
            "MedInc": rng.uniform(0.5, 15, n),
            "HouseAge": rng.uniform(1, 52, n),
            "AveRooms": rng.uniform(2, 10, n),
            "AveBedrms": rng.uniform(0.5, 2, n),
            "Population": rng.uniform(100, 3000, n),
            "AveOccup": rng.uniform(1, 6, n),
            "Latitude": rng.uniform(32, 42, n),
            "Longitude": rng.uniform(-124, -114, n),
            "MedHouseVal": rng.uniform(0.2, 5, n),
        }
    )


def test_valid_data_passes(conforming_df):
    """Vérifie qu'un DataFrame conforme passe la validation."""
    result = validate_dataframe(conforming_df)
    assert result.success is True


def test_out_of_range_fails(conforming_df):
    """Vérifie qu'une valeur hors plage géographique est rejetée."""
    df = conforming_df.copy()
    df.loc[0, "Latitude"] = 99.0  # hors plage
    result = validate_dataframe(df)
    assert result.success is False


def test_null_value_fails(conforming_df):
    """Vérifie qu'une valeur manquante dans les données est détectée."""
    df = conforming_df.copy()
    df.loc[0, "MedInc"] = np.nan  # valeur manquante
    result = validate_dataframe(df)
    assert result.success is False


def test_summarize_runs(conforming_df, capsys):
    """Vérifie que la fonction de résumé affiche un message de validation globale."""
    from src.validation.data_validation import summarize

    result = validate_dataframe(conforming_df)
    summarize(result)
    out = capsys.readouterr().out
    assert "Validation globale" in out
