"""Tests des utilitaires de chargement de données.

The data loader is tested with a mocked housing dataset fetch so no network
request is performed during the unit test.
"""

import types

import pandas as pd

import src.data.load_data as ld


def test_load_returns_dataframe(monkeypatch):
    """Mock the remote data fetch and verify a DataFrame is returned."""
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    monkeypatch.setattr(
        ld,
        "fetch_california_housing",
        lambda as_frame: types.SimpleNamespace(frame=df),
    )
    out = ld.load_california_housing()
    assert isinstance(out, pd.DataFrame)
    assert list(out.columns) == ["a", "b"]
