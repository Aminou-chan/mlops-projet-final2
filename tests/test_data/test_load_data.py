import types

import pandas as pd

import src.data.load_data as ld


def test_load_returns_dataframe(monkeypatch):
    """On mocke fetch_california_housing : pas de telechargement reseau."""
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    monkeypatch.setattr(
        ld,
        "fetch_california_housing",
        lambda as_frame: types.SimpleNamespace(frame=df),
    )
    out = ld.load_california_housing()
    assert isinstance(out, pd.DataFrame)
    assert list(out.columns) == ["a", "b"]
