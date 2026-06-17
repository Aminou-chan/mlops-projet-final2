import os
import pickle
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression

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


def _make_frame(n=60, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "MedInc": rng.uniform(0.5, 15, n),
            "HouseAge": rng.uniform(1, 52, n),
            "AveRooms": rng.uniform(2, 10, n),
            "AveBedrms": rng.uniform(0.5, 2, n),
            "Population": rng.uniform(100, 3000, n),
            "AveOccup": rng.uniform(1, 5, n),
            "Latitude": rng.uniform(32, 42, n),
            "Longitude": rng.uniform(-124, -114, n),
        }
    )[FEATURES]


# Un modele factice doit exister AVANT l'import de src.api.main
# (main.py charge le modele au moment de l'import).
_X = _make_frame()
_y = _X["MedInc"] * 2 + 1
_api_model = LinearRegression().fit(_X, _y)
_tmp_model = Path(tempfile.gettempdir()) / "immoprix_fake_model.pkl"
with open(_tmp_model, "wb") as f:
    pickle.dump(_api_model, f)
os.environ["MODEL_PATH"] = str(_tmp_model)


@pytest.fixture
def sample_frame():
    return _make_frame()


@pytest.fixture
def tree_model():
    X = _make_frame(n=80, seed=1)
    y = X["MedInc"] * 2 + X["AveOccup"] - 0.5
    return GradientBoostingRegressor(n_estimators=30, random_state=42).fit(X, y)


@pytest.fixture
def tiny_processed(tmp_path, monkeypatch):
    """Mini dataset traite + arborescence, dans un dossier temporaire.
    Sert aux tests de train.py et des analyzers (lecture sur disque)."""
    from sklearn.model_selection import train_test_split

    monkeypatch.chdir(tmp_path)
    (tmp_path / "data" / "processed").mkdir(parents=True)
    (tmp_path / "models").mkdir()
    (tmp_path / "reports" / "figures").mkdir(parents=True)
    (tmp_path / "reports" / "explainability").mkdir(parents=True)

    X = _make_frame(n=40, seed=3)
    y = X["MedInc"] * 2 + 1
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=42)

    for name, df in [
        ("X_train_tree", X_tr),
        ("X_test_tree", X_te),
        ("X_train_linear", X_tr),
        ("X_test_linear", X_te),
    ]:
        df.to_csv(f"data/processed/{name}.csv", index=False)
    y_tr.to_csv("data/processed/y_train.csv", index=False)
    y_te.to_csv("data/processed/y_test.csv", index=False)
    return tmp_path
