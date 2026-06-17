import numpy as np
import pandas as pd
import pytest

from src.features.preprocessing import TARGET, preprocess_data


@pytest.fixture
def raw_csv(tmp_path, monkeypatch):
    """Cree un mini dataset brut + l'arborescence, dans un dossier temporaire."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data" / "raw").mkdir(parents=True)
    (tmp_path / "data" / "processed").mkdir(parents=True)
    (tmp_path / "models").mkdir()

    n = 50
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            "MedInc": rng.uniform(0.5, 15, n),
            "HouseAge": rng.uniform(1, 52, n),
            "AveRooms": rng.uniform(2, 10, n),
            "AveBedrms": rng.uniform(0.5, 2, n),
            "Population": rng.uniform(100, 3000, n),
            "AveOccup": rng.uniform(1, 5, n),
            "Latitude": rng.uniform(32, 42, n),
            "Longitude": rng.uniform(-124, -114, n),
            TARGET: rng.uniform(0.5, 5, n),
        }
    )
    path = tmp_path / "data" / "raw" / "california_housing.csv"
    df.to_csv(path, index=False)
    return str(path)


def test_creates_all_files(raw_csv):
    import os

    preprocess_data(input_path=raw_csv)
    for f in [
        "data/processed/X_train_linear.csv",
        "data/processed/X_test_linear.csv",
        "data/processed/X_train_tree.csv",
        "data/processed/X_test_tree.csv",
        "data/processed/y_train.csv",
        "data/processed/y_test.csv",
        "models/robust_scaler.pkl",
    ]:
        assert os.path.exists(f), f"fichier manquant : {f}"


def test_split_sizes_and_no_nan(raw_csv):
    preprocess_data(input_path=raw_csv)
    xtr = pd.read_csv("data/processed/X_train_tree.csv")
    xte = pd.read_csv("data/processed/X_test_tree.csv")
    assert len(xtr) + len(xte) == 50
    assert len(xte) == 10  # 20 % de 50
    lin = pd.read_csv("data/processed/X_train_linear.csv")
    assert not lin.isna().any().any()  # aucune valeur manquante apres transfo


def test_tree_raw_vs_linear_transformed(raw_csv):
    preprocess_data(input_path=raw_csv)
    tree = pd.read_csv("data/processed/X_train_tree.csv")
    lin = pd.read_csv("data/processed/X_train_linear.csv")
    # memes colonnes dans les deux pipelines
    assert list(tree.columns) == list(lin.columns)
    # le tree est brut, le linear est log + scale -> valeurs differentes
    assert not np.allclose(tree["AveRooms"].values, lin["AveRooms"].values)
