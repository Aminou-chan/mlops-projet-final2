import contextlib
import os

import pytest
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

import src.models.train as train
from src.models.train import ModelConfig, ModelTrainer, get_models


def test_get_models_structure():
    models = get_models()
    assert [m.name for m in models] == [
        "linear_regression",
        "random_forest",
        "gradient_boosting",
    ]
    assert models[0].param_distributions == {}
    assert models[1].param_distributions
    assert models[2].param_distributions


def test_pipeline_property():
    lin = ModelConfig("x", LinearRegression(), features="linear")
    tree = ModelConfig("y", LinearRegression(), features="tree")
    assert lin.pipeline == "linear_pipeline"
    assert tree.pipeline == "tree_pipeline"


def test_update_best_picks_lowest_rmse():
    t = ModelTrainer.__new__(ModelTrainer)
    t.best = None
    t.best_model = None
    t._update_best("model_A", {"rmse": 0.8})
    assert t.best_model == "model_A"
    t._update_best("model_B", {"rmse": 0.5})
    assert t.best_model == "model_B"
    t._update_best("model_C", {"rmse": 0.9})
    assert t.best_model == "model_B"
    assert t.best["rmse"] == 0.5


def _light_models():
    """Modeles legers : 1 sans recherche (fit direct) + 1 avec RandomizedSearchCV."""
    return [
        ModelConfig("linear_regression", LinearRegression(), features="tree"),
        ModelConfig(
            "small_rf",
            RandomForestRegressor(random_state=42),
            features="tree",
            param_distributions={"n_estimators": [5, 8], "max_depth": [2, 3]},
            n_iter=2,
        ),
    ]


class _FakeRun:
    class info:
        run_id = "fake_run"


@contextlib.contextmanager
def _fake_start_run(*args, **kwargs):
    yield _FakeRun()


@pytest.fixture
def mock_mlflow(monkeypatch):
    """Neutralise tous les appels MLflow + l'enregistrement du modele."""
    monkeypatch.setattr(train.mlflow, "set_tracking_uri", lambda *a, **k: None)
    monkeypatch.setattr(train.mlflow, "set_experiment", lambda *a, **k: None)
    monkeypatch.setattr(train.mlflow, "start_run", _fake_start_run)
    monkeypatch.setattr(train.mlflow, "log_param", lambda *a, **k: None)
    monkeypatch.setattr(train.mlflow, "log_params", lambda *a, **k: None)
    monkeypatch.setattr(train.mlflow, "log_metric", lambda *a, **k: None)
    monkeypatch.setattr(train.mlflow, "log_metrics", lambda *a, **k: None)
    monkeypatch.setattr(train.mlflow.sklearn, "log_model", lambda *a, **k: None)
    monkeypatch.setattr(train, "register_best_model", lambda run_id: "1")


def test_full_training_run(tiny_processed, mock_mlflow, monkeypatch):
    monkeypatch.setattr(train, "get_models", _light_models)

    trainer = ModelTrainer()
    trainer.run()

    # un meilleur modele a ete choisi et sauvegarde
    assert trainer.best is not None
    assert os.path.exists("models/best_model.pkl")
    assert os.path.exists("models/best_model_metadata.json")
    assert os.path.exists("reports/metrics.csv")


def test_train_and_track_models_wrapper(tiny_processed, mock_mlflow, monkeypatch):
    monkeypatch.setattr(train, "get_models", _light_models)
    train.train_and_track_models()
    assert os.path.exists("models/best_model.pkl")
