import json
import pickle
from pathlib import Path

import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

from src.explainability.global_analysis import GlobalFeatureAnalyzer
from src.explainability.local_analysis import LocalFeatureAnalyzer


@pytest.fixture
def workdir(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "reports" / "figures").mkdir(parents=True)
    (tmp_path / "reports" / "explainability").mkdir(parents=True)
    return tmp_path


def test_global_importance_sorted_and_saved(workdir, tree_model, sample_frame):
    a = GlobalFeatureAnalyzer.__new__(GlobalFeatureAnalyzer)
    a.model = tree_model
    a.metadata = pd.Series({"pipeline": "tree_pipeline"})
    a._load_features = lambda: sample_frame

    result = a.compute_global_importance()
    assert list(result["importance"]) == sorted(result["importance"], reverse=True)
    assert set(result["feature"]) == set(sample_frame.columns)
    assert Path("reports/figures/global_importance.png").exists()
    assert Path("reports/explainability/global_importance.csv").exists()


def test_global_importance_unsupported_model(workdir, sample_frame):
    a = GlobalFeatureAnalyzer.__new__(GlobalFeatureAnalyzer)
    a.model = object()
    a.metadata = pd.Series({"pipeline": "tree_pipeline"})
    a._load_features = lambda: sample_frame
    with pytest.raises(ValueError):
        a.compute_global_importance()


def _write_model(model, pipeline):
    """Enregistre un modèle et ses métadonnées dans l'arborescence de test."""
    with open("models/best_model.pkl", "wb") as f:
        pickle.dump(model, f)
    with open("models/best_model_metadata.json", "w", encoding="utf-8") as f:
        json.dump({"pipeline": pipeline, "model": "test"}, f)


def test_global_real_init_tree(tiny_processed, tree_model):
    """Teste l'initialisation réelle de l'analyzer global pour un modèle d'arbre."""
    _write_model(tree_model, "tree_pipeline")
    analyzer = GlobalFeatureAnalyzer()  # __init__ + _load_model + _load_metadata
    result = analyzer.compute_global_importance()  # branche feature_importances_
    assert len(result) == 8
    assert Path("reports/figures/global_importance.png").exists()


def test_global_real_init_linear_coef(tiny_processed, sample_frame):
    lin = LinearRegression().fit(sample_frame, sample_frame["MedInc"] * 2)
    _write_model(lin, "linear_pipeline")
    analyzer = GlobalFeatureAnalyzer()
    result = (
        analyzer.compute_global_importance()
    )  # branche coef_ + _load_features linear
    assert len(result) == 8


def test_local_shap_values_and_plot(workdir, tree_model, sample_frame, monkeypatch):
    small = sample_frame.head(20)
    monkeypatch.setattr(
        LocalFeatureAnalyzer, "_load_model", staticmethod(lambda: tree_model)
    )
    monkeypatch.setattr(
        LocalFeatureAnalyzer,
        "_load_metadata",
        staticmethod(lambda: pd.Series({"pipeline": "tree_pipeline"})),
    )
    monkeypatch.setattr(LocalFeatureAnalyzer, "_load_sample", lambda self: small)

    analyzer = LocalFeatureAnalyzer()
    assert len(analyzer.shap_values) == len(small)
    analyzer.plot_beeswarm()
    assert Path("reports/figures/shap_beeswarm.png").exists()


def test_local_real_init_and_explain(tiny_processed, tree_model):
    """Teste l'initialisation réelle du local analyzer et la visualisation SHAP."""
    _write_model(tree_model, "tree_pipeline")
    analyzer = (
        LocalFeatureAnalyzer()
    )  # init réelle : _load_model/_load_metadata/_load_sample + shap
    analyzer.plot_beeswarm()
    analyzer.explain_sample(0)  # waterfall d'un exemple
    analyzer.explain_samples(2)  # plusieurs waterfalls
    assert Path("reports/figures/shap_beeswarm.png").exists()
    assert Path("reports/figures/shap_local_example_0.png").exists()
