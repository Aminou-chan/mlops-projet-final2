from dataclasses import dataclass, field
from pathlib import Path

import json
import pickle

import mlflow
import mlflow.sklearn
import pandas as pd

from scipy.stats import randint, uniform
from sklearn.base import RegressorMixin
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import RandomizedSearchCV

from src.models.evaluate import evaluate_model
from src.models.register import register_best_model


EXPERIMENT_NAME = "california_housing_regression"
TRACKING_URI = "sqlite:///mlflow.db"

CV_FOLDS = 5
SCORING = "neg_root_mean_squared_error"   # RandomizedSearchCV maximise -> on prend -RMSE
RANDOM_STATE = 42


@dataclass
class ModelConfig:
    """Un modele a tester + son espace de recherche d'hyperparametres."""

    name: str
    estimator: RegressorMixin
    features: str                       # "linear" ou "tree"
    param_distributions: dict = field(default_factory=dict)  # {} = rien a tuner
    n_iter: int = 10                    # nb de combinaisons tirees au hasard

    @property
    def pipeline(self) -> str:
        return "linear_pipeline" if self.features == "linear" else "tree_pipeline"


def get_models() -> list[ModelConfig]:
    """Les modeles a comparer. La regression lineaire n'a rien a tuner ;
    RF et GB sont explores par recherche aleatoire en cross-validation."""
    return [
        ModelConfig(
            name="linear_regression",
            estimator=LinearRegression(),
            features="linear",
        ),
        ModelConfig(
            name="random_forest",
            estimator=RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1),
            features="tree",
            param_distributions={
                "n_estimators": randint(100, 400),
                "max_depth": [10, 20, 30, None],
                "min_samples_split": randint(2, 10),
                "min_samples_leaf": randint(1, 5),
            },
            n_iter=10,
        ),
        ModelConfig(
            name="gradient_boosting",
            estimator=GradientBoostingRegressor(random_state=RANDOM_STATE),
            features="tree",
            param_distributions={
                "n_estimators": randint(100, 400),
                "learning_rate": uniform(0.01, 0.2),
                "max_depth": randint(2, 6),
                "subsample": uniform(0.7, 0.3),
            },
            n_iter=10,
        ),
    ]


class ModelTrainer:
    def __init__(self) -> None:
        Path("models").mkdir(exist_ok=True)
        Path("reports").mkdir(exist_ok=True)
        self._setup_mlflow()

        self.results: list[dict] = []
        self.best: dict | None = None
        self.best_model: RegressorMixin | None = None

    @staticmethod
    def _setup_mlflow() -> None:
        mlflow.set_tracking_uri(TRACKING_URI)
        mlflow.set_experiment(EXPERIMENT_NAME)

    @staticmethod
    def _load_targets() -> tuple[pd.Series, pd.Series]:
        y_train = pd.read_csv("data/processed/y_train.csv").squeeze()
        y_test = pd.read_csv("data/processed/y_test.csv").squeeze()
        return y_train, y_test

    @staticmethod
    def _load_features(features: str) -> tuple[pd.DataFrame, pd.DataFrame]:
        X_train = pd.read_csv(f"data/processed/X_train_{features}.csv")
        X_test = pd.read_csv(f"data/processed/X_test_{features}.csv")
        return X_train, X_test

    @staticmethod
    def _log_trials(config: ModelConfig, search: RandomizedSearchCV) -> None:
        """Logue chaque combinaison testee comme un run enfant dans MLflow."""
        params_list = search.cv_results_["params"]
        scores = search.cv_results_["mean_test_score"]
        for params, score in zip(params_list, scores):
            with mlflow.start_run(run_name=f"{config.name}_trial", nested=True):
                mlflow.log_params(params)
                mlflow.log_metric("cv_rmse", -score)

    def _fit_model(self, config, X_train, y_train):
        """Retourne (modele entraine, meilleurs params, rmse cross-validee)."""
        # Pas d'hyperparametres a chercher -> entrainement direct.
        if not config.param_distributions:
            config.estimator.fit(X_train, y_train)
            return config.estimator, config.estimator.get_params(), None

        search = RandomizedSearchCV(
            config.estimator,
            config.param_distributions,
            n_iter=config.n_iter,
            cv=CV_FOLDS,
            scoring=SCORING,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
        search.fit(X_train, y_train)
        self._log_trials(config, search)

        cv_rmse = float(-search.best_score_)
        return search.best_estimator_, search.best_params_, cv_rmse

    def _train_one(self, config: ModelConfig, y_train, y_test) -> None:
        X_train, X_test = self._load_features(config.features)

        with mlflow.start_run(run_name=config.name) as run:
            model, best_params, cv_rmse = self._fit_model(config, X_train, y_train)
            y_pred = model.predict(X_test)
            metrics = evaluate_model(y_test, y_pred)

            mlflow.log_param("model_name", config.name)
            mlflow.log_param("pipeline", config.pipeline)
            mlflow.log_params(best_params)
            if cv_rmse is not None:
                mlflow.log_metric("cv_rmse", cv_rmse)
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(model, name="model")

            run_info = {
                "run_id": run.info.run_id,
                "model": config.name,
                "pipeline": config.pipeline,
                "cv_rmse": cv_rmse,
                **metrics,
                # default=str : evite les soucis de serialisation (np.int64...)
                "best_params": json.dumps(best_params, default=str),
            }

        self.results.append(run_info)
        self._update_best(model, run_info)

        line = f"{config.name:20s} test RMSE = {metrics['rmse']:.4f}"
        if cv_rmse is not None:
            line += f" | CV RMSE = {cv_rmse:.4f}"
        print(line)

    def _update_best(self, model: RegressorMixin, run_info: dict) -> None:
        if self.best is None or run_info["rmse"] < self.best["rmse"]:
            self.best = run_info
            self.best_model = model

    def _save_results(self) -> None:
        df = pd.DataFrame(self.results).sort_values("rmse")
        df.to_csv("reports/metrics.csv", index=False)

    def _save_best_model(self) -> None:
        with open("models/best_model.pkl", "wb") as f:
            pickle.dump(self.best_model, f)
        with open("models/best_model_metadata.json", "w", encoding="utf-8") as f:
            json.dump(self.best, f, indent=4)

    def run(self) -> None:
        y_train, y_test = self._load_targets()

        for config in get_models():
            self._train_one(config, y_train, y_test)

        self._save_results()
        self._save_best_model()
        register_best_model(self.best["run_id"])

        print(f"\nMeilleur modele : {self.best['model']} "
              f"(test RMSE = {self.best['rmse']:.4f})")


def train_and_track_models() -> None:
    ModelTrainer().run()


if __name__ == "__main__":
    train_and_track_models()