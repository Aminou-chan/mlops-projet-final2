# ==================== IMPORTS ====================
import os
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


from dotenv import load_dotenv

# Charger les variables d'environnement
load_dotenv()


# ==================== CONFIGURATION ====================
# Nom de l'expérience MLflow
EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "california_housing_regression")

# URI du tracking MLflow (par défaut SQLite local)
TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")

# Nombre de folds pour la cross-validation
CV_FOLDS = 5

# Métrique utilisée pour RandomizedSearchCV
# Négatif car RandomizedSearchCV maximise la métrique
SCORING = "neg_root_mean_squared_error"

# Seed pour reproductibilité
RANDOM_STATE = 42


# ==================== CLASSES DE CONFIGURATION ====================


@dataclass
class ModelConfig:
    """
    Configuration d'un modèle à tester lors du training.

    Encapsule toutes les informations nécessaires pour entraîner et évaluer
    un modèle : son estimateur, ses hyperparamètres, et le pipeline de features
    à utiliser.

    Attributes:
        name (str): Nom identifiant du modèle (ex: "random_forest")
        estimator (RegressorMixin): Instance d'estimateur scikit-learn
        features (str): Type de features à utiliser ("linear" ou "tree")
        param_distributions (dict): Espace de recherche pour RandomizedSearchCV
                                   {} si pas de tuning (ex: LinearRegression)
        n_iter (int): Nombre de combinaisons à tester dans RandomizedSearchCV

    Example:
        >>> config = ModelConfig(
        ...     name="random_forest",
        ...     estimator=RandomForestRegressor(random_state=42, n_jobs=-1),
        ...     features="tree",
        ...     param_distributions={'n_estimators': randint(100, 400)},
        ...     n_iter=10
        ... )
    """

    name: str
    estimator: RegressorMixin
    features: str  # "linear" ou "tree"
    param_distributions: dict = field(default_factory=dict)  # {} = pas de tuning
    n_iter: int = 10  # nombre de combinaisons tirées au hasard

    @property
    def pipeline(self) -> str:
        """
        Retourne le nom du pipeline en fonction du type de features.

        Returns:
            str: "linear_pipeline" si linear, "tree_pipeline" sinon
        """
        return "linear_pipeline" if self.features == "linear" else "tree_pipeline"


# ==================== FONCTION D'INITIALISATION DES MODÈLES ====================


def get_models() -> list[ModelConfig]:
    """
    Retourne la liste des configurations de modèles à comparer.

    Trois modèles sont testés :

    1. Linear Regression (pas de tuning)
       - Baseline simple, utilisé avec features linéaires normalisées
       - Pas d'hyperparamètres à ajuster (pas de RandomizedSearchCV)

    2. Random Forest (tuning par RandomizedSearchCV)
       - Modèle tree-based, utilisé avec features brutes
       - Hyperparamètres : n_estimators, max_depth, min_samples_split, min_samples_leaf
       - 10 combinaisons testées en 5-fold CV

    3. Gradient Boosting (tuning par RandomizedSearchCV)
       - Modèle tree-based boosting, utilisé avec features brutes
       - Hyperparamètres : n_estimators, learning_rate, max_depth, subsample
       - 10 combinaisons testées en 5-fold CV

    Returns:
        list[ModelConfig]: Liste de 3 ModelConfig prêts à être entraînés

    Note:
        Chaque modèle correspond à un pipeline :
        - Linear Regression → linear_pipeline (features normalisées)
        - Random Forest → tree_pipeline (features brutes)
        - Gradient Boosting → tree_pipeline (features brutes)
    """
    return [
        # ===== MODÈLE 1 : Régression Linéaire =====
        ModelConfig(
            name="linear_regression",
            estimator=LinearRegression(),
            features="linear",
            # Pas de param_distributions = pas de tuning
        ),
        # ===== MODÈLE 2 : Random Forest =====
        ModelConfig(
            name="random_forest",
            estimator=RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1),
            features="tree",
            param_distributions={
                "n_estimators": randint(100, 400),  # nombre d'arbres
                "max_depth": [10, 20, 30, None],  # profondeur max des arbres
                "min_samples_split": randint(
                    2, 10
                ),  # min samples pour splitter un nœud
                "min_samples_leaf": randint(1, 5),  # min samples pour feuille
            },
            n_iter=10,
        ),
        # ===== MODÈLE 3 : Gradient Boosting =====
        ModelConfig(
            name="gradient_boosting",
            estimator=GradientBoostingRegressor(random_state=RANDOM_STATE),
            features="tree",
            param_distributions={
                "n_estimators": randint(100, 400),  # nombre de boosting stages
                "learning_rate": uniform(0.01, 0.2),  # taux d'apprentissage
                "max_depth": randint(2, 6),  # profondeur max des arbres
                "subsample": uniform(0.7, 0.3),  # fraction de samples pour chaque arbre
            },
            n_iter=10,
        ),
    ]


# ==================== CLASSE PRINCIPALE D'ENTRAÎNEMENT ====================


class ModelTrainer:
    """
    Classe orchestratrice pour entraîner et comparer plusieurs modèles.

    Responsabilités:
    1. Charger les données d'entraînement et test
    2. Entraîner chaque modèle (avec ou sans tuning)
    3. Évaluer les performances sur le test set
    4. Tracker toutes les informations dans MLflow
    5. Identifier le meilleur modèle
    6. Sauvegarder le meilleur modèle et ses métadonnées

    Attributes:
        results (list[dict]): Liste des résultats d'entraînement pour chaque modèle
        best (dict): Dictionnaire du meilleur run (contient métriques et params)
        best_model (RegressorMixin): Meilleur modèle entraîné
    """

    def __init__(self) -> None:
        """
        Initialise l'entraîneur.

        Crée les répertoires nécessaires et configure MLflow pour le tracking.
        """
        # Créer les répertoires de sortie s'ils n'existent pas
        Path("models").mkdir(exist_ok=True)
        Path("reports").mkdir(exist_ok=True)

        # Configurer MLflow
        self._setup_mlflow()

        # Initialiser les conteneurs de résultats
        self.results: list[dict] = []  # Liste de tous les runs
        self.best: dict | None = None  # Info du meilleur run
        self.best_model: RegressorMixin | None = None  # Instance du meilleur modèle

    @staticmethod
    def _setup_mlflow() -> None:
        """
        Configure MLflow pour le tracking.

        Défini l'URI de tracking et l'expérience à utiliser.
        """
        # Définir où MLflow doit stocker les métadonnées
        mlflow.set_tracking_uri(TRACKING_URI)

        # Créer/sélectionner l'expérience
        mlflow.set_experiment(EXPERIMENT_NAME)

    @staticmethod
    def _load_targets() -> tuple[pd.Series, pd.Series]:
        """
        Charge les cibles (y) d'entraînement et de test.

        Returns:
            tuple[pd.Series, pd.Series]: (y_train, y_test)
        """
        # Charger les fichiers CSV et convertir en Series
        y_train = pd.read_csv("data/processed/y_train.csv").squeeze()
        y_test = pd.read_csv("data/processed/y_test.csv").squeeze()
        return y_train, y_test

    @staticmethod
    def _load_features(features: str) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Charge les features d'entraînement et de test selon le type.

        Args:
            features (str): "linear" ou "tree"

        Returns:
            tuple[pd.DataFrame, pd.DataFrame]: (X_train, X_test) appropriés
        """
        # Charger les fichiers correspondant au type de pipeline
        X_train = pd.read_csv(f"data/processed/X_train_{features}.csv")
        X_test = pd.read_csv(f"data/processed/X_test_{features}.csv")
        return X_train, X_test

    @staticmethod
    def _log_trials(config: ModelConfig, search: RandomizedSearchCV) -> None:
        """
        Logue chaque combinaison testée comme un run enfant dans MLflow.

        Crée un run MLflow imbriqué pour chaque combinaison d'hyperparamètres
        testée pendant la RandomizedSearchCV, pour traçabilité complète.

        Args:
            config (ModelConfig): Configuration du modèle
            search (RandomizedSearchCV): Objet search avec les résultats CV
        """
        # Extraire les paramètres et scores de chaque essai
        params_list = search.cv_results_["params"]
        scores = search.cv_results_["mean_test_score"]

        # Créer un run enfant pour chaque combinaison d'hyperparamètres
        for params, score in zip(params_list, scores):
            with mlflow.start_run(run_name=f"{config.name}_trial", nested=True):
                # Logger les hyperparamètres testés
                mlflow.log_params(params)
                # Logger le score CV (transformé en RMSE positif)
                mlflow.log_metric("cv_rmse", -score)

    def _fit_model(self, config, X_train, y_train):
        """
        Entraîne un modèle avec ou sans tuning d'hyperparamètres.

        Si config.param_distributions est vide, entraîne directement l'estimateur.
        Sinon, effectue une RandomizedSearchCV pour trouver les meilleurs hyperparamètres.

        Args:
            config (ModelConfig): Configuration du modèle
            X_train (pd.DataFrame): Features d'entraînement
            y_train (pd.Series): Cible d'entraînement

        Returns:
            tuple: (modèle_entraîné, meilleurs_params, rmse_cv ou None)
        """
        # Cas 1 : Pas d'hyperparamètres à chercher (ex: LinearRegression)
        if not config.param_distributions:
            # Entraîner directement l'estimateur
            config.estimator.fit(X_train, y_train)
            # Retourner le modèle, ses params par défaut, et None pour CV RMSE
            return config.estimator, config.estimator.get_params(), None

        # Cas 2 : Recherche aléatoire d'hyperparamètres
        search = RandomizedSearchCV(
            config.estimator,
            config.param_distributions,
            n_iter=config.n_iter,  # Nombre de combinaisons à tester
            cv=CV_FOLDS,  # Cross-validation k-folds
            scoring=SCORING,  # Métrique d'optimisation
            random_state=RANDOM_STATE,  # Reproductibilité
            n_jobs=-1,  # Utiliser tous les cores
        )

        # Exécuter la recherche
        search.fit(X_train, y_train)

        # Logger tous les essais comme runs enfants
        self._log_trials(config, search)

        # Extraire le score CV du meilleur essai (en RMSE positif)
        cv_rmse = float(-search.best_score_)

        # Retourner le meilleur modèle trouvé, ses params, et son score CV
        return search.best_estimator_, search.best_params_, cv_rmse

    def _train_one(self, config: ModelConfig, y_train, y_test) -> None:
        """
        Entraîne UNun modèle complet, l'évalue et le log dans MLflow.

        Processus:
        1. Charger les features appropriées
        2. Entraîner le modèle (avec ou sans tuning)
        3. Évaluer sur le test set
        4. Logger tous les infos dans MLflow
        5. Mettre à jour le meilleur modèle si nécessaire

        Args:
            config (ModelConfig): Configuration du modèle à entraîner
            y_train (pd.Series): Cible d'entraînement
            y_test (pd.Series): Cible de test
        """
        # Charger les features appropriées (linear ou tree)
        X_train, X_test = self._load_features(config.features)

        # Créer un run MLflow parent pour ce modèle
        with mlflow.start_run(run_name=config.name) as run:
            # Entraîner le modèle
            model, best_params, cv_rmse = self._fit_model(config, X_train, y_train)

            # Faire les prédictions sur le test set
            y_pred = model.predict(X_test)

            # Calculer les métriques de test
            metrics = evaluate_model(y_test, y_pred)

            # Logger tous les infos dans MLflow
            mlflow.log_param("model_name", config.name)
            mlflow.log_param("pipeline", config.pipeline)
            mlflow.log_params(best_params)
            if cv_rmse is not None:
                mlflow.log_metric("cv_rmse", cv_rmse)
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(model, name="model")

            # Préparer un dictionnaire avec tous les résultats
            # (utiliser json.dumps avec default=str pour éviter les problèmes de sérialisation)
            run_info = {
                "run_id": run.info.run_id,
                "model": config.name,
                "pipeline": config.pipeline,
                "cv_rmse": cv_rmse,
                **metrics,  # Ajouter RMSE, MAE, R²
                "best_params": json.dumps(best_params, default=str),
            }

        # Ajouter aux résultats globaux
        self.results.append(run_info)

        # Mettre à jour le meilleur modèle si celui-ci est meilleur
        self._update_best(model, run_info)

        # Afficher un résumé de cet entraînement
        line = f"{config.name:20s} test RMSE = {metrics['rmse']:.4f}"
        if cv_rmse is not None:
            line += f" | CV RMSE = {cv_rmse:.4f}"
        print(line)

    def _update_best(self, model: RegressorMixin, run_info: dict) -> None:
        """
        Vérifie si ce run est meilleur que le meilleur trouvé et met à jour si c'est le cas.

        La métrique de comparaison est le RMSE sur le test set.

        Args:
            model (RegressorMixin): Modèle entraîné
            run_info (dict): Informations du run (incluant RMSE et autres métriques)
        """
        # Si c'est le premier modèle, ou s'il est meilleur que celui sauvegardé
        if self.best is None or run_info["rmse"] < self.best["rmse"]:
            # Mettre à jour le meilleur
            self.best = run_info
            self.best_model = model

    def _save_results(self) -> None:
        """
        Exporte tous les résultats d'entraînement en fichier CSV.

        Les résultats sont triés par RMSE (meilleur en premier).
        Sauvegarde dans reports/metrics.csv
        """
        # Créer un DataFrame à partir des résultats
        df = pd.DataFrame(self.results).sort_values("rmse")

        # Exporter en CSV
        df.to_csv("reports/metrics.csv", index=False)

    def _save_best_model(self) -> None:
        """
        Sauvegarde le meilleur modèle et ses métadonnées.

        Exporte:
        - Le modèle sérialisé en pickle → models/best_model.pkl
        - Les métadonnées du run (params, métriques) → models/best_model_metadata.json
        """
        # Sauvegarder le modèle en pickle
        with open("models/best_model.pkl", "wb") as f:
            pickle.dump(self.best_model, f)

        # Sauvegarder les métadonnées en JSON
        with open("models/best_model_metadata.json", "w", encoding="utf-8") as f:
            json.dump(self.best, f, indent=4)

    def run(self) -> None:
        """
        Lance l'entraînement complet de tous les modèles.

        Processus:
        1. Charger les cibles (y_train, y_test)
        2. Pour chaque modèle :
           - Entraîner et évaluer
           - Logger dans MLflow
           - Mettre à jour le meilleur
        3. Sauvegarder les résultats et le meilleur modèle
        """
        # Charger les cibles
        y_train, y_test = self._load_targets()

        # Entraîner chaque modèle
        for config in get_models():
            self._train_one(config, y_train, y_test)

        # Vérifier qu'au moins un modèle a été entraîné
        assert self.best is not None

        self._save_results()
        self._save_best_model()
        register_best_model(self.best["run_id"])

        print(
            f"\nMeilleur modele : {self.best['model']} "
            f"(test RMSE = {self.best['rmse']:.4f})"
        )


def train_and_track_models() -> None:
    ModelTrainer().run()


if __name__ == "__main__":
    train_and_track_models()
