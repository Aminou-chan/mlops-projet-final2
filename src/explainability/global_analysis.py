import pickle
from pathlib import Path

import matplotlib

# Configurer matplotlib pour utiliser un backend non-interactif
# Permet de générer des graphiques sans interface graphique (serveur)
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


# Répertoire où seront sauvegardés les figures et graphiques
FIGURES_DIR = Path("reports/figures")


class GlobalFeatureAnalyzer:
    """
    Classe pour analyser l'importance globale des features d'un modèle ML.

    Cette classe charge un modèle entraîné et ses métadonnées, puis calcule
    l'importance de chaque feature utilisée lors de l'entraînement.

    Elle supporte deux types de modèles:
    - Modèles avec `feature_importances_` (arbres, random forests, etc.)
    - Modèles avec `coef_` (modèles linéaires)

    Attributes:
        model: Modèle ML chargé depuis pickle
        metadata (pd.Series): Métadonnées du modèle (pipeline utilisé, etc.)
    """

    def __init__(self):
        """
        Initialise l'analyseur en créant les répertoires nécessaires
        et en chargeant le modèle et ses métadonnées.
        """
        # Créer le répertoire des figures s'il n'existe pas
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)

        # Charger le modèle entraîné depuis un fichier pickle
        self.model = self._load_model()

        # Charger les métadonnées du modèle (contient le pipeline utilisé, etc.)
        self.metadata = self._load_metadata()

    @staticmethod
    def _load_model():
        """
        Charge le modèle entraîné depuis un fichier pickle.

        Returns:
            object: Modèle ML désérialisé
        """
        with open("models/best_model.pkl", "rb") as f:
            return pickle.load(f)

    @staticmethod
    def _load_metadata():
        """
        Charge les métadonnées du modèle depuis un fichier JSON.

        Les métadonnées contiennent des informations sur le pipeline utilisé,
        les hyperparamètres, les performances, etc.

        Returns:
            pd.Series: Série pandas contenant les métadonnées du modèle
        """
        return pd.read_json("models/best_model_metadata.json", typ="series")

    def _load_features(self):
        """
        Charge les features d'entraînement en fonction du pipeline utilisé.

        Le pipeline utilisé (linéaire ou tree-based) détermine les features
        qui ont été sélectionnées lors de l'entraînement. Cette méthode
        charge les bonnes données prétraitées.

        Returns:
            pd.DataFrame: DataFrame contenant les features d'entraînement
        """
        # Récupérer le type de pipeline depuis les métadonnées
        pipeline = self.metadata["pipeline"]

        # Charger les features appropriées selon le pipeline
        if pipeline == "linear_pipeline":
            return pd.read_csv("data/processed/X_train_linear.csv")

        return pd.read_csv("data/processed/X_train_tree.csv")

    def compute_global_importance(self):
        """
        Calcule et exporte l'importance globale de chaque feature.

        Cette méthode:
        1. Charge les features d'entraînement
        2. Extrait les importances du modèle (selon son type)
        3. Crée un DataFrame avec les resultats
        4. Trie les features par importance décroissante
        5. Exporte les resultats en CSV
        6. Génère une visualisation
        7. Affiche les resultats

        Returns:
            pd.DataFrame: DataFrame avec colonnes 'feature' et 'importance',
                         trié par importance décroissante

        Raises:
            ValueError: Si le modèle ne supporte pas feature_importances_
        """
        # Charger les features d'entraînement
        X = self._load_features()

        # Extraire les importances selon le type de modèle
        if hasattr(self.model, "feature_importances_"):
            # Modèles tree-based (DecisionTree, RandomForest, etc.)
            importance = self.model.feature_importances_

        elif hasattr(self.model, "coef_"):
            # Modèles linéaires (LinearRegression, LogisticRegression, etc.)
            # Utiliser la valeur absolue car les coefficients peuvent être négatifs
            importance = abs(self.model.coef_)

        else:
            # Le modèle ne supporte pas l'extraction d'importances
            raise ValueError("Model does not support feature importance.")

        # Créer un DataFrame avec les features et leurs importances
        result = pd.DataFrame(
            {
                "feature": X.columns,
                "importance": importance,
            }
        )

        # Trier par importance décroissante (la plus importante en premier)
        result = result.sort_values(by="importance", ascending=False)

        # Exporter les resultats en fichier CSV
        result.to_csv("reports/explainability/global_importance.csv", index=False)

        # Générer et sauvegarder la visualisation
        self._plot_importance(result)

        # Afficher les resultats dans la console
        print(result)

        return result

    @staticmethod
    def _plot_importance(result):
        """
        Crée et sauvegarde une visualisation graphique des importances.

        Génère un graphique en barres horizontales (barh) montrant l'importance
        de chaque feature. Les features sont triées en ordre croissant pour que
        la barre la plus longue (la plus importante) soit au sommet.

        Args:
            result (pd.DataFrame): DataFrame contenant les colonnes 'feature' et 'importance'
        """
        # Trier en ordre croissant pour que la plus grande barre soit en haut
        ordered = result.sort_values(by="importance")

        # Créer une figure matplotlib
        plt.figure(figsize=(8, 5))

        # Créer un graphique en barres horizontales
        plt.barh(ordered["feature"], ordered["importance"])

        # Ajouter titre et labels
        plt.title("Importances natives des features")
        plt.xlabel("Importance")

        # Ajuster les marges pour éviter les coupures de texte
        plt.tight_layout()

        # Sauvegarder le graphique en fichier PNG
        plt.savefig(FIGURES_DIR / "global_importance.png", dpi=150)

        # Fermer la figure pour libérer la mémoire
        plt.close()
