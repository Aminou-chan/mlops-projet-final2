import pickle
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import shap


# Répertoire où seront sauvegardés les figures et graphiques
FIGURES_DIR = Path("reports/figures")

# Taille de l'échantillon pour les explications SHAP (limiter pour performance)
SAMPLE_SIZE = 500

# Seed pour reproductibilité du tirage aléatoire
RANDOM_STATE = 42


class LocalFeatureAnalyzer:
    """
    Classe pour analyser et expliquer les prédictions locales d'un modèle ML.

    Utilise la bibliothèque SHAP (SHapley Additive exPlanations) pour expliquer
    les décisions individuelles du modèle. Permet de comprendre pourquoi le modèle
    a fait une prédiction spécifique pour une donnée particulière.

    Features principales:
    - Génération de valeurs SHAP pour un échantillon de test
    - Visualisation globale (beeswarm plot) des impacts SHAP
    - Explications détaillées (waterfall plot) pour des prédictions individuelles

    Attributes:
        model: Modèle ML chargé depuis pickle
        metadata (pd.Series): Métadonnées du modèle
        sample (pd.DataFrame): Échantillon de données pour l'explication locale
        shap_values: Valeurs SHAP calculées pour l'échantillon
    """

    def __init__(self):
        """
        Initialise l'analyseur local.

        Processus:
        1. Crée le répertoire des figures
        2. Charge le modèle entraîné
        3. Charge les métadonnées
        4. Charge un échantillon de données de test
        5. Calcule les valeurs SHAP pour cet échantillon

        Note:
            Le calcul des valeurs SHAP peut être gourmand en temps/mémoire.
            La taille d'échantillon est limitée (SAMPLE_SIZE) pour la performance.
        """
        # Créer le répertoire des figures s'il n'existe pas
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)

        # Charger le modèle entraîné depuis un fichier pickle
        self.model = self._load_model()

        # Charger les métadonnées du modèle
        self.metadata = self._load_metadata()

        # Charger un échantillon de données de test
        self.sample = self._load_sample()

        # Créer un explainer SHAP adapté au type de modèle (tree-based)
        explainer = shap.TreeExplainer(self.model)

        # Calculer les valeurs SHAP pour l'échantillon (peut prendre du temps)
        self.shap_values = explainer(self.sample)

    @staticmethod
    def _load_model():
        """
        Charge le modèle entraîné depuis un fichier pickle.

        Returns:
            object: Modèle ML désérialisé (généralement un modèle tree-based)
        """
        with open("models/best_model.pkl", "rb") as f:
            return pickle.load(f)

    @staticmethod
    def _load_metadata():
        """
        Charge les métadonnées du modèle depuis un fichier JSON.

        Les métadonnées contiennent notamment le type de pipeline utilisé
        (linear_pipeline ou tree_pipeline).

        Returns:
            pd.Series: Série pandas contenant les métadonnées du modèle
        """
        return pd.read_json("models/best_model_metadata.json", typ="series")

    def _load_sample(self):
        """
        Charge un échantillon aléatoire de données de test.

        Sélectionne les features appropriées selon le pipeline utilisé
        (linéaire ou tree-based) et tire un échantillon aléatoire limité
        en taille pour améliorer les performances du calcul SHAP.

        Returns:
            pd.DataFrame: Échantillon de données (max SAMPLE_SIZE lignes)
        """
        # Charger les données de test appropriées selon le pipeline
        if self.metadata["pipeline"] == "linear_pipeline":
            X = pd.read_csv("data/processed/X_test_linear.csv")
        else:
            X = pd.read_csv("data/processed/X_test_tree.csv")

        # Limiter la taille de l'échantillon pour des raisons de performance
        n = min(SAMPLE_SIZE, len(X))

        # Tirer un échantillon aléatoire avec reproductibilité
        return X.sample(n=n, random_state=RANDOM_STATE).reset_index(drop=True)

    def plot_beeswarm(self):
        """
        Crée et sauvegarde un graphique beeswarm SHAP global.

        Le graphique beeswarm montre pour chaque feature :
        - En abscisse : l'impact SHAP (contribution à la prédiction)
        - En ordonnée : l'ordre des features par importance
        - En couleur : la valeur de la feature (rouge = valeur haute, bleu = valeur basse)

        Ce graphique permet de voir l'importance globale des features et la
        direction de leurs impacts sur le modèle.

        Exporte le graphique en PNG.
        """
        # Créer le graphique beeswarm SHAP (sans affichage interactif)
        shap.plots.beeswarm(self.shap_values, show=False)

        # Ajouter un titre
        plt.title("Impacts SHAP (beeswarm)")

        # Ajuster les marges
        plt.tight_layout()

        # Sauvegarder en fichier PNG
        plt.savefig(FIGURES_DIR / "shap_beeswarm.png", dpi=150, bbox_inches="tight")

        # Fermer la figure pour libérer la mémoire
        plt.close()

    def explain_sample(self, index=0):
        """
        Génère et exporte une explication détaillée (waterfall plot) pour une prédiction.

        Le graphique waterfall montre :
        - La valeur de base (base value) du modèle
        - Pour chaque feature, son impact positif ou négatif sur la prédiction
        - Empilées jusqu'à la prédiction finale

        Permet de comprendre exactement comment chaque feature a contribué
        à la prédiction finale pour cette instance spécifique.

        Args:
            index (int): Indice de la ligne dans l'échantillon à expliquer (défaut: 0)

        Exporte:
            - Un graphique waterfall en PNG
            - Les valeurs de la donnée en console
        """
        # Créer le graphique waterfall pour cet indice (sans affichage interactif)
        shap.plots.waterfall(self.shap_values[index], show=False)

        # Ajouter un titre avec l'indice
        plt.title(f"Impact local SHAP - exemple {index}")

        # Ajuster les marges
        plt.tight_layout()

        # Sauvegarder en fichier PNG avec indice unique
        plt.savefig(
            FIGURES_DIR / f"shap_local_example_{index}.png",
            dpi=150,
            bbox_inches="tight",
        )

        # Fermer la figure pour libérer la mémoire
        plt.close()

        # Afficher les valeurs d'entrée pour cette instance
        print(self.sample.iloc[[index]])

    def explain_samples(self, n=3):
        """
        Génère des explications détaillées (waterfall plots) pour plusieurs prédictions.

        Permet d'analyser rapidement les impacts locaux pour les n premiers
        exemples de l'échantillon, donnant une vue d'ensemble sur comment
        le modèle prend ses décisions sur différentes instances.

        Args:
            n (int): Nombre de prédictions à expliquer (défaut: 3)

        Exporte:
            - n graphiques waterfall en PNG (un par indice)
            - Les valeurs d'entrée pour chaque instance en console
        """
        # Boucler sur les n premiers indices de l'échantillon
        for i in range(n):
            # Expliquer chaque instance
            self.explain_sample(i)
