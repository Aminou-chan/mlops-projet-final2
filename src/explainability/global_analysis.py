import pickle
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # backend non-interactif : sauvegarde sans afficher
import matplotlib.pyplot as plt
import pandas as pd

FIGURES_DIR = Path("reports/figures")


class GlobalFeatureAnalyzer:
    def __init__(self):
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)

        self.model = self._load_model()
        self.metadata = self._load_metadata()

    @staticmethod
    def _load_model():
        with open("models/best_model.pkl", "rb") as f:
            return pickle.load(f)

    @staticmethod
    def _load_metadata():
        return pd.read_json("models/best_model_metadata.json", typ="series")

    def _load_features(self):
        pipeline = self.metadata["pipeline"]

        if pipeline == "linear_pipeline":
            return pd.read_csv("data/processed/X_train_linear.csv")

        return pd.read_csv("data/processed/X_train_tree.csv")

    def compute_global_importance(self):
        X = self._load_features()

        if hasattr(self.model, "feature_importances_"):
            importance = self.model.feature_importances_

        elif hasattr(self.model, "coef_"):
            importance = abs(self.model.coef_)

        else:
            raise ValueError("Model does not support feature importance.")

        result = pd.DataFrame({
            "feature": X.columns,
            "importance": importance,
        })

        result = result.sort_values(by="importance", ascending=False)

        result.to_csv("reports/explainability/global_importance.csv", index=False)
        self._plot_importance(result)

        print(result)

        return result

    @staticmethod
    def _plot_importance(result):
        # On trie en ordre croissant pour que la plus grande barre soit en haut.
        ordered = result.sort_values(by="importance")

        plt.figure(figsize=(8, 5))
        plt.barh(ordered["feature"], ordered["importance"])
        plt.title("Importances natives des features")
        plt.xlabel("Importance")
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "global_importance.png", dpi=150)
        plt.close()