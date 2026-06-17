import pickle
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import shap

FIGURES_DIR = Path("reports/figures")
SAMPLE_SIZE = 500
RANDOM_STATE = 42


class LocalFeatureAnalyzer:
    def __init__(self):
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)

        self.model = self._load_model()
        self.metadata = self._load_metadata()

        self.sample = self._load_sample()
        explainer = shap.TreeExplainer(self.model)
        self.shap_values = explainer(self.sample)

    @staticmethod
    def _load_model():
        with open("models/best_model.pkl", "rb") as f:
            return pickle.load(f)

    @staticmethod
    def _load_metadata():
        return pd.read_json("models/best_model_metadata.json", typ="series")

    def _load_sample(self):
        if self.metadata["pipeline"] == "linear_pipeline":
            X = pd.read_csv("data/processed/X_test_linear.csv")
        else:
            X = pd.read_csv("data/processed/X_test_tree.csv")

        n = min(SAMPLE_SIZE, len(X))
        return X.sample(n=n, random_state=RANDOM_STATE).reset_index(drop=True)

    def plot_beeswarm(self):
        shap.plots.beeswarm(self.shap_values, show=False)
        plt.title("Impacts SHAP (beeswarm)")
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "shap_beeswarm.png", dpi=150, bbox_inches="tight")
        plt.close()

    def explain_sample(self, index=0):
        shap.plots.waterfall(self.shap_values[index], show=False)
        plt.title(f"Impact local SHAP - exemple {index}")
        plt.tight_layout()
        plt.savefig(
            FIGURES_DIR / f"shap_local_example_{index}.png",
            dpi=150,
            bbox_inches="tight",
        )
        plt.close()

        print(self.sample.iloc[[index]])

    def explain_samples(self, n=3):
        for i in range(n):
            self.explain_sample(i)