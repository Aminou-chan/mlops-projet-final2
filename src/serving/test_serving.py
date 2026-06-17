"""Test du serving MLflow.

Prerequis : lancer le serveur dans un autre terminal :

    mlflow models serve -m "models:/california_housing_best_model@champion" \
        -p 5001 --env-manager local

Ce script envoie un (ou plusieurs) echantillon(s) au endpoint /invocations
au format `dataframe_split` et affiche la prediction renvoyee.
"""

import os

import requests
from dotenv import load_dotenv

load_dotenv()

# URL du serveur MLflow (configurable via .env)
SERVING_URL = os.getenv("MLFLOW_SERVING_URL", "http://127.0.0.1:5001/invocations")

# Ordre des features attendu par le modele (California Housing)
FEATURE_ORDER = [
    "MedInc", "HouseAge", "AveRooms", "AveBedrms",
    "Population", "AveOccup", "Latitude", "Longitude",
]

SAMPLES = [
    [8.3252, 41.0, 6.984, 1.024, 322.0, 2.556, 37.88, -122.23],
    [3.2596, 33.0, 5.017, 1.006, 2300.0, 3.691, 34.05, -118.24],
]


def predict(samples: list[list[float]]) -> list[float]:
    """Envoie les echantillons au serveur MLflow et renvoie les predictions."""
    payload = {
        "dataframe_split": {
            "columns": FEATURE_ORDER,
            "data": samples,
        }
    }
    response = requests.post(SERVING_URL, json=payload, timeout=10)
    response.raise_for_status()
    return response.json()["predictions"]


def main() -> None:
    predictions = predict(SAMPLES)
    print(f"Serveur : {SERVING_URL}\n")
    for sample, pred in zip(SAMPLES, predictions):
        # la cible est en centaines de milliers de dollars
        print(f"  Features : {sample}")
        print(f"  -> Prediction : {pred:.4f}  (~ {pred * 100_000:,.0f} $)\n")


if __name__ == "__main__":
    main()