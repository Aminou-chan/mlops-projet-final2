"""
Module de test du serving MLflow.

Objectif:
- Tester que le modèle déployé via MLflow répond correctement
- Envoyer des requêtes au endpoint REST du serveur MLflow
- Vérifier que les prédictions sont raisonnables

Prérequis:
- Le serveur MLflow doit être lancé dans un terminal séparé

Setup du serveur:
    mlflow models serve -m "models:/california_housing_best_model@champion" \\
        -p 5001 --env-manager local

Ce script envoie ensuite des requêtes de prédiction au endpoint /invocations.
"""

# ==================== IMPORTS ====================
import os
from typing import Any

import requests
from dotenv import load_dotenv

# Charger les variables d'environnement
load_dotenv()

# ==================== CONFIGURATION ====================
# URL du serveur MLflow (configurable via .env)
# Format: http://[host]:[port]/invocations
SERVING_URL = os.getenv("MLFLOW_SERVING_URL", "http://127.0.0.1:5001/invocations")

# Ordre des features attendu par le modèle (IMPORTANT: doit correspondre à l'entraînement)
# Si les colonnes ne sont pas dans le bon ordre, les prédictions seront incorrectes
FEATURE_ORDER = [
    "MedInc",
    "HouseAge",
    "AveRooms",
    "AveBedrms",
    "Population",
    "AveOccup",
    "Latitude",
    "Longitude",
]

# Deux exemples de maisons à prédire
# Chaque liste contient les 8 features dans l'ordre spécifié par FEATURE_ORDER
SAMPLES = [
    # Exemple 1 : Maison côtière californienne (San Francisco area)
    # Revenu haut, maison vieille, beaucoup de pièces, prix probablement haut
    [8.3252, 41.0, 6.984, 1.024, 322.0, 2.556, 37.88, -122.23],
    # Exemple 2 : Maison du sud de la Californie (Los Angeles area)
    # Revenu moyen, maison ancienne, moins de pièces, prix probablement moyen
    [3.2596, 33.0, 5.017, 1.006, 2300.0, 3.691, 34.05, -118.24],
]


# ==================== FONCTION DE PRÉDICTION ====================


def predict(samples: list[list[float]]) -> list[float]:
    """
    Envoie les échantillons au serveur MLflow et récupère les prédictions.

    Format de requête (dataframe_split):
    - Permet de passer les données sous forme tabulaire
    - Le serveur reconstruit un DataFrame Pandas à partir des données
    - Envoie au endpoint REST /invocations

    Args:
        samples (list[list[float]]): Liste d'échantillons, chacun étant
                                     une liste de 8 valeurs de features
                                     (dans l'ordre FEATURE_ORDER)

    Returns:
        list[float]: Liste des prédictions (une par échantillon)
                     Valeurs en unités de 100k$ (ex: 2.5 = 250 000$)

    Raises:
        requests.exceptions.RequestException: Si la requête échoue
        (erreur réseau, serveur non disponible, etc.)

    Example:
        >>> samples = [[8.3, 41.0, 6.9, 1.0, 322.0, 2.6, 37.9, -122.2]]
        >>> predictions = predict(samples)
        >>> print(predictions)
        [4.25]  # Maison estimée à 425 000$
    """
    # Construire le payload au format dataframe_split
    # (format standard de MLflow pour les requêtes REST)
    payload: dict[str, Any] = {
        "dataframe_split": {
            "columns": FEATURE_ORDER,  # Noms des colonnes (pour vérification)
            "data": samples,  # Valeurs des features
        }
    }

    # Envoyer la requête POST au serveur MLflow
    response = requests.post(SERVING_URL, json=payload, timeout=10)

    # Lever une exception si la requête a échoué (status != 2xx)
    response.raise_for_status()

    # Extraire les prédictions de la réponse JSON
    return response.json()["predictions"]


# ==================== FONCTION PRINCIPALE ====================


def main() -> None:
    """
    Point d'entrée du module : effectue des prédictions de test.

    Processus:
    1. Envoyer les échantillons prédéfinis au serveur
    2. Récupérer les prédictions
    3. Afficher les résultats avec formatage

    Affiche:
    - URL du serveur
    - Pour chaque échantillon :
      * Les features en entrée
      * La prédiction brute (en 100k$)
      * La prédiction en dollars USD
    """
    # Envoyer les échantillons et récupérer les prédictions
    predictions = predict(SAMPLES)

    # Afficher l'en-tête
    print(f"Serveur : {SERVING_URL}\n")

    # Afficher les résultats pour chaque échantillon
    for sample, pred in zip(SAMPLES, predictions):
        # Afficher les features d'entrée
        print(f"  Features : {sample}")

        # Afficher la prédiction
        # pred est en unités de 100 000$ (ex: 4.25 = 425 000$)
        print(f"  -> Prediction : {pred:.4f}  (~ {pred * 100_000:,.0f} $)\n")


if __name__ == "__main__":
    main()
