import os
from pathlib import Path
import pickle

import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

from dotenv import load_dotenv

load_dotenv()

MODEL_PATH = Path(os.getenv("MODEL_PATH", "models/best_model.pkl"))

# Ordre des features = ordre exact des colonnes utilisées lors de l'entraînement du modèle.
# Cette correspondance est CRUCIALE pour que le modèle prédise correctement.
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


# MODÈLES DE DONNÉES (Pydantic)
# Ces classes définissent le schéma de validation des données entrantes et sortantes


class HouseFeatures(BaseModel):
    """
    Schéma de validation pour les features d'entrée.
    Représente les caractéristiques d'une maison à prédire.

    Attributs:
        MedInc (float): Revenu médian du quartier
        HouseAge (float): Âge de la maison
        AveRooms (float): Nombre moyen de pièces
        AveBedrms (float): Nombre moyen de chambres
        Population (float): Population du quartier
        AveOccup (float): Taux d'occupation moyen
        Latitude (float): Latitude géographique
        Longitude (float): Longitude géographique
    """

    MedInc: float
    HouseAge: float
    AveRooms: float
    AveBedrms: float
    Population: float
    AveOccup: float
    Latitude: float
    Longitude: float


class Prediction(BaseModel):
    """
    Schéma de validation pour la réponse de prédiction.
    Contient les prédictions sous deux formats.

    Attributs:
        med_house_val (float): Valeur médiane prédite (en unités de 100k $)
        predicted_price_usd (float): Même valeur convertie en dollars USD
    """

    med_house_val: float  # cible brute prédite (en 100k $)
    predicted_price_usd: float  # même valeur convertie en dollars


# On charge le modèle une seule fois, au démarrage de l'API.
# Cela évite de le recharger à chaque requête (gain de performance).
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

# Initialiser l'application FastAPI avec titre et description
app = FastAPI(title="ImmoPrix - API de prediction de prix")


@app.get("/")
def root():
    """
    Endpoint racine - endpoint de bienvenue.
    Retourne un message de confirmation que l'API est opérationnelle.

    Returns:
        dict: Message de bienvenue et lien vers la documentation
    """
    return {"message": "API ImmoPrix operationnelle. Documentation sur /docs."}


@app.get("/health")
def health():
    """
    Endpoint de vérification de santé (health check).
    Permet de vérifier que l'API et le modèle sont correctement chargés.
    Utile pour les systems de monitoring et les orchestrateurs (Docker, Kubernetes).

    Returns:
        dict: Statut de l'API et booléen indiquant si le modèle est chargé
    """
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/predict", response_model=Prediction)
def predict(features: HouseFeatures):
    """
    Endpoint de prédiction - endpoint principal de l'API.
    Récoit les caractéristiques d'une maison et retourne la prédiction de prix.

    Processus:
        1. Valide les features reçues via le schéma HouseFeatures
        2. Convertit les données en DataFrame Pandas
        3. Réordonne les colonnes dans l'ordre exact vu à l'entraînement (CRUCIAL)
        4. Effectue la prédiction avec le modèle
        5. Formate la réponse avec deux formats de prix

    Args:
        features (HouseFeatures): Dictionnaire contenant les 8 features nécessaires

    Returns:
        Prediction: Objet contenant la prédiction en format 100k$ et en USD
    """
    # On reconstruit un DataFrame avec les bonnes colonnes dans le bon ordre,
    # pour que le modèle retrouve exactement le schema vu à l'entraînement.
    X = pd.DataFrame([features.model_dump()])[FEATURE_ORDER]

    # Effectuer la prédiction (retourne un array, on prend le premier élément)
    prediction = float(model.predict(X)[0])

    # Retourner la prédiction dans les deux formats attendus
    return Prediction(
        med_house_val=prediction,
        predicted_price_usd=prediction * 100_000,
    )
