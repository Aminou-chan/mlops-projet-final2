import os
from pathlib import Path
import pickle

import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

from dotenv import load_dotenv

load_dotenv()

MODEL_PATH = Path(os.getenv("MODEL_PATH", "models/best_model.pkl"))

# Ordre des features = ordre des colonnes utilisees a l'entrainement.
FEATURE_ORDER = [
    "MedInc", "HouseAge", "AveRooms", "AveBedrms",
    "Population", "AveOccup", "Latitude", "Longitude",
]


class HouseFeatures(BaseModel):
    MedInc: float
    HouseAge: float
    AveRooms: float
    AveBedrms: float
    Population: float
    AveOccup: float
    Latitude: float
    Longitude: float


class Prediction(BaseModel):
    med_house_val: float        # cible brute predite (en 100k $)
    predicted_price_usd: float  # meme valeur convertie en dollars


# On charge le modele une seule fois, au demarrage de l'API.
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

app = FastAPI(title="ImmoPrix - API de prediction de prix")


@app.get("/")
def root():
    return {"message": "API ImmoPrix operationnelle. Documentation sur /docs."}


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/predict", response_model=Prediction)
def predict(features: HouseFeatures):
    # On reconstruit un DataFrame avec les bonnes colonnes dans le bon ordre,
    # pour que le modele retrouve exactement le schema vu a l'entrainement.
    X = pd.DataFrame([features.model_dump()])[FEATURE_ORDER]

    prediction = float(model.predict(X)[0])
    return Prediction(
        med_house_val=prediction,
        predicted_price_usd=prediction * 100_000,
    )