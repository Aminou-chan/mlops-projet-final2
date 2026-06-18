"""
Task script pour récupérer et sauvegarder les données brutes.

Ce script importe la fonction `load_california_housing` du package
`src.data.load_data` et sauvegarde le dataset brut sur disque.

Usage:
    python tasks/load_data.py

Effets:
    - Télécharge le dataset California Housing
    - Sauvegarde le fichier dans `data/raw/california_housing.csv`
"""

import os

from src.data.load_data import load_california_housing

# Récupération des données brutes
df = load_california_housing()

# Création du dossier de destination si nécessaire, puis sauvegarde
os.makedirs("data/raw", exist_ok=True)
df.to_csv("data/raw/california_housing.csv", index=False)
print("Données brutes sauvegardées dans data/raw/california_housing.csv")