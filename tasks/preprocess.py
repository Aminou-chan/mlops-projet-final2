"""
Task script pour lancer le prétraitement des données.

Ce script importe la fonction `preprocess_data` du package `src.features.preprocessing`
et exécute le pipeline de prétraitement principal.

Usage:
    python tasks/preprocess.py

Effets:
    - Charge le dataset brut de `data/raw/california_housing.csv`
    - Crée les pipelines linéaire et tree
    - Sauvegarde les fichiers transformés dans `data/processed/`
    - Sauvegarde le scaler robuste dans `models/robust_scaler.pkl`
"""

from src.features.preprocessing import preprocess_data

# Exécution du prétraitement des données
preprocess_data()
