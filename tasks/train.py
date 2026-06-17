"""
Task script pour lancer l'entraînement et le suivi des modèles ML.

Ce script importe la fonction `train_and_track_models` du package `src.models.train`
et lance le workflow complet d'entraînement et d'évaluation.

Usage:
    python tasks/train.py

Effets:
    - Charge les données traitées de `data/processed/`
    - Entraîne les modèles définis dans `src.models.train`
    - Logue les runs dans MLflow
    - Sauvegarde le meilleur modèle et ses métadonnées
    - Génère un rapport des métriques dans `reports/metrics.csv`
"""

from src.models.train import train_and_track_models

# Exécution complète de l'entraînement et du tracking MLflow
train_and_track_models()
